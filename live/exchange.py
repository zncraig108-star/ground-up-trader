"""live/exchange.py — ExchangeAdapter seam + Paper/Coinbase implementations.

Defines the async ``ExchangeAdapter`` interface used by the executor, a ``PaperAdapter``
that simulates fills against a price stream (used by the "Deploy to Paper" button and
the offline demo), and a ``CoinbaseCDPAdapter`` skeleton for real trading.

Fill model (mirrors the backtester — see ``core/backtester.py``):
  * post-only limit fills only when the market price **crosses through** the limit by at
    least one tick (buy: ``low <= limit - tick``; sell: ``high >= limit + tick``);
  * a post-only order that would cross the spread at placement is rejected;
  * stop-market orders trigger on touch and fill at the stop price (taker);
  * maker fee on maker fills, taker fee on stop/market fills.

``CoinbaseCDPAdapter`` requires the ``cdp-sdk`` and network access; both are optional and
the demo never uses it. Endpoints/schemas are unverified and marked accordingly.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List, Optional

import numpy as np

from config import FillModelConfig, InstrumentSpec

__all__ = [
    "Candle",
    "Position",
    "Order",
    "ExchangeAdapter",
    "PaperAdapter",
    "CoinbaseCDPAdapter",
    "candles_from_dataframe",
]


@dataclass
class Candle:
    """A single 1m OHLCV candle (timestamp in epoch milliseconds)."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Position:
    """An open position (signed quantity: long > 0, short < 0)."""

    qty: float
    entry_price: float


@dataclass
class Order:
    """A resting or filled order."""

    id: str
    side: str  # "buy" | "sell"
    type: str  # "post_only_limit" | "stop_market"
    price: float
    size: float
    status: str = "open"  # open | filled | cancelled


class ExchangeAdapter(ABC):
    """Async exchange interface. All methods are async; streams are async generators."""

    @abstractmethod
    async def connect(self) -> None:
        ...

    @abstractmethod
    def stream_candles(self) -> AsyncIterator[Candle]:
        ...

    @abstractmethod
    async def place_post_only_limit(self, side: str, price: float, size: float) -> Optional[str]:
        ...

    @abstractmethod
    async def place_stop_market(self, side: str, price: float, size: float) -> Optional[str]:
        ...

    @abstractmethod
    async def cancel_all(self) -> None:
        ...

    @abstractmethod
    async def market_close_all(self) -> None:
        ...

    @abstractmethod
    async def get_equity(self) -> float:
        ...

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        ...


async def candles_from_dataframe(df: Any) -> AsyncIterator[Candle]:
    """Yield ``Candle`` objects from a 1m OHLCV DataFrame (used to build a paper feed)."""
    for ts, row in df.iterrows():
        yield Candle(
            timestamp=int(ts.value // 10**6),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )


class PaperAdapter(ExchangeAdapter):
    """Simulates a live exchange against a candle feed (no network).

    Fills are applied on the feed candles in ``stream_candles``. The account is a signed
    quantity ``qty`` plus a realized ``cash`` balance; ``get_equity`` marks to market.
    """

    def __init__(self, instrument: InstrumentSpec, initial_equity: float = 10_000.0,
                 feed: Optional[AsyncIterator[Candle]] = None,
                 fill_model: Optional[FillModelConfig] = None) -> None:
        self.instrument = instrument
        self.fills = fill_model if fill_model is not None else FillModelConfig()
        self.initial_equity = float(initial_equity)
        self.cash = float(initial_equity)
        self.qty = 0.0
        self.entry_price = 0.0
        self.last_close: Optional[float] = None
        self.orders: List[Order] = []
        self.connected = False
        self.entry_fills = 0
        self.exit_fills = 0
        self.liquidations = 0
        self._ids = itertools.count(1)
        self._feed = feed if feed is not None else self._default_feed()

    # ------------------------------------------------------------------ stream
    async def connect(self) -> None:
        self.connected = True

    async def _default_feed(self) -> AsyncIterator[Candle]:
        """Deterministic random-walk feed used when none is injected (offline demo)."""
        rng = np.random.default_rng(12345)
        price = 100.0
        t = 0
        while True:
            ret = float(rng.normal(0.0, 0.001))
            o = price
            c = price * (1.0 + ret)
            hi = max(o, c) * (1.0 + abs(float(rng.normal(0.0, 0.0005))))
            lo = min(o, c) * (1.0 - abs(float(rng.normal(0.0, 0.0005))))
            yield Candle(timestamp=t, open=o, high=hi, low=lo, close=c, volume=1.0)
            price = c
            t += 60_000

    async def stream_candles(self) -> AsyncIterator[Candle]:
        if not self.connected:
            await self.connect()
        async for candle in self._feed:
            self._process(candle)
            yield candle

    # ---------------------------------------------------------------- simulation
    def _round_tick(self, price: float) -> float:
        tick = self.instrument.tick_size
        return np.floor(price / tick + 0.5) * tick

    def _process(self, candle: Candle) -> None:
        self.last_close = candle.close
        self._fill_stops(candle)
        self._fill_limits(candle)
        self._check_liquidation(candle)

    def _fill_stops(self, candle: Candle) -> None:
        slip = self.fills.stop_slippage_ticks * self.instrument.tick_size
        for order in list(self.orders):
            if order.status != "open" or order.type != "stop_market":
                continue
            triggered = (order.side == "sell" and candle.low <= order.price) or \
                        (order.side == "buy" and candle.high >= order.price)
            if triggered:
                fill_price = order.price - slip if order.side == "sell" else order.price + slip
                if self.fills.stop_gap_through:
                    # A bar opening beyond the stop fills at the (worse) open.
                    if order.side == "sell":
                        fill_price = min(candle.open, fill_price)
                    else:
                        fill_price = max(candle.open, fill_price)
                self._fill(order.side, order.size, fill_price, self.instrument.taker_fee_rate)
                self.exit_fills += 1
                order.status = "filled"

    def _fill_limits(self, candle: Candle) -> None:
        pen = self.fills.maker_penetration_ticks * self.instrument.tick_size
        for order in list(self.orders):
            if order.status != "open" or order.type != "post_only_limit":
                continue
            crossed = (order.side == "buy" and candle.low <= order.price - pen) or \
                      (order.side == "sell" and candle.high >= order.price + pen)
            if crossed:
                was_flat = abs(self.qty) < 1e-12
                self._fill(order.side, order.size, order.price, self.instrument.maker_fee_rate)
                if was_flat:
                    self.entry_fills += 1   # limit from flat = entry
                else:
                    self.exit_fills += 1    # limit against a position = take-profit
                order.status = "filled"

    def _check_liquidation(self, candle: Candle) -> None:
        """Maintenance-margin liquidation at the bar's adverse extreme (base mode)."""
        if abs(self.qty) < 1e-12:
            return
        worst = candle.low if self.qty > 0 else candle.high
        mark = worst if self.fills.intrabar_liquidation else candle.close
        equity = self.cash + self.qty * (mark - self.entry_price)
        maint = abs(self.qty) * mark * self.instrument.maintenance_margin_rate
        if equity >= maint:
            return
        slip = self.fills.taker_slippage_ticks * self.instrument.tick_size
        side = "sell" if self.qty > 0 else "buy"
        fill_price = worst - slip if self.qty > 0 else worst + slip
        self._fill(side, abs(self.qty), fill_price, self.instrument.taker_fee_rate)
        self.liquidations += 1
        self.exit_fills += 1
        for order in self.orders:
            if order.status == "open":
                order.status = "cancelled"

    def _fill(self, side: str, size: float, price: float, fee_rate: float) -> None:
        """Apply a fill to the futures-style account.

        ``cash`` is the *realized* balance: opening a leveraged position does not move
        cash (only fees), while closing realizes PnL into cash. Mark-to-market equity is
        ``cash + qty * (last_close - entry_price)`` (see ``get_equity``).
        """
        fee = size * price * fee_rate
        self.cash -= fee
        if side == "buy":
            if self.qty < 0:  # closing a short
                c = min(size, -self.qty)
                self.cash += c * (self.entry_price - price)
                self.qty += c
                size -= c
            if size > 0:  # opening/adding long (no cash move beyond fees)
                if self.qty == 0:
                    self.qty = size
                    self.entry_price = price
                else:
                    self.entry_price = (self.entry_price * self.qty + price * size) / (self.qty + size)
                    self.qty += size
        else:  # sell
            if self.qty > 0:  # closing a long
                c = min(size, self.qty)
                self.cash += c * (price - self.entry_price)
                self.qty -= c
                size -= c
            if size > 0:  # opening/adding short (no cash move beyond fees)
                if self.qty == 0:
                    self.qty = -size
                    self.entry_price = price
                else:
                    self.entry_price = (self.entry_price * (-self.qty) + price * size) / ((-self.qty) + size)
                    self.qty -= size

    # ---------------------------------------------------------------- orders
    async def place_post_only_limit(self, side: str, price: float, size: float) -> Optional[str]:
        if size <= 0:
            return None
        price = self._round_tick(price)
        # Post-only rejection: a buy at/above the market or a sell at/below the market
        # would take liquidity.
        if self.last_close is not None:
            if side == "buy" and price >= self.last_close:
                return None
            if side == "sell" and price <= self.last_close:
                return None
        if size * price < self.instrument.min_notional:
            return None
        order = Order(id=f"order-{next(self._ids)}", side=side, type="post_only_limit",
                      price=price, size=size)
        self.orders.append(order)
        return order.id

    async def place_stop_market(self, side: str, price: float, size: float) -> Optional[str]:
        if size <= 0:
            return None
        price = self._round_tick(price)
        order = Order(id=f"order-{next(self._ids)}", side=side, type="stop_market",
                      price=price, size=size)
        self.orders.append(order)
        return order.id

    async def cancel_all(self) -> None:
        for order in self.orders:
            if order.status == "open":
                order.status = "cancelled"
        self.orders = [o for o in self.orders if o.status != "open"]

    async def market_close_all(self) -> None:
        if self.qty != 0 and self.last_close is not None:
            side = "sell" if self.qty > 0 else "buy"
            self._fill(side, abs(self.qty), self.last_close, self.instrument.taker_fee_rate)
            self.exit_fills += 1

    # ---------------------------------------------------------------- account
    async def get_equity(self) -> float:
        unrealized = 0.0
        if self.qty != 0 and self.last_close is not None:
            unrealized = self.qty * (self.last_close - self.entry_price)
        return self.cash + unrealized

    async def get_positions(self) -> List[Position]:
        if abs(self.qty) < 1e-12:
            return []
        return [Position(qty=self.qty, entry_price=self.entry_price)]


class CoinbaseCDPAdapter(ExchangeAdapter):
    """Real Coinbase Advanced Trade adapter (OPTIONAL — requires cdp-sdk + network).

    The exact cdp-sdk method names and websocket message schemas are unverified; mark
    every integration point ``# VERIFY`` before live use. All methods raise
    ``RuntimeError`` with guidance until credentials/network are provided.
    """

    def __init__(self, instrument: InstrumentSpec, api_key_name: str,
                 api_key_private: str) -> None:
        self.instrument = instrument
        self.api_key_name = api_key_name
        self.api_key_private = api_key_private

    def _require(self) -> None:
        if not (self.api_key_name and self.api_key_private):
            raise RuntimeError("live mode requires CDP_API_KEY_NAME / CDP_API_KEY_PRIVATE_KEY env vars")
        try:
            import cdp  # noqa: F401  # VERIFY module/package name vs installed cdp-sdk
        except ImportError as exc:
            raise RuntimeError("cdp-sdk is required for live trading") from exc

    async def connect(self) -> None:
        self._require()  # VERIFY: cdp-sdk connection/bootstrap steps
        raise NotImplementedError("CoinbaseCDPAdapter.connect — VERIFY cdp-sdk bootstrap")

    async def stream_candles(self) -> AsyncIterator[Candle]:
        self._require()  # VERIFY: Coinbase websocket product/channel + message schema
        raise NotImplementedError("CoinbaseCDPAdapter.stream_candles — VERIFY websocket feed")
        yield  # pragma: no cover

    async def place_post_only_limit(self, side: str, price: float, size: float) -> Optional[str]:
        self._require()
        raise NotImplementedError("CoinbaseCDPAdapter.place_post_only_limit — VERIFY order API")

    async def place_stop_market(self, side: str, price: float, size: float) -> Optional[str]:
        self._require()
        raise NotImplementedError("CoinbaseCDPAdapter.place_stop_market — VERIFY order API")

    async def cancel_all(self) -> None:
        self._require()
        raise NotImplementedError("CoinbaseCDPAdapter.cancel_all — VERIFY cancel API")

    async def market_close_all(self) -> None:
        self._require()
        raise NotImplementedError("CoinbaseCDPAdapter.market_close_all — VERIFY close API")

    async def get_equity(self) -> float:
        self._require()
        raise NotImplementedError("CoinbaseCDPAdapter.get_equity — VERIFY account API")

    async def get_positions(self) -> List[Position]:
        self._require()
        raise NotImplementedError("CoinbaseCDPAdapter.get_positions — VERIFY positions API")
