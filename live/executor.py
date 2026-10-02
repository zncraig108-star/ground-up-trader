"""live/executor.py — async live/paper execution engine with ensemble netting & trailing stops.

Consumes either a single deployed ``Genome`` or a multi-strategy ``Ensemble``,
runs the strategy live against an ``ExchangeAdapter`` (paper by default),
routes post-only entries + maker TP + stop-market SL, and enforces the risk guardrails:

* **Volatility-Targeted Sizing**:
    ``contracts = min((equity * risk_pct) / stop_dist, (equity * leverage) / price)``
    floored to ``lot_size`` and capped at ``max_leverage``.
* **Breakeven & Trailing Stops**:
    Snaps stop to breakeven when excursion >= +1.0 ATR in profit, and trails
    favorable peak price at ``peak - trailing_atr_mult * ATR``.
* **Ensemble Order Netting**:
    Aggregates desired positions across all active ensemble members:
    ``net_qty = sum(w_i * member_target_qty_i)``.
* **Circuit breaker**:
    Halts if rolling 24h drawdown hits ``drawdown_breaker_pct``.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from config import Config, InstrumentSpec
from core.strategy import Genome, SignalFeatures, compute_signals
from core.xp import asnumpy
from live.exchange import Candle, ExchangeAdapter, Position

__all__ = ["Executor"]


class Executor:
    """Live/paper strategy runner supporting single genomes and multi-strategy ensembles."""

    def __init__(self, config: Config, instrument: InstrumentSpec,
                 adapter: ExchangeAdapter) -> None:
        self.config = config
        self.instrument = instrument
        self.adapter = adapter
        self.mode: str = "single"  # "single" | "ensemble"
        self.genome: Optional[Genome] = None
        self.ensemble_members: List[Dict[str, Any]] = []
        self.breaker_tripped = False
        self._reset_state()

    def _reset_state(self) -> None:
        self.equity_samples: List[tuple] = []
        self.h1_open: List[float] = []
        self.h1_high: List[float] = []
        self.h1_low: List[float] = []
        self.h1_close: List[float] = []
        self.h1_volume: List[float] = []
        self.cur_hour: Optional[int] = None
        self.cur_o = self.cur_h = self.cur_l = self.cur_c = 0.0
        self.cur_vol = 0.0
        self.regime_dir = 0
        self.entry_price: Optional[float] = None
        self.atr: Optional[float] = None
        self.max_hold = 0
        self.entry_placed = False
        self.tp_sl_placed = False
        self.stop_placed = False
        self.fresh_signal = False
        self.entry_hour: Optional[int] = None
        self._stop = False

        # Trailing stop state
        self.active_sl_price: Optional[float] = None
        self.active_tp_price: Optional[float] = None
        self.peak_price: Optional[float] = None
        self.pos_atr: Optional[float] = None
        self.pos_trail_mult: float = 2.0

        # Market telemetry
        self.cur_vol_z: float = 0.0
        self.cur_fund_z: float = 0.0
        self.cur_fund_rate: float = 0.0

        # Ensemble state
        self.member_signals: Dict[str, Dict[str, Any]] = {}
        self.net_target_qty: float = 0.0

    # ------------------------------------------------------------------ control
    async def deploy(self, genome: Genome) -> None:
        """Deploy a single champion genome, resetting account/risk state."""
        self.mode = "single"
        self.genome = genome
        self.ensemble_members = []
        self.breaker_tripped = False
        self._reset_state()
        self.max_hold = int(round(genome.max_hold_bars))
        self.pos_trail_mult = float(getattr(genome, "trailing_atr_mult", 2.0))
        await self.adapter.cancel_all()

    async def deploy_ensemble(self, members: List[Dict[str, Any]]) -> None:
        """Deploy a multi-strategy ensemble, resetting account/risk state."""
        self.mode = "ensemble"
        self.genome = None
        self.ensemble_members = members
        self.breaker_tripped = False
        self._reset_state()
        if members:
            # Anchor max_hold to the weighted average max_hold
            holds = [float(m.get("params", {}).get("max_hold_bars", 72.0)) * float(m.get("weight", 0.0))
                     for m in members]
            self.max_hold = int(round(sum(holds))) if holds else 72
        await self.adapter.cancel_all()

    def stop(self) -> None:
        self._stop = True

    async def run(self) -> None:
        """Consume the candle stream until ``stop()`` is called or the feed ends."""
        await self.adapter.connect()
        async for candle in self.adapter.stream_candles():
            if self._stop:
                break
            await self._on_candle(candle)

    # --------------------------------------------------------------- per-candle
    async def _on_candle(self, candle: Candle) -> None:
        hour = (candle.timestamp - 1) // 3_600_000 + 1
        new_hour = self.cur_hour is not None and hour != self.cur_hour

        if new_hour:
            self._finalize_hour()
            self.entry_placed = False
            self._compute_signal()
            self.fresh_signal = True

        self._aggregate(candle)

        # Risk: circuit breaker on every candle.
        equity = await self.adapter.get_equity()
        if await self._check_breaker(candle.timestamp, equity):
            return

        positions = await self.adapter.get_positions()
        flat = not positions or abs(positions[0].qty) < 1e-9

        if flat:
            self.tp_sl_placed = False
            self.stop_placed = False
            self.active_sl_price = None
            self.peak_price = None

            if self.mode == "single":
                if self.fresh_signal:
                    if self.genome is not None and self.regime_dir != 0 and self.entry_price is not None:
                        side = "buy" if self.regime_dir > 0 else "sell"
                        size = self._dynamic_size(equity, self.entry_price, genome=self.genome, atr=self.atr)
                        await self.adapter.cancel_all()
                        self.entry_placed = True
                        if size > 0:
                            await self.adapter.place_post_only_limit(side, self.entry_price, size)
                    elif self.config.fills.cancel_stale_orders and self.entry_placed:
                        await self.adapter.cancel_all()
                        self.entry_placed = False
                    self.fresh_signal = False
            elif self.mode == "ensemble":
                # Ensemble execution: rebalance toward net target quantity
                if self.fresh_signal:
                    net_qty = self.net_target_qty
                    if abs(net_qty) > 0:
                        side = "buy" if net_qty > 0 else "sell"
                        px = candle.close
                        await self.adapter.cancel_all()
                        self.entry_placed = True
                        await self.adapter.place_post_only_limit(side, px, abs(net_qty))
                    elif self.config.fills.cancel_stale_orders and self.entry_placed:
                        await self.adapter.cancel_all()
                        self.entry_placed = False
                    self.fresh_signal = False
        else:
            pos = positions[0]
            if not self.tp_sl_placed:
                await self._place_tp_sl(pos)
                self.entry_hour = self.entry_hour or hour
                self.peak_price = pos.entry_price
                self.pos_atr = self.atr or (pos.entry_price * 0.01)

            # Intra-bar Trailing Stop & Breakeven Check
            if self.active_sl_price is not None and self.peak_price is not None:
                atr_val = self.pos_atr or 1.0
                tick = self.instrument.tick_size
                fees = (self.instrument.maker_fee_rate + self.instrument.taker_fee_rate) * pos.entry_price

                if pos.qty > 0:  # Long
                    # Trigger stop if low touches active SL
                    if candle.low <= self.active_sl_price:
                        await self.adapter.cancel_all()
                        await self.adapter.market_close_all()
                        self._reset_position_state()
                        return
                    if self.pos_trail_mult > 0.0:
                        self.peak_price = max(self.peak_price, candle.high)
                        # Breakeven lock: +1.0 ATR excursion
                        if self.peak_price >= pos.entry_price + atr_val:
                            be_level = pos.entry_price + fees + tick
                            self.active_sl_price = max(self.active_sl_price, be_level)
                        # Continuous trailing stop
                        trail_level = self.peak_price - self.pos_trail_mult * atr_val
                        self.active_sl_price = max(self.active_sl_price, trail_level)
                else:  # Short
                    # Trigger stop if high touches active SL
                    if candle.high >= self.active_sl_price:
                        await self.adapter.cancel_all()
                        await self.adapter.market_close_all()
                        self._reset_position_state()
                        return
                    if self.pos_trail_mult > 0.0:
                        self.peak_price = min(self.peak_price, candle.low)
                        # Breakeven lock: -1.0 ATR excursion
                        if self.peak_price <= pos.entry_price - atr_val:
                            be_level = pos.entry_price - fees - tick
                            self.active_sl_price = min(self.active_sl_price, be_level)
                        # Continuous trailing stop
                        trail_level = self.peak_price + self.pos_trail_mult * atr_val
                        self.active_sl_price = min(self.active_sl_price, trail_level)

            # Time stop check
            if self.entry_hour is not None and (hour - self.entry_hour) >= self.max_hold:
                await self.adapter.cancel_all()
                await self.adapter.market_close_all()
                self._reset_position_state()

    def _reset_position_state(self) -> None:
        self.tp_sl_placed = False
        self.stop_placed = False
        self.entry_hour = None
        self.active_sl_price = None
        self.active_tp_price = None
        self.peak_price = None

    # -------------------------------------------------------------- aggregation
    def _aggregate(self, candle: Candle) -> None:
        if self.cur_hour is None:
            self.cur_hour = (candle.timestamp - 1) // 3_600_000 + 1
            self.cur_o = candle.open
            self.cur_h = candle.high
            self.cur_l = candle.low
            self.cur_c = candle.close
            self.cur_vol = candle.volume
        else:
            self.cur_h = max(self.cur_h, candle.high)
            self.cur_l = min(self.cur_l, candle.low)
            self.cur_c = candle.close
            self.cur_vol += candle.volume

    def _finalize_hour(self) -> None:
        self.h1_open.append(self.cur_o)
        self.h1_high.append(self.cur_h)
        self.h1_low.append(self.cur_l)
        self.h1_close.append(self.cur_c)
        self.h1_volume.append(self.cur_vol)
        self.cur_hour += 1
        self.cur_o = self.cur_c
        self.cur_h = self.cur_c
        self.cur_l = self.cur_c
        self.cur_vol = 0.0

    def _compute_signal(self) -> None:
        n = len(self.h1_close)
        if n < 2:
            self.regime_dir = 0
            return

        # Calculate causal volume z-score
        vols = np.asarray(self.h1_volume, dtype=np.float64)
        if n >= 24:
            v_mean = float(np.mean(vols[-24:-1]))
            v_std = float(np.std(vols[-24:-1]))
            self.cur_vol_z = float((vols[-1] - v_mean) / (v_std + 1e-9))
        else:
            self.cur_vol_z = 0.0

        feat = SignalFeatures(
            open=np.asarray(self.h1_open, dtype=np.float64),
            high=np.asarray(self.h1_high, dtype=np.float64),
            low=np.asarray(self.h1_low, dtype=np.float64),
            close=np.asarray(self.h1_close, dtype=np.float64),
            volume_zscore=np.full(n, self.cur_vol_z, dtype=np.float64),
            funding_zscore=np.full(n, self.cur_fund_z, dtype=np.float64),
        )

        if self.mode == "single" and self.genome is not None:
            g = np.asarray([self.genome.encode()], dtype=np.float64)
            sig = compute_signals(feat, g)
            self.regime_dir = int(asnumpy(sig.regime)[0, -1])
            self.entry_price = float(asnumpy(sig.entry_price)[0, -1])
            self.atr = float(asnumpy(sig.atr)[0, -1])
            self.pos_trail_mult = float(getattr(self.genome, "trailing_atr_mult", 2.0))
        elif self.mode == "ensemble" and self.ensemble_members:
            # Batch signal computation across all ensemble members
            vecs = []
            weights = []
            for m in self.ensemble_members:
                g_arr = m.get("genome")
                if g_arr is not None:
                    vecs.append(g_arr)
                    weights.append(float(m.get("weight", 0.0)))
            if vecs:
                batch = np.asarray(vecs, dtype=np.float64)
                sigs = compute_signals(feat, batch)
                regimes = asnumpy(sigs.regime)[:, -1]
                atrs = asnumpy(sigs.atr)[:, -1]
                entries = asnumpy(sigs.entry_price)[:, -1]
                self.atr = float(np.mean(atrs))

                # Compute weighted target notional / signed position
                net_qty = 0.0
                for idx, m in enumerate(self.ensemble_members):
                    w = weights[idx]
                    r_dir = int(regimes[idx])
                    g_obj = Genome.decode(batch[idx])
                    # Sizing for 100% of nominal equity allocated by weight w
                    nominal_equity = self.config.risk.initial_equity * w
                    entry_px = float(entries[idx]) if not np.isnan(entries[idx]) else self.cur_c
                    if r_dir != 0:
                        size = self._dynamic_size(nominal_equity, entry_px, genome=g_obj, atr=float(atrs[idx]))
                        net_qty += float(r_dir) * size
                self.net_target_qty = self.instrument.round_lot(net_qty)
                self.regime_dir = 1 if self.net_target_qty > 0 else (-1 if self.net_target_qty < 0 else 0)

    # ------------------------------------------------------------------- orders
    async def _place_tp_sl(self, pos: Position) -> None:
        qty = abs(pos.qty)
        atr = self.atr if self.atr and self.atr > 0 else (pos.entry_price * 0.01)
        entry = pos.entry_price

        if self.genome is not None:
            tp_mult = float(self.genome.tp_atr_mult)
            sl_mult = float(self.genome.sl_atr_mult)
        else:
            tp_mult = 3.0
            sl_mult = 1.5

        if pos.qty > 0:  # long
            tp = self.instrument.round_tick(entry + tp_mult * atr)
            sl = self.instrument.round_tick(entry - sl_mult * atr)
            close_side = "sell"
        else:  # short
            tp = self.instrument.round_tick(entry - tp_mult * atr)
            sl = self.instrument.round_tick(entry + sl_mult * atr)
            close_side = "buy"

        self.active_sl_price = sl
        self.active_tp_price = tp

        if not self.stop_placed:
            await self.adapter.place_stop_market(close_side, sl, qty)
            self.stop_placed = True
        order_id = await self.adapter.place_post_only_limit(close_side, tp, qty)
        if order_id is None:
            await self.adapter.cancel_all()
            await self.adapter.market_close_all()
            self._reset_position_state()
        else:
            self.tp_sl_placed = True

    def _dynamic_size(self, equity: float, price: float,
                      genome: Optional[Genome] = None, atr: Optional[float] = None) -> float:
        g = genome or self.genome
        if g is None or price <= 0 or equity <= 0:
            return 0.0
        lev = min(float(g.target_leverage), float(self.instrument.max_leverage))
        a = atr or (self.atr if self.atr and self.atr > 0 else (price * 0.01))

        # Volatility-targeted risk sizing:
        # risk_dollars = equity * risk_per_trade_pct
        # stop_dist = max(sl_atr_mult * atr, tick_size)
        risk_pct = float(getattr(g, "risk_per_trade_pct", 0.015))
        stop_dist = max(float(g.sl_atr_mult) * a, self.instrument.tick_size)
        desired_qty = (equity * risk_pct) / stop_dist
        max_lev_qty = (equity * lev) / price

        qty = self.instrument.floor_to_lot(min(desired_qty, max_lev_qty))
        if qty * price < self.instrument.min_notional:
            return 0.0
        return qty

    # --------------------------------------------------------------------- risk
    async def _check_breaker(self, ts: int, equity: float) -> bool:
        if self.breaker_tripped:
            return True
        self.equity_samples.append((ts, equity))
        window_ms = self.config.risk.breaker_window_hours * 3_600_000
        cutoff = ts - window_ms
        self.equity_samples = [s for s in self.equity_samples if s[0] >= cutoff]
        peak = max(e for _, e in self.equity_samples) if self.equity_samples else equity
        dd = 1.0 - equity / peak if peak > 0 else 0.0
        if dd >= self.config.risk.drawdown_breaker_pct / 100.0:
            self.breaker_tripped = True
            await self.adapter.cancel_all()
            await self.adapter.market_close_all()
            return True
        return False

    # ---------------------------------------------------------------- telemetry
    def get_telemetry(self) -> Dict[str, Any]:
        """Real-time operational telemetry for UI dashboard & REST API."""
        return {
            "mode": self.mode,
            "genome": asdict(self.genome) if self.genome else None,
            "ensemble_members": self.ensemble_members,
            "breaker_tripped": self.breaker_tripped,
            "cur_hour": self.cur_hour,
            "regime_dir": self.regime_dir,
            "entry_price": self.entry_price,
            "atr": self.atr,
            "trailing_sl": self.active_sl_price,
            "tp_price": self.active_tp_price,
            "peak_price": self.peak_price,
            "volume_zscore": self.cur_vol_z,
            "funding_zscore": self.cur_fund_z,
            "funding_rate": self.cur_fund_rate,
            "net_target_qty": self.net_target_qty,
        }
