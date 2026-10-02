"""core/backtester.py — GPU-vectorized population backtest with strict real-world emulation.

Runs a full population in one batched pass: arrays are shaped ``(P, N)`` where ``P`` is
the population size and ``N`` is the number of **1m fill bars**. The time axis is
stepped sequentially (a Python loop over bars — path-dependent position/margin state),
while the population axis is fully vectorized via ``xp``. Dual timeframe: **1h signals**
drive entry decisions; **1m bars** drive post-only fill detection, stop/TP triggering,
hourly funding, and the maintenance-margin liquidation check.

Causality (enforced structurally): the signal from the 1h candle closing at ``T`` is only
ever mapped to 1m bars with timestamp ``> T`` — i.e. signal index ``s`` acts on 1m hour
``s + 1`` (bars ``[(s+1)*bph, (s+2)*bph)``). No value from a future bar is ever read.

Fill model (see ``config.FillModelConfig``; selected via the ``fill_model`` argument):
  * Post-only maker limit fills only when the market price **crosses through** the limit
    by at least ``maker_penetration_ticks`` ticks (buy: ``low <= limit - n*tick``).
  * Post-only orders that would cross the spread at placement are **rejected**, and a
    resting order is cancelled when the latest signal is flat or its replacement rejects
    (``cancel_stale_orders``) — matching the live executor's quote policy.
  * Stop-loss is a **stop-market (taker)** order: triggers on touch, fills at the stop
    shifted adversely by ``stop_slippage_ticks`` (and at the open when a bar gaps
    through it, unless disabled).
  * Take-profit is a **maker limit**, filled on cross-through.
  * When a single 1m bar spans both TP and SL, the SL is assumed to trigger first
    (conservative worst case — ASSUMPTION).
  * ``same_bar_stop_check`` re-checks SL/TP on the bar that fills the entry (instead of
    ignoring the rest of that bar).
  * Liquidation uses the bar's **adverse extreme** (intrabar) with the worst-case fill
    price, not just the close.
  * Maker fee on maker fills, taker fee on stop/time/liquidation closes; hourly funding
    settled at each hour boundary on open positions.
  * An optional rolling-drawdown circuit breaker (``breaker_pct``) mirrors the live
    ``RiskConfig`` guard: once tripped a genome halts permanently (no new entries).

Single position per genome (no pyramiding) — ASSUMPTION.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from config import FillModelConfig, InstrumentSpec
from core.strategy import GENE_INDEX, SignalIntents
from core.xp import asnumpy, backend, xp

__all__ = [
    "Trade",
    "BacktestResult",
    "backtest",
    "assert_no_lookahead",
]

_FLOAT = xp.float64
_INT = xp.int64


def _round_tick(x: Any, tick: float) -> Any:
    """Round to the nearest tick (round-half-up) for fill comparisons."""
    return xp.floor(x / tick + 0.5) * tick


@dataclass
class Trade:
    """A single round-trip trade (entry + exit)."""

    genome_idx: int
    side: int  # +1 long, -1 short
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    qty: float  # unsigned (SOL)
    pnl: float  # gross PnL net of all fees (funding tracked separately)
    fees: float  # total fees paid (entry + exit)
    funding: float  # net funding paid (+) / received (-)
    exit_reason: str  # "tp" | "sl" | "time" | "liquidation"


@dataclass
class BacktestResult:
    """Per-genome backtest outputs (arrays shaped ``(P, ...)``)."""

    equity_curve: Any  # (P, M) mark-to-market equity per 1m bar
    equity_1h: Any  # (P, H) equity resampled to 1h
    trades: List[List[Trade]]  # per-genome trade logs
    num_trades: Any  # (P,)
    maker_fills: Any  # (P,)
    taker_exits: Any  # (P,)
    liquidations: Any  # (P,)
    breaker_trips: Any  # (P,)
    post_only_rejections: Any  # (P,)
    total_fees: Any  # (P,) cumulative USD fees paid (maker + taker)
    total_funding: Any  # (P,) cumulative USD funding paid (+) / received (-)
    final_equity: Any  # (P,)


class _Sim:
    """Mutable per-bar simulation state, vectorized over ``P``.

    ``cash`` is the realized USD balance; mark-to-market equity adds unrealized PnL.
    ``pos`` is the signed SOL quantity (long > 0, short < 0).
    """

    def __init__(self, p: int, m: int, initial_equity: float,
                 breaker_window_hours: int = 0) -> None:
        self.p = p
        self.m = m
        self.cash = xp.full(p, float(initial_equity), dtype=_FLOAT)
        self.pos = xp.zeros(p, dtype=_FLOAT)
        self.entry_px = xp.zeros(p, dtype=_FLOAT)
        self.sl_px = xp.zeros(p, dtype=_FLOAT)
        self.tp_px = xp.zeros(p, dtype=_FLOAT)
        self.entry_bar = xp.full(p, -1, dtype=_INT)
        self.entry_sig = xp.full(p, -1, dtype=_INT)
        self.entry_fee = xp.zeros(p, dtype=_FLOAT)
        self.pos_funding = xp.zeros(p, dtype=_FLOAT)
        self.halted = xp.zeros(p, dtype=xp.bool_)

        # Resting entry order (maker limit or taker breakout stop).
        self.rest_active = xp.zeros(p, dtype=xp.bool_)
        self.rest_is_breakout = xp.zeros(p, dtype=xp.bool_)
        self.rest_side = xp.zeros(p, dtype=_INT)
        self.rest_price = xp.zeros(p, dtype=_FLOAT)
        self.rest_qty = xp.zeros(p, dtype=_FLOAT)
        self.rest_tp = xp.zeros(p, dtype=_FLOAT)
        self.rest_sl = xp.zeros(p, dtype=_FLOAT)
        self.rest_sig = xp.full(p, -1, dtype=_INT)
        self.rest_atr = xp.zeros(p, dtype=_FLOAT)
        self.rest_trail = xp.zeros(p, dtype=_FLOAT)
        self.pos_atr = xp.zeros(p, dtype=_FLOAT)
        self.pos_trail = xp.zeros(p, dtype=_FLOAT)
        self.peak_px = xp.zeros(p, dtype=_FLOAT)
        self.pyramid_count = xp.zeros(p, dtype=_INT)

        self.n_trades = xp.zeros(p, dtype=_INT)
        self.maker_fills = xp.zeros(p, dtype=_INT)
        self.taker_exits = xp.zeros(p, dtype=_INT)
        self.liquidations = xp.zeros(p, dtype=_INT)
        self.breaker_trips = xp.zeros(p, dtype=_INT)
        self.post_only_rejections = xp.zeros(p, dtype=_INT)
        self.total_fees = xp.zeros(p, dtype=_FLOAT)
        self.total_funding = xp.zeros(p, dtype=_FLOAT)

        self.equity_curve = xp.empty((p, m), dtype=_FLOAT)
        self.trades: List[List[Trade]] = [[] for _ in range(p)]
        self.t = 0

        # Circuit-breaker rolling window (hours). Only allocated when enabled.
        self.breaker_hist: Any = None
        if breaker_window_hours > 0:
            self.breaker_hist = xp.full((p, breaker_window_hours), float(initial_equity),
                                        dtype=_FLOAT)

    def finalize_exit(self, mask: Any, exit_price: Any, reason: str,
                      fee_rate: float) -> None:
        """Close positions flagged by ``mask`` at ``exit_price`` (taker/maker per caller)."""
        m = mask
        pnl = self.pos * (exit_price - self.entry_px)
        fee = xp.abs(self.pos) * exit_price * fee_rate
        self.cash = self.cash + xp.where(m, pnl - fee, 0.0)
        self.total_fees = self.total_fees + xp.where(m, fee, 0.0)

        inc = m.astype(_INT)
        self.n_trades += inc
        if reason == "tp":
            self.maker_fills += inc
        else:
            self.taker_exits += inc
        if reason == "liquidation":
            self.liquidations += inc

        idxs = asnumpy(xp.nonzero(m)[0])
        if idxs.size:
            cash_np = asnumpy(self.cash)
            pos_np = asnumpy(self.pos)
            entry_px_np = asnumpy(self.entry_px)
            entry_bar_np = asnumpy(self.entry_bar)
            entry_fee_np = asnumpy(self.entry_fee)
            fund_np = asnumpy(self.pos_funding)
            pnl_np = asnumpy(pnl)
            fee_np = asnumpy(fee)
            exit_px_np = asnumpy(exit_price)
            for pidx in idxs:
                pidx = int(pidx)
                side = 1 if pos_np[pidx] > 0 else -1
                self.trades[pidx].append(Trade(
                    genome_idx=pidx,
                    side=side,
                    entry_bar=int(entry_bar_np[pidx]),
                    exit_bar=self.t,
                    entry_price=float(entry_px_np[pidx]),
                    exit_price=float(exit_px_np[pidx]),
                    qty=abs(float(pos_np[pidx])),
                    # Net of all fees (entry + exit); funding is tracked separately.
                    pnl=float(pnl_np[pidx]) - float(fee_np[pidx]) - float(entry_fee_np[pidx]),
                    fees=float(entry_fee_np[pidx]) + float(fee_np[pidx]),
                    funding=float(fund_np[pidx]),
                    exit_reason=reason,
                ))

        self.pos = xp.where(m, 0.0, self.pos)
        self.entry_px = xp.where(m, 0.0, self.entry_px)
        self.sl_px = xp.where(m, 0.0, self.sl_px)
        self.tp_px = xp.where(m, 0.0, self.tp_px)
        self.pos_atr = xp.where(m, 0.0, self.pos_atr)
        self.pos_trail = xp.where(m, 0.0, self.pos_trail)
        self.peak_px = xp.where(m, 0.0, self.peak_px)
        self.entry_bar = xp.where(m, -1, self.entry_bar)
        self.entry_sig = xp.where(m, -1, self.entry_sig)
        self.entry_fee = xp.where(m, 0.0, self.entry_fee)
        self.pos_funding = xp.where(m, 0.0, self.pos_funding)
        self.pyramid_count = xp.where(m, 0, self.pyramid_count)

    def open_position(self, mask: Any, side: Any, fill_price: Any, fee_rate: float, is_taker: bool = False) -> None:
        """Open positions for resting-entry fills at ``fill_price``."""
        m = mask
        qty = self.rest_qty
        entry_fee = qty * fill_price * fee_rate
        self.cash = self.cash - xp.where(m, entry_fee, 0.0)
        self.total_fees = self.total_fees + xp.where(m, entry_fee, 0.0)
        self.pos = xp.where(m, qty * side, self.pos)
        self.entry_px = xp.where(m, fill_price, self.entry_px)
        self.sl_px = xp.where(m, self.rest_sl, self.sl_px)
        self.tp_px = xp.where(m, self.rest_tp, self.tp_px)
        self.pos_atr = xp.where(m, self.rest_atr, self.pos_atr)
        self.pos_trail = xp.where(m, self.rest_trail, self.pos_trail)
        self.peak_px = xp.where(m, fill_price, self.peak_px)
        self.entry_bar = xp.where(m, self.t, self.entry_bar)
        self.entry_sig = xp.where(m, self.rest_sig, self.entry_sig)
        self.entry_fee = xp.where(m, entry_fee, self.entry_fee)
        self.pos_funding = xp.where(m, 0.0, self.pos_funding)
        self.pyramid_count = xp.where(m, 0, self.pyramid_count)
        if is_taker:
            self.taker_exits += m.astype(_INT)
        else:
            self.maker_fills += m.astype(_INT)
        self.rest_active = xp.where(m, False, self.rest_active)
        self.rest_is_breakout = xp.where(m, False, self.rest_is_breakout)


def _check_exits(sim: _Sim, o: Any, hi: Any, lo: Any, tick: float,
                 taker_rate: float, maker_rate: float,
                 fill_model: FillModelConfig, mask: Any = None) -> None:
    """Trigger and fill SL (taker, slippage-aware) then TP (maker) for open positions.

    Applies profit-lock breakeven jump (at +1.0 ATR favorable excursion) and continuous
    trailing stops (peak - trailing_atr_mult * ATR) before evaluating fills.
    """
    openm = sim.pos != 0
    if mask is not None:
        openm = openm & mask
    if not bool(asnumpy(xp.any(openm))):
        return
    long = sim.pos > 0
    short = sim.pos < 0

    # 1. Check stop and take-profit hits against active stop/TP levels coming into the bar
    sl_r = _round_tick(sim.sl_px, tick)
    tp_r = _round_tick(sim.tp_px, tick)
    sl_hit = openm & ((long & (lo <= sl_r)) | (short & (hi >= sl_r)))
    tp_hit = openm & ~sl_hit & ((long & (hi >= tp_r + tick)) |
                                (short & (lo <= tp_r - tick)))

    if bool(asnumpy(xp.any(sl_hit))):
        slip = fill_model.stop_slippage_ticks * tick
        # Stop-market with adverse slippage; on a gap the open is used when it is worse.
        long_fill = sl_r - slip
        short_fill = sl_r + slip
        if fill_model.stop_gap_through:
            long_fill = xp.minimum(o, long_fill)
            short_fill = xp.maximum(o, short_fill)
        sl_fill = xp.where(long, long_fill, short_fill)
        sim.finalize_exit(sl_hit, sl_fill, "sl", taker_rate)

    if bool(asnumpy(xp.any(tp_hit))):
        sim.finalize_exit(tp_hit, tp_r, "tp", maker_rate)

    # 2. For positions that survived the bar: dynamic trailing / breakeven stops for next bar
    has_trail = (sim.pos != 0) & (sim.pos_trail > 0.0) & (sim.pos_atr > 0.0)
    if mask is not None:
        has_trail = has_trail & mask
    if not bool(asnumpy(xp.any(has_trail))):
        return

    sim.peak_px = xp.where(has_trail & long, xp.maximum(sim.peak_px, hi), sim.peak_px)
    sim.peak_px = xp.where(has_trail & short, xp.minimum(sim.peak_px, lo), sim.peak_px)

    # Breakeven lock: if excursion >= +1.0 ATR in profit
    long_be_trig = has_trail & long & (sim.peak_px >= sim.entry_px + sim.pos_atr)
    short_be_trig = has_trail & short & (sim.peak_px <= sim.entry_px - sim.pos_atr)
    long_be_px = sim.entry_px + (maker_rate + taker_rate) * sim.entry_px + tick
    short_be_px = sim.entry_px - (maker_rate + taker_rate) * sim.entry_px - tick

    sim.sl_px = xp.where(long_be_trig, xp.maximum(sim.sl_px, long_be_px), sim.sl_px)
    sim.sl_px = xp.where(short_be_trig, xp.minimum(sim.sl_px, short_be_px), sim.sl_px)

    # Continuous trailing stop
    long_trail_px = sim.peak_px - sim.pos_trail * sim.pos_atr
    short_trail_px = sim.peak_px + sim.pos_trail * sim.pos_atr

    sim.sl_px = xp.where(has_trail & long, xp.maximum(sim.sl_px, long_trail_px), sim.sl_px)
    sim.sl_px = xp.where(has_trail & short, xp.minimum(sim.sl_px, short_trail_px), sim.sl_px)


def backtest(
    ohlcv_1m: Any,
    ohlcv_1h: Any,
    funding: Any,
    signals: SignalIntents,
    genome_batch: Any,
    instrument: InstrumentSpec,
    initial_equity: float = 10_000.0,
    bars_per_hour: int = 60,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    progress_every: int = 0,
    use_numba: bool = False,
    fill_model: Optional[FillModelConfig] = None,
    breaker_pct: float = 0.0,
    breaker_window_hours: int = 24,
) -> BacktestResult:
    """Run a population backtest over the full 1m series.

    Args:
        ohlcv_1m: DataFrame with a DatetimeIndex and columns ``[open, high, low, close,
            volume]`` (``M`` bars).
        ohlcv_1h: DataFrame with the same columns, resampled so each 1h candle covers
            exactly ``bars_per_hour`` preceding 1m candles (``H = M // bars_per_hour``).
        funding: DataFrame with a ``funding_rate`` column, reindexed onto the 1h index.
        signals: ``SignalIntents`` from ``strategy.compute_signals`` (``(P, H)`` arrays).
        genome_batch: ``(P, NUM_GENES)`` array (numpy or xp).
        instrument: ``InstrumentSpec`` (fees/ticks/lots/margin).
        initial_equity: starting USD per genome.
        bars_per_hour: 1m bars per 1h signal candle (default 60; 1m fill + 1h signal).
        progress_cb: optional ``cb(done_bars, total_bars)`` called every
            ``progress_every`` bars (used by the UI for live progress).
        progress_every: bar interval for ``progress_cb`` (0 disables progress).
        fill_model: exchange-emulation stress configuration (defaults to ``base``).
        breaker_pct: rolling-drawdown breaker threshold in percent (0 disables it).
        breaker_window_hours: rolling peak window for the breaker.

    Returns:
        ``BacktestResult`` with ``(P, ...)`` arrays.
    """
    fm = fill_model if fill_model is not None else FillModelConfig()

    # Optional Numba-JIT, multithreaded CPU path (parallel over the population axis).
    # Falls back to the xp loop when numba is absent or a GPU backend is active.
    if use_numba:
        from core.numba_backtester import NUMBA_AVAILABLE, backtest_numba

        if NUMBA_AVAILABLE and backend == "numpy":
            return backtest_numba(ohlcv_1m, ohlcv_1h, funding, signals, genome_batch,
                                  instrument, initial_equity=initial_equity,
                                  bars_per_hour=bars_per_hour, fill_model=fm,
                                  breaker_pct=breaker_pct,
                                  breaker_window_hours=breaker_window_hours)

    tick = instrument.tick_size
    lot = instrument.lot_size
    min_notional = instrument.min_notional
    maker_rate = instrument.maker_fee_rate
    taker_rate = instrument.taker_fee_rate
    maint_rate = instrument.maintenance_margin_rate
    max_lev = instrument.max_leverage
    slip_taker = fm.taker_slippage_ticks * tick
    penetration = fm.maker_penetration_ticks * tick

    open_1m = xp.asarray(ohlcv_1m["open"].to_numpy(), dtype=_FLOAT)
    high_1m = xp.asarray(ohlcv_1m["high"].to_numpy(), dtype=_FLOAT)
    low_1m = xp.asarray(ohlcv_1m["low"].to_numpy(), dtype=_FLOAT)
    close_1m = xp.asarray(ohlcv_1m["close"].to_numpy(), dtype=_FLOAT)
    m = close_1m.shape[0]
    h = m // bars_per_hour
    if m % bars_per_hour != 0:
        raise ValueError("ohlcv_1m length must be a multiple of bars_per_hour")

    # Funding rates aligned to the 1h index (forward-fill any mismatch).
    fund_series = funding["funding_rate"].reindex(ohlcv_1h.index).ffill().fillna(0.0).to_numpy()
    fund_rate = xp.asarray(fund_series[:h], dtype=_FLOAT)

    g = xp.asarray(genome_batch, dtype=_FLOAT)
    p = g.shape[0]

    sig_regime = xp.asarray(signals.regime, dtype=_FLOAT)
    sig_atr = xp.asarray(signals.atr, dtype=_FLOAT)
    sig_long = xp.asarray(signals.long_entry, dtype=xp.bool_)
    sig_short = xp.asarray(signals.short_entry, dtype=xp.bool_)
    sig_entry = xp.asarray(signals.entry_price, dtype=_FLOAT)

    tp_mult = g[:, GENE_INDEX["tp_atr_mult"]]
    sl_mult = g[:, GENE_INDEX["sl_atr_mult"]]
    leverage = xp.minimum(g[:, GENE_INDEX["target_leverage"]], max_lev)
    max_hold = xp.rint(g[:, GENE_INDEX["max_hold_bars"]]).astype(_INT)
    risk_pct = g[:, GENE_INDEX["risk_per_trade_pct"]]
    trail_mult = g[:, GENE_INDEX["trailing_atr_mult"]]

    use_breaker = breaker_pct > 0.0
    sim = _Sim(p, m, initial_equity,
               breaker_window_hours=breaker_window_hours if use_breaker else 0)
    breaker_thr = breaker_pct / 100.0

    for t in range(m):
        sim.t = t
        o = open_1m[t]
        hi = high_1m[t]
        lo = low_1m[t]
        c = close_1m[t]
        is_boundary = (t % bars_per_hour) == 0
        h_idx = t // bars_per_hour

        if is_boundary:
            # E. hourly funding — settle the just-completed hour (h_idx-1) on positions
            # still open entering this boundary (they held through that hour). Applied
            # BEFORE time-stop/entry so positions held over the hour are charged.
            if h_idx >= 1:
                open_pos = sim.pos != 0
                if bool(asnumpy(xp.any(open_pos))):
                    rate = fund_rate[h_idx - 1]
                    # funding_paid = qty * mark * rate (positive means the trader PAYS).
                    # A long pays when funding_rate > 0; a short receives.
                    funding_paid = sim.pos * c * rate
                    sim.cash = sim.cash - xp.where(open_pos, funding_paid, 0.0)
                    sim.pos_funding = sim.pos_funding + xp.where(open_pos, funding_paid, 0.0)
                    sim.total_funding = sim.total_funding + xp.where(open_pos, funding_paid, 0.0)

            # A. time stop (close at market close, taker). Measured from the fill bar
            # (base/pessimistic) or the signal bar (optimistic legacy).
            # Profit-protected runners: trades that reached +1.0 ATR favorable excursion and have
            # active trailing stops are allowed to run with the trend, rather than being cut short.
            if fm.max_hold_from_fill:
                ref_hour = xp.floor_divide(sim.entry_bar, bars_per_hour)
            else:
                ref_hour = sim.entry_sig
            in_profit = (sim.pos_trail > 0.0) & (
                ((sim.pos > 0) & (sim.peak_px >= sim.entry_px + sim.pos_atr)) |
                ((sim.pos < 0) & (sim.peak_px <= sim.entry_px - sim.pos_atr))
            )
            time_exit = (sim.pos != 0) & (sim.entry_sig >= 0) & ~in_profit & \
                ((xp.full(p, h_idx, dtype=_INT) - ref_hour) >= max_hold)
            if bool(asnumpy(xp.any(time_exit))):
                time_fill = xp.where(sim.pos > 0, c - slip_taker, c + slip_taker)
                sim.finalize_exit(time_exit, time_fill, "time", taker_rate)

            # B. place new entries from signal s = h_idx - 1
            s = h_idx - 1
            if s >= 0:
                side = xp.where(sig_long[:, s], 1, xp.where(sig_short[:, s], -1, 0))
                flat = sim.pos == 0
                want = flat & (side != 0) & ~sim.halted
                raw_l = sig_entry[:, s]
                raw_bk = getattr(signals, "breakout_price", None)
                is_bk_sig = getattr(signals, "is_breakout", None)

                # Determine if this order is a breakout stop or a limit pullback
                is_bk = xp.zeros(p, dtype=xp.bool_)
                if is_bk_sig is not None and raw_bk is not None:
                    bk_s = xp.asarray(raw_bk[:, s], dtype=_FLOAT)
                    is_bk = xp.asarray(is_bk_sig[:, s], dtype=xp.bool_) & ~xp.isnan(bk_s)
                    target_px = xp.where(is_bk, bk_s, xp.where(xp.isnan(raw_l), c, raw_l))
                else:
                    target_px = xp.where(xp.isnan(raw_l), c, raw_l)
                L = _round_tick(target_px, tick)

                # Post-only rejection applies only to resting limit orders (not breakout stop triggers)
                reject = ~is_bk & (((side == 1) & (L >= o)) | ((side == -1) & (L <= o)))
                place = want & ~reject

                # Volatility-targeted risk sizing:
                # Cash risked = cash * risk_pct; stop distance = max(sl_mult * atr_s, tick)
                atr_s = sig_atr[:, s]
                stop_dist = xp.maximum(sl_mult * atr_s, tick)
                max_lev_qty = (sim.cash * leverage) / L
                desired_qty = xp.where(risk_pct > 0.0, (sim.cash * risk_pct) / stop_dist, max_lev_qty)
                qty = xp.floor(xp.minimum(desired_qty, max_lev_qty) / lot + 1e-9) * lot
                valid = (qty > 0) & (qty * L >= min_notional)
                place = place & valid
                sim.post_only_rejections += (want & reject).astype(_INT)

                if fm.cancel_stale_orders:
                    # Flat genomes re-quote to this hour's intent or cancel stale orders.
                    sim.rest_active = xp.where(flat, place, sim.rest_active)
                sim.rest_active = xp.where(place, True, sim.rest_active)
                sim.rest_is_breakout = xp.where(place, is_bk, sim.rest_is_breakout)
                sim.rest_side = xp.where(place, side, sim.rest_side)
                sim.rest_price = xp.where(place, L, sim.rest_price)
                sim.rest_qty = xp.where(place, qty, sim.rest_qty)
                sim.rest_tp = xp.where(place, _round_tick(L + side * tp_mult * atr_s, tick),
                                       sim.rest_tp)
                sim.rest_sl = xp.where(place, _round_tick(L - side * sl_mult * atr_s, tick),
                                       sim.rest_sl)
                sim.rest_sig = xp.where(place, s, sim.rest_sig)
                sim.rest_atr = xp.where(place, atr_s, sim.rest_atr)
                sim.rest_trail = xp.where(place, trail_mult, sim.rest_trail)

            # G. rolling-drawdown circuit breaker (mirrors the live RiskConfig guard).
            # Evaluated at the hour boundary on mark-to-market equity; a trip closes the
            # position and permanently halts new entries for that genome.
            if use_breaker:
                eq_now = sim.cash + xp.where(sim.pos != 0, sim.pos * (c - sim.entry_px), 0.0)
                sim.breaker_hist = xp.roll(sim.breaker_hist, -1, axis=1)
                sim.breaker_hist[:, -1] = eq_now
                peak = sim.breaker_hist.max(axis=1)
                dd = 1.0 - eq_now / xp.maximum(peak, 1e-12)
                trip = (~sim.halted) & (sim.pos != 0) & (dd >= breaker_thr)
                if bool(asnumpy(xp.any(trip))):
                    trip_fill = xp.where(sim.pos > 0, c - slip_taker, c + slip_taker)
                    sim.finalize_exit(trip, trip_fill, "breaker", taker_rate)
                    sim.breaker_trips += trip.astype(_INT)
                    sim.halted = xp.where(trip, True, sim.halted)
                    sim.rest_active = xp.where(trip, False, sim.rest_active)

        # C. exit fills for open positions (SL first, then TP — conservative)
        _check_exits(sim, o, hi, lo, tick, taker_rate, maker_rate, fm)

        # D. entry fills for resting orders (flat genomes)
        rest = sim.rest_active & ~sim.halted
        if bool(asnumpy(xp.any(rest))):
            # Limit fills (maker):
            rest_lim = rest & ~sim.rest_is_breakout
            buy_lim_fill = rest_lim & (sim.rest_side == 1) & (lo <= sim.rest_price - penetration)
            sell_lim_fill = rest_lim & (sim.rest_side == -1) & (hi >= sim.rest_price + penetration)
            lim_fill = buy_lim_fill | sell_lim_fill

            # Breakout stop-market fills (taker):
            rest_bk = rest & sim.rest_is_breakout
            buy_bk_fill = rest_bk & (sim.rest_side == 1) & (hi >= sim.rest_price)
            sell_bk_fill = rest_bk & (sim.rest_side == -1) & (lo <= sim.rest_price)
            bk_fill = buy_bk_fill | sell_bk_fill

            if bool(asnumpy(xp.any(lim_fill))):
                sim.open_position(lim_fill, sim.rest_side, sim.rest_price, maker_rate, is_taker=False)
                if fm.same_bar_stop_check:
                    _check_exits(sim, o, hi, lo, tick, taker_rate, maker_rate, fm, mask=lim_fill)

            if bool(asnumpy(xp.any(bk_fill))):
                bk_fill_px = xp.where(sim.rest_side == 1, xp.maximum(o, sim.rest_price) + slip_taker,
                                                          xp.minimum(o, sim.rest_price) - slip_taker)
                sim.open_position(bk_fill, sim.rest_side, bk_fill_px, taker_rate, is_taker=True)
                if fm.same_bar_stop_check:
                    _check_exits(sim, o, hi, lo, tick, taker_rate, maker_rate, fm, mask=bk_fill)

        # D3. Pyramiding on profit-locked runners:
        # If trade reached +2.0 ATR excursion AND trailing stop has locked in >= +0.5 ATR profit:
        # Reinvest floating profit by adding +35% size at market while stop trails.
        can_pyr = (sim.pos != 0) & (sim.pyramid_count == 0) & (
            ((sim.pos > 0) & (sim.peak_px >= sim.entry_px + 2.0 * sim.pos_atr) & (sim.sl_px >= sim.entry_px + 0.5 * sim.pos_atr)) |
            ((sim.pos < 0) & (sim.peak_px <= sim.entry_px - 2.0 * sim.pos_atr) & (sim.sl_px <= sim.entry_px - 0.5 * sim.pos_atr))
        )
        if bool(asnumpy(xp.any(can_pyr))):
            pyr_qty = xp.floor(xp.abs(sim.pos) * 0.35 / lot + 1e-9) * lot
            valid_pyr = can_pyr & (pyr_qty > 0)
            if bool(asnumpy(xp.any(valid_pyr))):
                pyr_side = xp.sign(sim.pos)
                pyr_px = xp.where(pyr_side > 0, c + slip_taker, c - slip_taker)
                pyr_fee = pyr_qty * pyr_px * taker_rate
                sim.cash -= xp.where(valid_pyr, pyr_fee, 0.0)
                sim.total_fees += xp.where(valid_pyr, pyr_fee, 0.0)
                old_notional = xp.abs(sim.pos) * sim.entry_px
                add_notional = pyr_qty * pyr_px
                new_pos = sim.pos + xp.where(valid_pyr, pyr_side * pyr_qty, 0.0)
                sim.entry_px = xp.where(valid_pyr, (old_notional + add_notional) / xp.maximum(xp.abs(new_pos), 1e-12), sim.entry_px)
                sim.pos = new_pos
                sim.pyramid_count = xp.where(valid_pyr, 1, sim.pyramid_count)

        # E. hourly funding handled at the hour boundary (see above).

        # F. mark-to-market + maintenance-margin liquidation (intrabar extreme).
        if fm.intrabar_liquidation:
            worst = xp.where(sim.pos > 0, lo, hi)
            eq_worst = sim.cash + xp.where(sim.pos != 0, sim.pos * (worst - sim.entry_px), 0.0)
            maint = xp.abs(sim.pos) * worst * maint_rate
            liq = (sim.pos != 0) & (eq_worst < maint)
        else:
            equity_close = sim.cash + xp.where(sim.pos != 0, sim.pos * (c - sim.entry_px), 0.0)
            maint = xp.abs(sim.pos) * c * maint_rate
            liq = (sim.pos != 0) & (equity_close < maint)
            worst = xp.full(p, c, dtype=_FLOAT)
        if bool(asnumpy(xp.any(liq))):
            liq_fill = xp.where(sim.pos > 0, worst - slip_taker, worst + slip_taker)
            sim.finalize_exit(liq, liq_fill, "liquidation", taker_rate)

        unrealized = sim.pos * (c - sim.entry_px)
        sim.equity_curve[:, t] = sim.cash + unrealized

        if progress_cb is not None and progress_every > 0 and (t % progress_every == 0):
            progress_cb(t, m)

    if progress_cb is not None and progress_every > 0:
        progress_cb(m, m)

    equity_1h = sim.equity_curve[:, bars_per_hour - 1::bars_per_hour]

    return BacktestResult(
        equity_curve=sim.equity_curve,
        equity_1h=equity_1h,
        trades=sim.trades,
        num_trades=sim.n_trades,
        maker_fills=sim.maker_fills,
        taker_exits=sim.taker_exits,
        liquidations=sim.liquidations,
        breaker_trips=sim.breaker_trips,
        post_only_rejections=sim.post_only_rejections,
        total_fees=sim.total_fees,
        total_funding=sim.total_funding,
        final_equity=sim.equity_curve[:, -1],
    )


def assert_no_lookahead(original: Any, perturbed: Any, cutoff: int,
                        atol: float = 1e-9) -> None:
    """Raise ``AssertionError`` if ``original`` and ``perturbed`` differ before
    ``cutoff``.

    Used to verify that perturbing data strictly after ``cutoff`` bars leaves the
    pre-``cutoff`` output identical — i.e. no future information leaked backward.
    """
    a = xp.asarray(original)[:, :cutoff]
    b = xp.asarray(perturbed)[:, :cutoff]
    ok = bool(asnumpy(xp.allclose(a, b, atol=atol, equal_nan=True)))
    if not ok:
        raise AssertionError("lookahead detected: output changed before the cutoff bar")
