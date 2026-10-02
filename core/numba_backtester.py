"""core/numba_backtester.py — optional Numba-JIT, multithreaded CPU backtester.

A compiled drop-in for ``core.backtester.backtest`` that trades the ``xp`` array
abstraction for a Numba ``@njit(parallel=True)`` kernel. Parallelism is over the
**population axis** (``prange`` over genomes): each genome is an independent,
path-dependent simulation, while the 1m time axis stays strictly sequential inside it.
This removes the Python-interpreter overhead that dominates the NumPy path and spreads
the population across CPU threads.

Correctness is preserved: the kernel mirrors ``backtester.backtest`` step-for-step
(cross-through-by-1-tick post-only fills, post-only rejection, stale-order cancellation,
maker/taker fees, hourly funding sign, slippage-aware stop fills, same-bar stop checks,
intrabar liquidation, time stop measured from fill, optional rolling-drawdown breaker).
Detailed per-trade logs are omitted on this path (only counts + equity curves are
produced) — the ``xp`` path remains authoritative for trade-level inspection.

Selected automatically by ``core.backtester.backtest(..., use_numba=True)`` when numba is
installed and the NumPy backend is active; otherwise it falls back to the ``xp`` loop.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from config import FillModelConfig, InstrumentSpec
from core.strategy import GENE_INDEX, SignalIntents

try:  # pragma: no cover - depends on optional dependency
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore
        def _deco(func):
            return func
        return _deco

    prange = range  # type: ignore

__all__ = ["NUMBA_AVAILABLE", "backtest_numba"]


@njit(parallel=True, fastmath=False, cache=True)
def _kernel(o, h, l, c, fund_rate, sig_regime, sig_atr, sig_long, sig_short, sig_entry,
            tp_mult, sl_mult, lev, max_hold, risk_pct, trail_mult,
            tick, lot, min_notional, maker_rate,
            taker_rate, maint_rate, initial_equity, bph, pen_ticks, stop_slip_ticks,
            gap_through, cancel_stale, max_hold_from_fill, same_bar_check, intrabar_liq,
            taker_slip_ticks, breaker_pct, breaker_window):
    p, n_sig = sig_regime.shape
    m = c.shape[0]

    equity = np.empty((p, m), dtype=np.float64)
    n_trades = np.zeros(p, dtype=np.int64)
    maker_fills = np.zeros(p, dtype=np.int64)
    taker_exits = np.zeros(p, dtype=np.int64)
    liquidations = np.zeros(p, dtype=np.int64)
    rejections = np.zeros(p, dtype=np.int64)
    breakers = np.zeros(p, dtype=np.int64)
    total_fees = np.zeros(p, dtype=np.float64)
    total_funding = np.zeros(p, dtype=np.float64)

    stop_slip = stop_slip_ticks * tick
    penetration = pen_ticks * tick
    slip_taker = taker_slip_ticks * tick
    breaker_thr = breaker_pct / 100.0
    use_breaker = breaker_pct > 0.0

    for pi in prange(p):
        cash = initial_equity
        pos = 0.0
        entry_px = 0.0
        sl_px = 0.0
        tp_px = 0.0
        entry_sig = -1
        entry_fill_hour = -1
        rest_active = False
        rest_side = 0
        rest_price = 0.0
        rest_qty = 0.0
        rest_tp = 0.0
        rest_sl = 0.0
        rest_sig = -1
        rest_atr = 0.0
        pos_atr = 0.0
        peak_px = 0.0
        halted = False
        mh = int(round(max_hold[pi]))
        levp = lev[pi]
        tpp = tp_mult[pi]
        slp = sl_mult[pi]
        riskp = risk_pct[pi]
        trailp = trail_mult[pi]
        breaker = np.full(breaker_window, initial_equity, dtype=np.float64) \
            if use_breaker else np.empty(0, dtype=np.float64)
        b_idx = 0

        for t in range(m):
            oo = o[t]
            hh = h[t]
            ll = l[t]
            cc = c[t]
            is_b = (t % bph) == 0
            h_idx = t // bph

            if is_b:
                # hourly funding for positions held through the prior hour (h_idx-1)
                if h_idx >= 1 and pos != 0.0:
                    cash -= pos * cc * fund_rate[h_idx - 1]
                    total_funding[pi] += pos * cc * fund_rate[h_idx - 1]
                # time stop (taker close at the bar close)
                if pos != 0.0 and entry_sig >= 0:
                    ref = entry_fill_hour if max_hold_from_fill else entry_sig
                    if ref >= 0 and (h_idx - ref) >= mh:
                        if pos > 0.0:
                            fill = cc - slip_taker
                        else:
                            fill = cc + slip_taker
                        cash += pos * (fill - entry_px) - abs(pos) * fill * taker_rate
                        total_fees[pi] += abs(pos) * fill * taker_rate
                        n_trades[pi] += 1
                        taker_exits[pi] += 1
                        pos = 0.0
                        entry_sig = -1
                # place a new post-only entry from signal s = h_idx - 1
                s = h_idx - 1
                if s >= 0 and pos == 0.0 and not halted:
                    d = 0
                    if sig_long[pi, s]:
                        d = 1
                    elif sig_short[pi, s]:
                        d = -1
                    placed = False
                    if d != 0:
                        raw = sig_entry[pi, s]
                        L = cc if math.isnan(raw) else raw
                        L = math.floor(L / tick + 0.5) * tick
                        reject = (d == 1 and L >= oo) or (d == -1 and L <= oo)
                        if reject:
                            rejections[pi] += 1
                        else:
                            atr_s = sig_atr[pi, s]
                            stop_dist = max(slp * atr_s, tick)
                            max_qty = (cash * levp) / L
                            if riskp > 0.0:
                                desired_qty = (cash * riskp) / stop_dist
                                qty = math.floor(min(desired_qty, max_qty) / lot + 1e-9) * lot
                            else:
                                qty = math.floor(max_qty / lot + 1e-9) * lot
                            if qty > 0.0 and qty * L >= min_notional:
                                rest_active = True
                                rest_side = d
                                rest_price = L
                                rest_qty = qty
                                rest_tp = math.floor((L + d * tpp * atr_s) / tick + 0.5) * tick
                                rest_sl = math.floor((L - d * slp * atr_s) / tick + 0.5) * tick
                                rest_sig = s
                                rest_atr = atr_s
                                placed = True
                    if (not placed) and cancel_stale:
                        rest_active = False

                # rolling-drawdown circuit breaker (hour boundary, mark-to-market)
                if use_breaker:
                    eq_now = cash
                    if pos != 0.0:
                        eq_now = cash + pos * (cc - entry_px)
                    breaker[b_idx] = eq_now
                    b_idx += 1
                    if b_idx >= breaker_window:
                        b_idx = 0
                    peak = breaker[0]
                    for k in range(1, breaker_window):
                        if breaker[k] > peak:
                            peak = breaker[k]
                    dd = 1.0 - eq_now / peak if peak > 0.0 else 0.0
                    if (not halted) and pos != 0.0 and dd >= breaker_thr:
                        if pos > 0.0:
                            fill = cc - slip_taker
                        else:
                            fill = cc + slip_taker
                        cash += pos * (fill - entry_px) - abs(pos) * fill * taker_rate
                        total_fees[pi] += abs(pos) * fill * taker_rate
                        n_trades[pi] += 1
                        taker_exits[pi] += 1
                        breakers[pi] += 1
                        pos = 0.0
                        entry_sig = -1
                        halted = True
                        rest_active = False

            # exits for open positions (evaluated against active SL/TP)
            if pos != 0.0:
                sl_r = math.floor(sl_px / tick + 0.5) * tick
                tp_r = math.floor(tp_px / tick + 0.5) * tick
                if pos > 0.0:
                    sl_hit = ll <= sl_r
                    tp_hit = (not sl_hit) and (hh >= tp_r + tick)
                else:
                    sl_hit = hh >= sl_r
                    tp_hit = (not sl_hit) and (ll <= tp_r - tick)
                if sl_hit:
                    if pos > 0.0:
                        fill = sl_r - stop_slip
                        if gap_through and oo < fill:
                            fill = oo
                    else:
                        fill = sl_r + stop_slip
                        if gap_through and oo > fill:
                            fill = oo
                    cash += pos * (fill - entry_px) - abs(pos) * fill * taker_rate
                    total_fees[pi] += abs(pos) * fill * taker_rate
                    n_trades[pi] += 1
                    taker_exits[pi] += 1
                    pos = 0.0
                    entry_sig = -1
                elif tp_hit:
                    cash += pos * (tp_r - entry_px) - abs(pos) * tp_r * maker_rate
                    total_fees[pi] += abs(pos) * tp_r * maker_rate
                    n_trades[pi] += 1
                    maker_fills[pi] += 1
                    pos = 0.0
                    entry_sig = -1
                else:
                    # position still open: update peak and ratchet trailing/breakeven stops for next bar
                    if pos > 0.0:
                        if hh > peak_px:
                            peak_px = hh
                        if pos_atr > 0.0 and peak_px >= entry_px + pos_atr:
                            be_sl = entry_px + (maker_rate + taker_rate) * entry_px + tick
                            if be_sl > sl_px:
                                sl_px = be_sl
                        if trailp > 0.0 and pos_atr > 0.0:
                            trail_sl = peak_px - trailp * pos_atr
                            if trail_sl > sl_px:
                                sl_px = trail_sl
                    else:
                        if ll < peak_px:
                            peak_px = ll
                        if pos_atr > 0.0 and peak_px <= entry_px - pos_atr:
                            be_sl = entry_px - (maker_rate + taker_rate) * entry_px - tick
                            if be_sl < sl_px:
                                sl_px = be_sl
                        if trailp > 0.0 and pos_atr > 0.0:
                            trail_sl = peak_px + trailp * pos_atr
                            if trail_sl < sl_px:
                                sl_px = trail_sl

            # resting post-only entry fills
            fresh = False
            if rest_active and pos == 0.0 and not halted:
                if rest_side == 1 and ll <= rest_price - penetration:
                    cash -= rest_qty * rest_price * maker_rate
                    total_fees[pi] += rest_qty * rest_price * maker_rate
                    pos = rest_qty
                    entry_px = rest_price
                    sl_px = rest_sl
                    tp_px = rest_tp
                    pos_atr = rest_atr
                    peak_px = entry_px
                    entry_sig = rest_sig
                    entry_fill_hour = t // bph
                    maker_fills[pi] += 1
                    rest_active = False
                    fresh = True
                elif rest_side == -1 and hh >= rest_price + penetration:
                    cash -= rest_qty * rest_price * maker_rate
                    total_fees[pi] += rest_qty * rest_price * maker_rate
                    pos = -rest_qty
                    entry_px = rest_price
                    sl_px = rest_sl
                    tp_px = rest_tp
                    pos_atr = rest_atr
                    peak_px = entry_px
                    entry_sig = rest_sig
                    entry_fill_hour = t // bph
                    maker_fills[pi] += 1
                    rest_active = False
                    fresh = True

            # same-bar stop/TP check for an entry filled on this bar
            if fresh and same_bar_check and pos != 0.0:
                sl_r = math.floor(sl_px / tick + 0.5) * tick
                tp_r = math.floor(tp_px / tick + 0.5) * tick
                if pos > 0.0:
                    sl_hit = ll <= sl_r
                    tp_hit = (not sl_hit) and (hh >= tp_r + tick)
                else:
                    sl_hit = hh >= sl_r
                    tp_hit = (not sl_hit) and (ll <= tp_r - tick)
                if sl_hit:
                    if pos > 0.0:
                        fill = sl_r - stop_slip
                        if gap_through and oo < fill:
                            fill = oo
                    else:
                        fill = sl_r + stop_slip
                        if gap_through and oo > fill:
                            fill = oo
                    cash += pos * (fill - entry_px) - abs(pos) * fill * taker_rate
                    total_fees[pi] += abs(pos) * fill * taker_rate
                    n_trades[pi] += 1
                    taker_exits[pi] += 1
                    pos = 0.0
                    entry_sig = -1
                elif tp_hit:
                    cash += pos * (tp_r - entry_px) - abs(pos) * tp_r * maker_rate
                    total_fees[pi] += abs(pos) * tp_r * maker_rate
                    n_trades[pi] += 1
                    maker_fills[pi] += 1
                    pos = 0.0
                    entry_sig = -1
                else:
                    # position survived the entry bar: ratchet trailing/breakeven stops for next bar
                    if pos > 0.0:
                        if hh > peak_px:
                            peak_px = hh
                        if pos_atr > 0.0 and peak_px >= entry_px + pos_atr:
                            be_sl = entry_px + (maker_rate + taker_rate) * entry_px + tick
                            if be_sl > sl_px:
                                sl_px = be_sl
                        if trailp > 0.0 and pos_atr > 0.0:
                            trail_sl = peak_px - trailp * pos_atr
                            if trail_sl > sl_px:
                                sl_px = trail_sl
                    else:
                        if ll < peak_px:
                            peak_px = ll
                        if pos_atr > 0.0 and peak_px <= entry_px - pos_atr:
                            be_sl = entry_px - (maker_rate + taker_rate) * entry_px - tick
                            if be_sl < sl_px:
                                sl_px = be_sl
                        if trailp > 0.0 and pos_atr > 0.0:
                            trail_sl = peak_px + trailp * pos_atr
                            if trail_sl < sl_px:
                                sl_px = trail_sl

            # mark-to-market + maintenance-margin liquidation
            if pos != 0.0:
                if intrabar_liq:
                    worst = ll if pos > 0.0 else hh
                else:
                    worst = cc
                eq_worst = cash + pos * (worst - entry_px)
                maint = abs(pos) * worst * maint_rate
                if eq_worst < maint:
                    if pos > 0.0:
                        fill = worst - slip_taker
                    else:
                        fill = worst + slip_taker
                    cash += pos * (fill - entry_px) - abs(pos) * fill * taker_rate
                    total_fees[pi] += abs(pos) * fill * taker_rate
                    n_trades[pi] += 1
                    taker_exits[pi] += 1
                    liquidations[pi] += 1
                    pos = 0.0
                    entry_sig = -1
            eq = cash
            if pos != 0.0:
                eq = cash + pos * (cc - entry_px)
            equity[pi, t] = eq

    return (equity, n_trades, maker_fills, taker_exits, liquidations, rejections,
            breakers, total_fees, total_funding)


def backtest_numba(ohlcv_1m: Any, ohlcv_1h: Any, funding: Any, signals: SignalIntents,
                   genome_batch: Any, instrument: InstrumentSpec,
                   initial_equity: float = 10_000.0, bars_per_hour: int = 60,
                   fill_model: Optional[FillModelConfig] = None,
                   breaker_pct: float = 0.0,
                   breaker_window_hours: int = 24) -> Any:
    """Numba-JIT, multithreaded population backtest (counts + equity; no trade log)."""
    from core.backtester import BacktestResult

    fm = fill_model if fill_model is not None else FillModelConfig()

    o = ohlcv_1m["open"].to_numpy(dtype=np.float64)
    h = ohlcv_1m["high"].to_numpy(dtype=np.float64)
    l = ohlcv_1m["low"].to_numpy(dtype=np.float64)
    c = ohlcv_1m["close"].to_numpy(dtype=np.float64)
    m = c.shape[0]
    if m % bars_per_hour != 0:
        raise ValueError("ohlcv_1m length must be a multiple of bars_per_hour")

    fund = funding["funding_rate"].reindex(ohlcv_1h.index).ffill().fillna(0.0).to_numpy(
        dtype=np.float64)

    g = np.asarray(genome_batch, dtype=np.float64)
    regime = np.ascontiguousarray(np.asarray(signals.regime, dtype=np.float64))
    atr = np.ascontiguousarray(np.asarray(signals.atr, dtype=np.float64))
    long_e = np.ascontiguousarray(np.asarray(signals.long_entry, dtype=np.bool_))
    short_e = np.ascontiguousarray(np.asarray(signals.short_entry, dtype=np.bool_))
    entry = np.ascontiguousarray(np.asarray(signals.entry_price, dtype=np.float64))

    tp = np.ascontiguousarray(g[:, GENE_INDEX["tp_atr_mult"]])
    sl = np.ascontiguousarray(g[:, GENE_INDEX["sl_atr_mult"]])
    lev = np.ascontiguousarray(np.minimum(g[:, GENE_INDEX["target_leverage"]],
                                          instrument.max_leverage))
    max_hold = np.ascontiguousarray(g[:, GENE_INDEX["max_hold_bars"]])
    risk = np.ascontiguousarray(g[:, GENE_INDEX["risk_per_trade_pct"]])
    trail = np.ascontiguousarray(g[:, GENE_INDEX["trailing_atr_mult"]])

    breaker_window = int(breaker_window_hours) if breaker_pct > 0.0 else 1
    equity, n_trades, maker_fills, taker_exits, liq, rej, brk, fees, funding_paid = _kernel(
        o, h, l, c, fund, regime, atr, long_e, short_e, entry,
        tp, sl, lev, max_hold, risk, trail,
        float(instrument.tick_size), float(instrument.lot_size), float(instrument.min_notional),
        float(instrument.maker_fee_rate), float(instrument.taker_fee_rate),
        float(instrument.maintenance_margin_rate), float(initial_equity), int(bars_per_hour),
        int(fm.maker_penetration_ticks), int(fm.stop_slippage_ticks),
        bool(fm.stop_gap_through), bool(fm.cancel_stale_orders),
        bool(fm.max_hold_from_fill), bool(fm.same_bar_stop_check),
        bool(fm.intrabar_liquidation), int(fm.taker_slippage_ticks),
        float(breaker_pct), int(breaker_window))

    p = g.shape[0]
    equity_1h = equity[:, bars_per_hour - 1::bars_per_hour]
    return BacktestResult(
        equity_curve=equity,
        equity_1h=equity_1h,
        trades=[[] for _ in range(p)],
        num_trades=n_trades,
        maker_fills=maker_fills,
        taker_exits=taker_exits,
        liquidations=liq,
        breaker_trips=brk,
        post_only_rejections=rej,
        total_fees=fees,
        total_funding=funding_paid,
        final_equity=equity[:, -1],
    )
