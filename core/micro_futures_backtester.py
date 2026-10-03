"""core/micro_futures_backtester.py — Simulation Engine for $500 Micro Futures.

Simulates CFTC-regulated crypto micro futures trading under US / NYS constraints:
- Starts with $500.00 initial capital.
- Strictly whole integer contracts (no fractional futures).
- $450 Soft Floor: When equity is between $450 and $500, enters a defensive posture
  (requires ER >= 0.38, max 1 contract, maker fees only).
- House Money Ratchet: Locks $500 floor once equity >= $650, $1k floor at $1,500, $2k floor at $3,000.
- Allow Shorting: Both Long and Short trades supported.
- Realistic 5.0% APY USDC yield on unallocated margin cash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from config import MicroFuturesConfig
from core.micro_futures_strategy import MicroFuturesSignals, MicroAssetSignals


@dataclass
class MicroTrade:
    """Trade record for micro futures."""
    product_id: str
    side: int                  # +1 (Long), -1 (Short)
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_px: float
    exit_px: float
    contracts: int             # integer number of contracts
    qty: float                 # actual underlying crypto quantity
    pnl: float                 # realized net PnL in USD
    fees: float                # total maker/taker fees paid
    exit_reason: str           # "STOP" | "CLIMAX" | "EMA_EXIT" | "FLOOR_DEFENSE"


@dataclass
class MicroBacktestResult:
    """Full backtest results for the 10x micro futures portfolio."""
    timestamps: pd.DatetimeIndex
    equity_curve: np.ndarray
    cash_curve: np.ndarray
    floor_curve: np.ndarray
    trades: List[MicroTrade]
    initial_equity: float
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_peak_drawdown: float
    deposit_drawdown: float     # Drawdown strictly against the $500 initial deposit
    min_equity_ever: float
    total_fees: float
    usdc_yield_earned: float
    win_rate: float
    is_mult: float              # In-Sample multiplier (2021-2023)
    is_dd: float                # In-Sample max peak drawdown
    oos_mult: float             # Out-of-Sample multiplier (2024-2026)
    oos_dd: float               # Out-of-Sample max peak drawdown


# Contract multipliers for CFTC-regulated micro/nano futures
MICRO_CONTRACT_SIZES = {
    "BTC-PERP": 0.01,   # 1 Nano contract = 0.01 BTC (BTI)
    "ETH-PERP": 0.10,   # 1 Micro contract = 0.10 ETH (MET)
    "SOL-PERP": 1.00,   # 1 Micro contract = 1.00 SOL
}

# Margin requirement per contract (approx 33% margin = 3x max leverage)
MARGIN_RATES = {
    "BTC-PERP": 0.33,
    "ETH-PERP": 0.33,
    "SOL-PERP": 0.33,
}


def backtest_micro_futures(
    signals: MicroFuturesSignals,
    cfg: MicroFuturesConfig,
) -> MicroBacktestResult:
    """Execute micro futures simulation with integer contracts and soft capital floor."""
    universe = signals.universe
    ts = signals.timestamps
    n_bars = len(ts)
    tf_hours = cfg.signal_timeframe_hours

    initial_capital = float(cfg.initial_equity)
    cash = initial_capital
    soft_floor = float(cfg.soft_floor_capital)
    active_floor = 0.0  # Ratchets up to 500, 1000, 2000 as equity builds

    equity_curve = np.zeros(n_bars, dtype=np.float64)
    cash_curve = np.zeros(n_bars, dtype=np.float64)
    floor_curve = np.zeros(n_bars, dtype=np.float64)

    pos_cnt = {p: 0 for p in universe}       # signed integer contracts
    pos_entry_px = {p: 0.0 for p in universe}
    pos_entry_bar = {p: -1 for p in universe}
    pos_stop_px = {p: 0.0 for p in universe}
    pos_trimmed = {p: False for p in universe}
    pos_be_locked = {p: False for p in universe}

    all_trades: List[MicroTrade] = []
    total_fees_paid = 0.0
    total_usdc_yield = 0.0
    min_equity_ever = initial_capital
    last_stop_asset = ""
    last_stop_bar = -999

    hourly_usdc_rate = cfg.usdc_yield_apr / 8760.0

    # Warmup period: 100 days
    warmup_bars = min(n_bars // 4, 1200)

    for t in range(warmup_bars, n_bars):
        current_time = ts[t]
        unrealized = 0.0
        used_margin = 0.0

        # --- 1. Hourly Evaluation & Stop Checks ---
        for p in universe:
            cnt = pos_cnt[p]
            if cnt != 0:
                asig = signals.asset_signals[p]
                px = asig.close[t]
                hi = asig.high[t]
                lo = asig.low[t]
                stop = pos_stop_px[p]
                mult = MICRO_CONTRACT_SIZES.get(p, 1.0)
                qty = abs(cnt) * mult

                hit_stop = False
                exit_px = 0.0

                if cnt > 0 and lo <= stop and stop > 0:
                    exit_px = min(px, stop)
                    hit_stop = True
                elif cnt < 0 and hi >= stop and stop > 0:
                    exit_px = max(px, stop)
                    hit_stop = True

                if hit_stop and exit_px > 0:
                    side = 1 if cnt > 0 else -1
                    fee = qty * exit_px * 0.0005  # Taker exit fee
                    pnl = (exit_px - pos_entry_px[p]) * qty * side - fee
                    cash += pnl
                    total_fees_paid += fee

                    all_trades.append(MicroTrade(
                        product_id=p,
                        side=side,
                        entry_time=ts[pos_entry_bar[p]],
                        exit_time=current_time,
                        entry_px=pos_entry_px[p],
                        exit_px=exit_px,
                        contracts=abs(cnt),
                        qty=qty,
                        pnl=pnl,
                        fees=fee,
                        exit_reason="STOP",
                    ))
                    pos_cnt[p] = 0
                    pos_entry_px[p] = 0.0
                    pos_stop_px[p] = 0.0
                    pos_trimmed[p] = False
                    pos_be_locked[p] = False
                    last_stop_asset = p
                    last_stop_bar = t
                    continue

                # Dynamic Breakeven Snap
                if cfg.be_lock_enabled and not pos_be_locked[p]:
                    atr = asig.atr[t]
                    entry = pos_entry_px[p]
                    if not np.isnan(atr) and atr > 0:
                        if cnt > 0 and px >= entry + cfg.be_lock_mult_atr * atr:
                            pos_stop_px[p] = max(pos_stop_px[p], entry + 0.05 * atr)
                            pos_be_locked[p] = True
                        elif cnt < 0 and px <= entry - cfg.be_lock_mult_atr * atr:
                            pos_stop_px[p] = min(pos_stop_px[p], entry - 0.05 * atr) if pos_stop_px[p] > 0 else entry - 0.05 * atr
                            pos_be_locked[p] = True

                # Parabolic Climax Trimming (lock 50% at +2.8 ATR on runners)
                if cfg.climax_trim_enabled and not pos_trimmed[p] and abs(cnt) >= 2:
                    s20 = asig.sma_20[t]
                    atr_val = asig.atr[t]
                    if not np.isnan(s20) and not np.isnan(atr_val) and atr_val > 0:
                        is_climax = False
                        if cnt > 0 and (px - s20) >= cfg.climax_trim_atr * atr_val and px > pos_entry_px[p]:
                            is_climax = True
                        elif cnt < 0 and (s20 - px) >= cfg.climax_trim_atr * atr_val and px < pos_entry_px[p]:
                            is_climax = True

                        if is_climax:
                            trim_cnt = abs(cnt) // 2
                            trim_qty = trim_cnt * mult
                            side = 1 if cnt > 0 else -1
                            fee = trim_qty * px * 0.0002  # Maker fee on limit trim
                            pnl = (px - pos_entry_px[p]) * trim_qty * side - fee
                            cash += pnl
                            total_fees_paid += fee

                            all_trades.append(MicroTrade(
                                product_id=p,
                                side=side,
                                entry_time=ts[pos_entry_bar[p]],
                                exit_time=current_time,
                                entry_px=pos_entry_px[p],
                                exit_px=px,
                                contracts=trim_cnt,
                                qty=trim_qty,
                                pnl=pnl,
                                fees=fee,
                                exit_reason="CLIMAX",
                            ))
                            pos_cnt[p] = (abs(cnt) - trim_cnt) * side
                            pos_trimmed[p] = True
                            pos_stop_px[p] = pos_entry_px[p]  # Lock stop to breakeven!

                # Dynamic Trailing Stop Ratchet
                if cnt > 0:
                    ex_l = asig.trailing_exit_low[t]
                    if not np.isnan(ex_l) and ex_l > pos_stop_px[p]:
                        pos_stop_px[p] = ex_l
                elif cnt < 0:
                    ex_h = asig.trailing_exit_high[t]
                    if not np.isnan(ex_h) and (pos_stop_px[p] == 0 or ex_h < pos_stop_px[p]):
                        pos_stop_px[p] = ex_h

                side = 1 if cnt > 0 else -1
                unrealized += (px - pos_entry_px[p]) * qty * side
                used_margin += qty * px * MARGIN_RATES.get(p, 0.33)

        # Accrue realistic 5% APY USDC yield on unallocated margin
        idle_cash = max(0.0, cash - used_margin)
        hourly_yield = idle_cash * hourly_usdc_rate
        cash += hourly_yield
        total_usdc_yield += hourly_yield

        current_equity = cash + unrealized
        min_equity_ever = min(min_equity_ever, current_equity)

        # --- 2. House Money Ratchet Logic ---
        # Milestone 1: Lock $500 principal once equity hits $650 (+30% cushion)
        if current_equity >= cfg.lock_floor_at_equity and active_floor < 500.0:
            active_floor = 500.0
        # Milestone 2: Ratchet floor to $1,000 once equity hits $1,500 (+200%)
        elif current_equity >= 1500.0 and active_floor < 1000.0:
            active_floor = 1000.0
        # Milestone 3: Ratchet floor to $2,000 once equity hits $3,000 (+500%)
        elif current_equity >= 3000.0 and active_floor < 2000.0:
            active_floor = 2000.0

        # Floor Defense: If floor is active and equity touches floor, close out to protect capital
        if active_floor > 0 and current_equity <= active_floor:
            for p in universe:
                cnt = pos_cnt[p]
                if cnt != 0:
                    px = signals.asset_signals[p].close[t]
                    mult = MICRO_CONTRACT_SIZES.get(p, 1.0)
                    qty = abs(cnt) * mult
                    side = 1 if cnt > 0 else -1
                    fee = qty * px * 0.0005
                    pnl = (px - pos_entry_px[p]) * qty * side - fee
                    cash += pnl
                    total_fees_paid += fee
                    all_trades.append(MicroTrade(
                        product_id=p, side=side, entry_time=ts[pos_entry_bar[p]], exit_time=current_time,
                        entry_px=pos_entry_px[p], exit_px=px, contracts=abs(cnt), qty=qty, pnl=pnl,
                        fees=fee, exit_reason="FLOOR_DEFENSE"
                    ))
                    pos_cnt[p] = 0
                    pos_be_locked[p] = False
            current_equity = cash
            unrealized = 0.0

        # --- 3. Discrete Signal Execution ---
        is_rebal_step = (current_time.hour % tf_hours == 0)

        if is_rebal_step:
            # 50d EMA Trend Exits
            for p in universe:
                cnt = pos_cnt[p]
                if cnt != 0:
                    asig = signals.asset_signals[p]
                    px = asig.close[t]
                    e50 = asig.ema_50d[t]
                    mult = MICRO_CONTRACT_SIZES.get(p, 1.0)
                    qty = abs(cnt) * mult
                    side = 1 if cnt > 0 else -1

                    trend_exit = False
                    if cnt > 0 and px < e50:
                        trend_exit = True
                    elif cnt < 0 and px > e50:
                        trend_exit = True

                    if trend_exit:
                        fee = qty * px * 0.0002  # Maker exit on trend change
                        pnl = (px - pos_entry_px[p]) * qty * side - fee
                        cash += pnl
                        total_fees_paid += fee
                        all_trades.append(MicroTrade(
                            product_id=p, side=side, entry_time=ts[pos_entry_bar[p]], exit_time=current_time,
                            entry_px=pos_entry_px[p], exit_px=px, contracts=abs(cnt), qty=qty, pnl=pnl,
                            fees=fee, exit_reason="EMA_EXIT"
                        ))
                        pos_cnt[p] = 0
                        pos_trimmed[p] = False
                        pos_be_locked[p] = False

            # Position Entry Check
            # If active floor breached, do not open new positions until cash recovers
            can_enter = (active_floor == 0 or current_equity > active_floor) and (current_equity >= soft_floor)

            if can_enter:
                # Check if in DEFENSIVE REGIME (equity between $450 and $500)
                is_defensive = (current_equity < 500.0)

                active_positions = sum(1 for p in universe if pos_cnt[p] != 0)
                max_slots = 1 if is_defensive else cfg.max_concurrent_positions
                slots = max_slots - active_positions

                if slots > 0:
                    excess = max(0.0, current_equity - 500.0)
                    add_cnt = int(excess // cfg.compounding_tier_step)
                    base_contracts = 1 if is_defensive else (1 + add_cnt)

                    tradable = ["SOL-PERP", "ETH-PERP"]
                    if current_equity >= 1000.0:
                        tradable.append("BTC-PERP")

                    # Relative strength momentum ranking (20-day return)
                    lookback_bars = 20 * (24 // tf_hours)
                    scored = []
                    for p in tradable:
                        if pos_cnt[p] != 0:
                            continue
                        asig = signals.asset_signals[p]
                        px = asig.close[t]
                        px_prev = asig.close[max(0, t - lookback_bars)]
                        mom = (px - px_prev) / px_prev if px_prev > 0 else 0.0
                        scored.append((abs(mom), p))
                    scored.sort(reverse=True)

                    for _, p in scored:
                        if slots <= 0:
                            break
                        if pos_cnt[p] != 0:
                            continue

                        # Same-asset cooldown: don't immediately re-enter asset that just stopped out
                        if cfg.cooldown_hours > 0 and p == last_stop_asset and (t - last_stop_bar) < cfg.cooldown_hours:
                            continue

                        asig = signals.asset_signals[p]
                        px = asig.close[t]
                        up = asig.upper_breakout[t]
                        dn = asig.lower_breakout[t]
                        e50 = asig.ema_50d[t]
                        slp = asig.ema_slope[t]
                        er_val = asig.er[t]
                        atr_val = asig.atr[t]
                        mult = MICRO_CONTRACT_SIZES.get(p, 1.0)

                        if np.isnan(px) or np.isnan(atr_val) or atr_val <= 0:
                            continue

                        # In defensive posture: require ER >= er_min_defensive (0.38)
                        req_er = cfg.er_min_defensive if is_defensive else cfg.er_min_normal
                        if er_val < req_er:
                            continue

                        stop_dist = max(cfg.atr_stop_mult * atr_val, px * 0.035)
                        
                        # Size integer contracts:
                        cnt_target = max(1, base_contracts // 2) if p == "BTC-PERP" else base_contracts
                        if is_defensive:
                            cnt_target = 1

                        notional = cnt_target * mult * px
                        margin_needed = notional * MARGIN_RATES.get(p, 0.33)

                        # Margin and CFTC leverage compliance limit
                        if margin_needed <= (current_equity * 0.45) and (notional / current_equity) <= cfg.max_portfolio_leverage:
                            # LONG BREAKOUT
                            if px > up and px > e50 and slp > 0:
                                fee = notional * 0.0002  # Maker limit entry
                                cash -= fee
                                total_fees_paid += fee
                                pos_cnt[p] = cnt_target
                                pos_entry_px[p] = px
                                pos_entry_bar[p] = t
                                pos_stop_px[p] = px - stop_dist
                                pos_trimmed[p] = False
                                pos_be_locked[p] = False
                                slots -= 1

                            # SHORT BREAKDOWN (SHORTING ENABLED WITH ER BOOST!)
                            elif cfg.allow_shorting and px < dn and px < e50 and slp < 0 and er_val >= (req_er + cfg.short_er_boost):
                                fee = notional * 0.0002  # Maker limit entry
                                cash -= fee
                                total_fees_paid += fee
                                pos_cnt[p] = -cnt_target
                                pos_entry_px[p] = px
                                pos_entry_bar[p] = t
                                pos_stop_px[p] = px + stop_dist
                                pos_trimmed[p] = False
                                pos_be_locked[p] = False
                                slots -= 1

        equity_curve[t] = max(0.0, current_equity)
        cash_curve[t] = cash
        floor_curve[t] = active_floor

    # Fill warmup curve
    equity_curve[:warmup_bars] = initial_capital
    cash_curve[:warmup_bars] = initial_capital
    floor_curve[:warmup_bars] = 0.0

    final_eq = float(equity_curve[-1])
    total_ret = float(final_eq / initial_capital - 1.0)
    days = (ts[-1] - ts[0]).total_seconds() / 86400.0
    cagr = float((final_eq / initial_capital) ** (365.25 / max(days, 1.0)) - 1.0) if final_eq > 0 else -1.0

    # Sharpe & Sortino (hourly returns)
    eq_ser = pd.Series(equity_curve, index=ts)
    hourly_rets = eq_ser.pct_change().fillna(0.0).to_numpy()
    m_ret = np.mean(hourly_rets)
    s_ret = np.std(hourly_rets)
    d_ret = np.std(np.minimum(hourly_rets, 0.0))
    sharpe = float(m_ret / s_ret * math.sqrt(8760.0)) if s_ret > 1e-9 else 0.0
    sortino = float(m_ret / d_ret * math.sqrt(8760.0)) if d_ret > 1e-9 else 0.0

    peaks = np.maximum.accumulate(equity_curve)
    dds = (peaks - equity_curve) / np.maximum(peaks, 1e-6)
    max_peak_dd = float(np.max(dds))

    deposit_dd = float(max(0.0, (initial_capital - min_equity_ever) / initial_capital))

    wins = [tr for tr in all_trades if tr.pnl > 0]
    wr = float(len(wins) / max(len(all_trades), 1))

    # Split IS / OOS (2021-2023 vs 2024-2026)
    split_idx = warmup_bars + int((n_bars - warmup_bars) * 0.5)
    is_eq = equity_curve[warmup_bars:split_idx]
    oos_eq = equity_curve[split_idx:]

    is_mult = float(is_eq[-1] / initial_capital) if len(is_eq) > 0 else 1.0
    is_p = np.maximum.accumulate(is_eq)
    is_dd = float(np.max((is_p - is_eq) / np.maximum(is_p, 1e-6))) if len(is_p) > 0 else 0.0

    oos_mult = float(oos_eq[-1] / is_eq[-1]) if len(oos_eq) > 0 and is_eq[-1] > 0 else 1.0
    oos_p = np.maximum.accumulate(oos_eq)
    oos_dd = float(np.max((oos_p - oos_eq) / np.maximum(oos_p, 1e-6))) if len(oos_p) > 0 else 0.0

    return MicroBacktestResult(
        timestamps=ts,
        equity_curve=equity_curve,
        cash_curve=cash_curve,
        floor_curve=floor_curve,
        trades=all_trades,
        initial_equity=initial_capital,
        final_equity=final_eq,
        total_return=total_ret,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_peak_drawdown=max_peak_dd,
        deposit_drawdown=deposit_dd,
        min_equity_ever=float(min_equity_ever),
        total_fees=total_fees_paid,
        usdc_yield_earned=total_usdc_yield,
        win_rate=wr,
        is_mult=is_mult,
        is_dd=is_dd,
        oos_mult=oos_mult,
        oos_dd=oos_dd,
    )
