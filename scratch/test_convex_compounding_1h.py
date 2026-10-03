"""scratch/test_convex_compounding_1h.py — Test Convex Excess-Capital Sizing on 1h NYS Engine.

Hypothesis H-012:
Replace static $250 tier steps with dynamic integer contract sizing based on excess capital above active floor.
Verify:
1. Deposit DD <= 10% (min equity >= $450 at all times).
2. Ratchet floors ($500 @ $650, $1k @ $1.5k, $2k @ $3k) never breached once locked.
3. In-Sample vs Out-of-Sample return scaling.
4. CFTC max leverage <= 3.0x.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import MicroFuturesConfig
from core.portfolio_data import load_portfolio_data
from core.micro_futures_strategy import compute_micro_futures_signals
from core.micro_futures_backtester import backtest_micro_futures, MICRO_CONTRACT_SIZES, MARGIN_RATES

def run_convex_test():
    cfg_base = MicroFuturesConfig(signal_timeframe_hours=1)
    bundle = load_portfolio_data(cfg_base, data_dir="data")
    signals = compute_micro_futures_signals(bundle.market_data_1h, cfg_base)

    ts = signals.timestamps
    universe = cfg_base.universe
    n_bars = len(ts)
    hourly_usdc_rate = cfg_base.usdc_yield_apr / 8760.0

    print("=====================================================================")
    print("TESTING CONVEX EXCESS-CAPITAL COMPANDING ON 1-HOUR INTRADAY ENGINE")
    print("=====================================================================\n")

    for risk_fraction in [0.08, 0.12, 0.15, 0.20, 0.25, 0.30]:
        initial_capital = 500.0
        soft_floor = 450.0
        active_floor = 0.0
        cash = initial_capital
        min_equity_ever = cash
        total_fees = 0.0
        total_yield = 0.0

        pos_cnt = {p: 0 for p in universe}
        pos_entry_px = {p: 0.0 for p in universe}
        pos_stop_px = {p: 0.0 for p in universe}
        pos_be_locked = {p: False for p in universe}
        pos_entry_bar = {p: 0 for p in universe}
        pos_trimmed = {p: False for p in universe}

        trades = []
        equity_curve = np.zeros(n_bars, dtype=np.float64)

        for t in range(1200, n_bars):
            current_time = ts[t]
            unrealized = 0.0
            used_margin = 0.0

            # 1. Stop loss & exits
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
                    side = 1 if cnt > 0 else -1

                    hit_stop = False
                    exit_px = 0.0
                    if cnt > 0 and lo <= stop and stop > 0:
                        exit_px = min(px, stop)
                        hit_stop = True
                    elif cnt < 0 and hi >= stop and stop > 0:
                        exit_px = max(px, stop)
                        hit_stop = True

                    if hit_stop:
                        fee = qty * exit_px * 0.0005
                        pnl = (exit_px - pos_entry_px[p]) * qty * side - fee
                        cash += pnl
                        total_fees += fee
                        trades.append((p, "STOP", pnl, current_time))
                        pos_cnt[p] = 0
                        pos_entry_px[p] = 0.0
                        pos_stop_px[p] = 0.0
                        pos_be_locked[p] = False
                        pos_trimmed[p] = False
                        continue

                    # Breakeven snap at +2.0 ATR
                    atr = asig.atr[t]
                    entry = pos_entry_px[p]
                    if not pos_be_locked[p] and not np.isnan(atr) and atr > 0:
                        if cnt > 0 and px >= entry + 2.0 * atr:
                            pos_stop_px[p] = max(pos_stop_px[p], entry + 0.05 * atr)
                            pos_be_locked[p] = True
                        elif cnt < 0 and px <= entry - 2.0 * atr:
                            pos_stop_px[p] = min(pos_stop_px[p], entry - 0.05 * atr) if pos_stop_px[p] > 0 else entry - 0.05 * atr
                            pos_be_locked[p] = True

                    # Climax trim on 2+ contracts at +2.8 ATR
                    s20 = asig.sma_20[t]
                    if not pos_trimmed[p] and abs(cnt) >= 2 and not np.isnan(s20) and not np.isnan(atr) and atr > 0:
                        if (cnt > 0 and (px - s20) >= 2.8 * atr and px > entry) or (cnt < 0 and (s20 - px) >= 2.8 * atr and px < entry):
                            trim_cnt = abs(cnt) // 2
                            trim_qty = trim_cnt * mult
                            fee = trim_qty * px * 0.0002
                            pnl = (px - entry) * trim_qty * side - fee
                            cash += pnl
                            total_fees += fee
                            trades.append((p, "CLIMAX", pnl, current_time))
                            pos_cnt[p] = (abs(cnt) - trim_cnt) * side
                            pos_trimmed[p] = True
                            pos_stop_px[p] = entry

                    # Trailing stop
                    if cnt > 0:
                        ex_l = asig.trailing_exit_low[t]
                        if not np.isnan(ex_l) and ex_l > pos_stop_px[p]:
                            pos_stop_px[p] = ex_l
                    elif cnt < 0:
                        ex_h = asig.trailing_exit_high[t]
                        if not np.isnan(ex_h) and (pos_stop_px[p] == 0 or ex_h < pos_stop_px[p]):
                            pos_stop_px[p] = ex_h

                    unrealized += (px - pos_entry_px[p]) * qty * side
                    used_margin += qty * px * MARGIN_RATES.get(p, 0.33)

            idle_cash = max(0.0, cash - used_margin)
            hourly_yield = idle_cash * hourly_usdc_rate
            cash += hourly_yield
            total_yield += hourly_yield

            current_equity = cash + unrealized
            min_equity_ever = min(min_equity_ever, current_equity)

            # Ratchet logic
            if current_equity >= 650.0 and active_floor < 500.0:
                active_floor = 500.0
            elif current_equity >= 1500.0 and active_floor < 1000.0:
                active_floor = 1000.0
            elif current_equity >= 3000.0 and active_floor < 2000.0:
                active_floor = 2000.0

            # Floor defense trigger
            if active_floor > 0 and current_equity <= active_floor:
                for p in universe:
                    cnt = pos_cnt[p]
                    if cnt != 0:
                        px = signals.asset_signals[p].close[t]
                        mult = MICRO_CONTRACT_SIZES.get(p, 1.0)
                        qty = abs(cnt) * mult
                        side = 1 if cnt > 0 else -1
                        fee = qty * px * 0.0005
                        cash += (px - pos_entry_px[p]) * qty * side - fee
                        total_fees += fee
                        pos_cnt[p] = 0
                current_equity = cash
                unrealized = 0.0

            # 2. Entries on 1h bars
            can_enter = (active_floor == 0 or current_equity > active_floor) and (current_equity >= soft_floor)
            if can_enter:
                # 50d EMA exits
                for p in universe:
                    cnt = pos_cnt[p]
                    if cnt != 0:
                        asig = signals.asset_signals[p]
                        px = asig.close[t]
                        e50 = asig.ema_50d[t]
                        mult = MICRO_CONTRACT_SIZES.get(p, 1.0)
                        qty = abs(cnt) * mult
                        side = 1 if cnt > 0 else -1
                        if (cnt > 0 and px < e50) or (cnt < 0 and px > e50):
                            fee = qty * px * 0.0002
                            cash += (px - pos_entry_px[p]) * qty * side - fee
                            total_fees += fee
                            trades.append((p, "EMA_EXIT", (px - pos_entry_px[p]) * qty * side - fee, current_time))
                            pos_cnt[p] = 0
                            pos_trimmed[p] = False
                            pos_be_locked[p] = False

                is_defensive = (current_equity < 500.0)
                active_pos = sum(1 for p in universe if pos_cnt[p] != 0)
                max_slots = 1 if is_defensive else cfg_base.max_concurrent_positions
                slots = max_slots - active_pos

                if slots > 0:
                    tradable = ["SOL-PERP", "ETH-PERP"]
                    if current_equity >= 1000.0:
                        tradable.append("BTC-PERP")

                    # Rank by 20-day momentum
                    lookback_bars = 20 * 24
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

                        req_er = cfg_base.er_min_defensive if is_defensive else cfg_base.er_min_normal
                        if er_val < req_er:
                            continue

                        stop_dist = max(cfg_base.atr_stop_mult * atr_val, px * 0.035)
                        dollar_risk_per_contract = stop_dist * mult

                        # CONVEX DYNAMIC CONTRACT SIZING:
                        if is_defensive:
                            cnt_target = 1
                        else:
                            # Excess capital above locked floor (or $500 if no floor active yet)
                            effective_floor = max(active_floor, 450.0)
                            excess_cap = max(20.0, current_equity - effective_floor)
                            
                            # Risk a calibrated fraction of excess capital per available slot
                            risk_budget = (excess_cap * risk_fraction) / max(slots, 1)
                            
                            # Safety cap: Stopout of this position alone must not lose more than 35% of excess capital
                            risk_budget = min(risk_budget, excess_cap * 0.35)
                            
                            # Calculate integer contracts
                            cnt_target = max(1, int(risk_budget // dollar_risk_per_contract))

                        if p == "BTC-PERP":
                            cnt_target = max(1, cnt_target // 2)

                        notional = cnt_target * mult * px
                        margin_needed = notional * MARGIN_RATES.get(p, 0.33)

                        # Check margin availability & CFTC 3.0x leverage cap
                        avail_equity_for_margin = current_equity * 0.65
                        if margin_needed <= avail_equity_for_margin and (notional / current_equity) <= 3.0:
                            # LONG BREAKOUT
                            if px > up and px > e50 and slp > 0:
                                fee = notional * 0.0002
                                cash -= fee
                                total_fees += fee
                                pos_cnt[p] = cnt_target
                                pos_entry_px[p] = px
                                pos_stop_px[p] = px - stop_dist
                                pos_trimmed[p] = False
                                pos_be_locked[p] = False
                                slots -= 1

                            # SHORT BREAKDOWN
                            elif cfg_base.allow_shorting and px < dn and px < e50 and slp < 0 and er_val >= (req_er + cfg_base.short_er_boost):
                                fee = notional * 0.0002
                                cash -= fee
                                total_fees += fee
                                pos_cnt[p] = -cnt_target
                                pos_entry_px[p] = px
                                pos_stop_px[p] = px + stop_dist
                                pos_trimmed[p] = False
                                pos_be_locked[p] = False
                                slots -= 1

            equity_curve[t] = current_equity

        equity_curve[:1200] = initial_capital
        final_eq = equity_curve[-1]
        mult = final_eq / initial_capital
        peaks = np.maximum.accumulate(equity_curve)
        dds = (peaks - equity_curve) / peaks
        max_dd = np.max(dds)
        deposit_dd = max(0.0, (initial_capital - min_equity_ever) / initial_capital)
        wins = [tr for tr in trades if tr[2] > 0]
        wr = len(wins) / max(len(trades), 1)

        # In-sample vs Out-of-sample breakdown (split at 2024-01-01)
        split_idx = ts.searchsorted(pd.Timestamp("2024-01-01"))
        is_eq = equity_curve[split_idx]
        is_mult = is_eq / initial_capital
        oos_mult = final_eq / is_eq

        print(f"RiskFrac={risk_fraction:.2f} | Final=${final_eq:10,.2f} ({mult:6.2f}x) | PeakDD={max_dd:5.2%} | DepositDD={deposit_dd:5.2%} (MinEq=${min_equity_ever:6.2f}) | Trades={len(trades)} (WR={wr:4.1%}) | IS={is_mult:5.2f}x | OOS={oos_mult:5.2f}x")

if __name__ == "__main__":
    run_convex_test()
