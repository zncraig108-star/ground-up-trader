"""scratch/test_1h_asymmetric_boost.py — Optimized 1h Intraday Micro Futures Engine with Asymmetric R:R.

Tests:
1. 1h Signal Resolution with dynamic Breakeven Ratchet (+1.5R snap to breakeven)
2. Climax profit taking (+2.5R take 50%)
3. House-money contract scaling: contracts scale with excess capital above the soft floor ($450)
4. $450 Soft Floor & $500 / $1k / $2k ratchets
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import MicroFuturesConfig, PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.micro_futures_strategy import compute_micro_futures_signals

def test_boosted_1h():
    p_cfg = PortfolioConfig()
    bundle = load_portfolio_data(p_cfg, data_dir="data")
    
    cfg = MicroFuturesConfig(
        signal_timeframe_hours=1,
        trend_lookback_days=10,  # 10 days on 1h = 240 bars
        exit_lookback_days=3,    # 3 days on 1h = 72 bars
        atr_stop_mult=2.0,       # 2.0x ATR for tighter 1h stops
    )

    signals = compute_micro_futures_signals(bundle.market_data_1h, cfg)
    universe = signals.universe
    ts = signals.timestamps
    n_bars = len(ts)

    initial_capital = 500.0
    soft_floor = 450.0

    # Contract specifications
    contract_sizes = {"BTC-PERP": 0.01, "ETH-PERP": 0.10, "SOL-PERP": 1.00}
    margin_rates = {"BTC-PERP": 0.33, "ETH-PERP": 0.33, "SOL-PERP": 0.33}

    for excess_mult in [0.003, 0.005, 0.008, 0.012, 0.015]:
        cash = initial_capital
        equity_curve = np.zeros(n_bars)
        trades = []
        min_equity_ever = cash
        active_floor = 0.0

        pos_cnt = {p: 0 for p in universe}
        pos_entry_px = {p: 0.0 for p in universe}
        pos_stop_px = {p: 0.0 for p in universe}
        pos_trimmed = {p: False for p in universe}
        pos_be_locked = {p: False for p in universe}

        warmup = 600

        for t in range(warmup, n_bars):
            current_time = ts[t]
            unrealized = 0.0
            used_margin = 0.0

            # 1. Hourly Evaluation
            for p in universe:
                cnt = pos_cnt[p]
                if cnt != 0:
                    asig = signals.asset_signals[p]
                    px = asig.close[t]
                    hi = asig.high[t]
                    lo = asig.low[t]
                    stop = pos_stop_px[p]
                    mult = contract_sizes[p]
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
                        fee = qty * exit_px * 0.0005
                        pnl = (exit_px - pos_entry_px[p]) * qty * side - fee
                        cash += pnl
                        trades.append((p, "STOP", pnl, current_time))
                        pos_cnt[p] = 0
                        pos_entry_px[p] = 0.0
                        pos_stop_px[p] = 0.0
                        pos_trimmed[p] = False
                        pos_be_locked[p] = False
                        continue

                    entry = pos_entry_px[p]
                    atr = asig.atr[t]

                    # A. Breakeven Lock at +1.5 ATR profit
                    if not pos_be_locked[p] and atr > 0:
                        if cnt > 0 and px >= entry + 1.5 * atr:
                            pos_stop_px[p] = max(pos_stop_px[p], entry + 0.1 * atr)
                            pos_be_locked[p] = True
                        elif cnt < 0 and px <= entry - 1.5 * atr:
                            pos_stop_px[p] = min(pos_stop_px[p], entry - 0.1 * atr) if pos_stop_px[p] > 0 else entry - 0.1 * atr
                            pos_be_locked[p] = True

                    # B. Climax 50% Trim at +3.0 ATR profit
                    if not pos_trimmed[p] and abs(cnt) >= 2 and atr > 0:
                        is_climax = False
                        if cnt > 0 and px >= entry + 3.0 * atr:
                            is_climax = True
                        elif cnt < 0 and px <= entry - 3.0 * atr:
                            is_climax = True

                        if is_climax:
                            trim_cnt = abs(cnt) // 2
                            trim_qty = trim_cnt * mult
                            side = 1 if cnt > 0 else -1
                            fee = trim_qty * px * 0.0002
                            pnl = (px - entry) * trim_qty * side - fee
                            cash += pnl
                            trades.append((p, "CLIMAX", pnl, current_time))
                            pos_cnt[p] = (abs(cnt) - trim_cnt) * side
                            pos_trimmed[p] = True
                            pos_stop_px[p] = entry + (0.5 * atr * side)

                    # C. Trailing Stop
                    if cnt > 0:
                        ex_l = asig.trailing_exit_low[t]
                        if not np.isnan(ex_l) and ex_l > pos_stop_px[p]:
                            pos_stop_px[p] = ex_l
                    elif cnt < 0:
                        ex_h = asig.trailing_exit_high[t]
                        if not np.isnan(ex_h) and (pos_stop_px[p] == 0 or ex_h < pos_stop_px[p]):
                            pos_stop_px[p] = ex_h

                    side = 1 if cnt > 0 else -1
                    unrealized += (px - entry) * qty * side
                    used_margin += qty * px * margin_rates[p]

            # 5% USDC Staked Yield
            idle_cash = max(0.0, cash - used_margin)
            cash += idle_cash * (0.05 / 8760.0)

            current_equity = cash + unrealized
            min_equity_ever = min(min_equity_ever, current_equity)

            # Ratchet milestones
            if current_equity >= 650.0 and active_floor < 500.0:
                active_floor = 500.0
            elif current_equity >= 1500.0 and active_floor < 1000.0:
                active_floor = 1000.0
            elif current_equity >= 3000.0 and active_floor < 2000.0:
                active_floor = 2000.0

            # Floor protection
            if active_floor > 0 and current_equity <= active_floor:
                for p in universe:
                    cnt = pos_cnt[p]
                    if cnt != 0:
                        px = signals.asset_signals[p].close[t]
                        mult = contract_sizes[p]
                        qty = abs(cnt) * mult
                        side = 1 if cnt > 0 else -1
                        fee = qty * px * 0.0005
                        pnl = (px - pos_entry_px[p]) * qty * side - fee
                        cash += pnl
                        pos_cnt[p] = 0
                current_equity = cash
                unrealized = 0.0

            # 2. Entries on 1h bars
            can_enter = (active_floor == 0 or current_equity > active_floor) and (current_equity >= soft_floor)

            if can_enter:
                is_defensive = (current_equity < 500.0)
                active_pos = sum(1 for p in universe if pos_cnt[p] != 0)
                max_slots = 1 if is_defensive else 2
                slots = max_slots - active_pos

                if slots > 0:
                    # Excess scaling
                    excess = max(0.0, current_equity - soft_floor)
                    extra = int(math.floor(excess * excess_mult))
                    base_cnt = 1 + extra
                    if is_defensive:
                        base_cnt = 1

                    tradable = ["SOL-PERP", "ETH-PERP"]
                    if current_equity >= 1000.0:
                        tradable.append("BTC-PERP")

                    for p in tradable:
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
                        mult = contract_sizes[p]

                        if np.isnan(px) or np.isnan(atr_val) or atr_val <= 0:
                            continue

                        req_er = 0.38 if is_defensive else 0.28
                        if er_val < req_er:
                            continue

                        stop_dist = max(2.0 * atr_val, px * 0.03)
                        cnt_target = max(1, base_cnt // 2) if p == "BTC-PERP" else base_cnt
                        if is_defensive:
                            cnt_target = 1

                        notional = cnt_target * mult * px
                        margin_req = notional * margin_rates[p]

                        if margin_req <= current_equity * 0.50:
                            # LONG
                            if px > up and px > e50 and slp > 0:
                                fee = notional * 0.0002
                                cash -= fee
                                pos_cnt[p] = cnt_target
                                pos_entry_px[p] = px
                                pos_stop_px[p] = px - stop_dist
                                pos_trimmed[p] = False
                                pos_be_locked[p] = False
                                slots -= 1

                            # SHORT
                            elif px < dn and px < e50 and slp < 0:
                                fee = notional * 0.0002
                                cash -= fee
                                pos_cnt[p] = -cnt_target
                                pos_entry_px[p] = px
                                pos_stop_px[p] = px + stop_dist
                                pos_trimmed[p] = False
                                pos_be_locked[p] = False
                                slots -= 1

            equity_curve[t] = max(0.0, current_equity)

        eq_arr = equity_curve[warmup:]
        final_eq = eq_arr[-1]
        mult = final_eq / initial_capital
        peaks = np.maximum.accumulate(eq_arr)
        dds = (peaks - eq_arr) / peaks
        max_dd = np.max(dds)
        deposit_dd = max(0.0, (initial_capital - min_equity_ever) / initial_capital)
        wins = [tr for tr in trades if tr[2] > 0]
        wr = len(wins) / max(len(trades), 1)

        print(f"ExcessMult={excess_mult:.4f} => Final=${final_eq:9,.2f} ({mult:5.2f}x) | Peak MaxDD={max_dd:5.2%} | Deposit DD={deposit_dd:5.2%} (MinEq=${min_equity_ever:.2f}) | Trades={len(trades)} | WR={wr:4.1%}")

if __name__ == "__main__":
    test_boosted_1h()
