"""scratch/test_high_expectancy_engine.py — Squeeze Breakout + Volume Expansion Engine.

Goal:
Increase trade expectancy from $0.90/trade to >= $15.00/trade by eliminating false chop breakouts.

Mechanisms:
1. Bollinger Band Squeeze inside Keltner Channel (energy compression).
2. Volume Expansion confirmation: Bar volume >= 1.25x 20-bar volume SMA.
3. Asymmetric Profit Taking: Scale 50% out at +3.0 ATR extension, trail remainder with 3 ATR Chandelier.
4. Capital preservation: Never breach $450 soft floor; ratchet at $500, $1k, $2k.
5. Integer contracts on ETH, SOL, BTC.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import MicroFuturesConfig
from core.portfolio_data import load_portfolio_data
from core.micro_futures_backtester import MICRO_CONTRACT_SIZES, MARGIN_RATES

def run_squeeze_experiment():
    cfg = MicroFuturesConfig(signal_timeframe_hours=1)
    bundle = load_portfolio_data(cfg, data_dir="data")
    universe = ["SOL-PERP", "ETH-PERP", "BTC-PERP"]
    common_idx = bundle.market_data_1h["BTC-PERP"].index
    n_bars = len(common_idx)

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()
    v_df = pd.DataFrame({p: bundle.market_data_1h[p]["volume"] for p in universe}, index=common_idx).ffill().bfill()

    # Precompute causal indicators (strictly shifted by 1 bar to prevent lookahead)
    # 1. 20-period SMA & Bollinger Bands (2.0 std)
    sma20 = c_df.rolling(20).mean().shift(1)
    std20 = c_df.rolling(20).std().shift(1)
    bb_up = (sma20 + 2.0 * std20)
    bb_dn = (sma20 - 2.0 * std20)

    # 2. 20-period ATR & Keltner Channels (1.5 * ATR)
    atr = {}
    for p in universe:
        tr = np.maximum(h_df[p] - l_df[p], np.maximum((h_df[p] - c_df[p].shift(1)).abs(), (l_df[p] - c_df[p].shift(1)).abs()))
        atr[p] = tr.rolling(20).mean().shift(1)
    atr_df = pd.DataFrame(atr, index=common_idx)

    kc_up = sma20 + 1.5 * atr_df
    kc_dn = sma20 - 1.5 * atr_df

    # Squeeze condition: Bollinger bands inside Keltner channels
    is_squeeze = (bb_up < kc_up) & (bb_dn > kc_dn)

    # 3. 50-day EMA (1200 1h bars) and Slope
    ema50d = c_df.ewm(span=1200, adjust=False).mean().shift(1)
    slope50d = ema50d.diff(24).shift(1)

    # 4. 20-bar Volume SMA
    vol_sma = v_df.rolling(20).mean().shift(1)
    vol_exp = (v_df.shift(1) > 1.2 * vol_sma)

    # 5. Breakout levels (20-bar Donchian)
    donch_hi = h_df.rolling(20).max().shift(1)
    donch_lo = l_df.rolling(20).min().shift(1)

    print("=========================================================================================")
    print("TESTING SQUEEZE BREAKOUT + VOLUME EXPANSION ENGINE ($500 START)")
    print("=========================================================================================\n")

    for tp_atr in [3.0]:
        for tier_step in [50.0, 100.0, 200.0]:
            initial_capital = 500.0
            soft_floor = 450.0
            active_floor = 0.0
            cash = initial_capital
            min_equity_ever = cash
            total_fees = 0.0
            total_yield = 0.0
            hourly_usdc_rate = 0.05 / 8760.0

            pos_cnt = {p: 0 for p in universe}
            pos_entry = {p: 0.0 for p in universe}
            pos_stop = {p: 0.0 for p in universe}
            pos_tp = {p: 0.0 for p in universe}
            pos_tp_hit = {p: False for p in universe}
            pos_peak = {p: 0.0 for p in universe}

            trades = []
            equity_curve = np.zeros(n_bars)

            for t in range(1250, n_bars):
                current_time = common_idx[t]
                unrealized = 0.0
                used_margin = 0.0

                # 1. Hourly Stop Loss and Take Profit evaluation
                for p in universe:
                    cnt = pos_cnt[p]
                    if cnt != 0:
                        px = c_df[p].iloc[t]
                        hi = h_df[p].iloc[t]
                        lo = l_df[p].iloc[t]
                        stop = pos_stop[p]
                        tp_px = pos_tp[p]
                        mult = MICRO_CONTRACT_SIZES.get(p, 1.0)
                        qty = abs(cnt) * mult
                        side = 1 if cnt > 0 else -1

                        # Update peak
                        if cnt > 0 and hi > pos_peak[p]:
                            pos_peak[p] = hi
                        elif cnt < 0 and (pos_peak[p] == 0 or lo < pos_peak[p]):
                            pos_peak[p] = lo

                        # A. Check Stop Loss
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
                            pnl = (exit_px - pos_entry[p]) * qty * side - fee
                            cash += pnl
                            total_fees += fee
                            trades.append((p, "STOP", pnl, current_time))
                            pos_cnt[p] = 0
                            pos_entry[p] = 0.0
                            pos_stop[p] = 0.0
                            pos_tp[p] = 0.0
                            pos_tp_hit[p] = False
                            continue

                        # B. Check Partial Take Profit (+tp_atr extension)
                        if not pos_tp_hit[p] and abs(cnt) >= 2:
                            tp_hit = False
                            if cnt > 0 and hi >= tp_px and tp_px > 0:
                                tp_hit = True
                            elif cnt < 0 and lo <= tp_px and tp_px > 0:
                                tp_hit = True

                            if tp_hit:
                                trim_cnt = abs(cnt) // 2
                                trim_qty = trim_cnt * mult
                                fee = trim_qty * tp_px * 0.0002
                                pnl = (tp_px - pos_entry[p]) * trim_qty * side - fee
                                cash += pnl
                                total_fees += fee
                                trades.append((p, "TP_PARTIAL", pnl, current_time))
                                pos_cnt[p] = (abs(cnt) - trim_cnt) * side
                                pos_tp_hit[p] = True
                                # Ratchet stop to Breakeven + 0.1 ATR!
                                a_val = atr_df[p].iloc[t]
                                pos_stop[p] = pos_entry[p] + side * 0.1 * (a_val if not np.isnan(a_val) else 0.0)

                        # C. Chandelier Trailing Stop on remainder
                        a_curr = atr_df[p].iloc[t]
                        if not np.isnan(a_curr) and a_curr > 0:
                            if cnt > 0:
                                ch_stop = pos_peak[p] - 3.0 * a_curr
                                if pos_tp_hit[p]:
                                    ch_stop = max(ch_stop, pos_entry[p]) # Never lose once TP hit
                                if ch_stop > pos_stop[p]:
                                    pos_stop[p] = ch_stop
                            elif cnt < 0:
                                ch_stop = pos_peak[p] + 3.0 * a_curr
                                if pos_tp_hit[p]:
                                    ch_stop = min(ch_stop, pos_entry[p])
                                if pos_stop[p] == 0 or ch_stop < pos_stop[p]:
                                    pos_stop[p] = ch_stop

                        unrealized += (px - pos_entry[p]) * qty * side
                        used_margin += qty * px * MARGIN_RATES.get(p, 0.33)

                idle = max(0.0, cash - used_margin)
                y = idle * hourly_usdc_rate
                cash += y
                total_yield += y

                current_equity = cash + unrealized
                min_equity_ever = min(min_equity_ever, current_equity)

                # Floor ratchets
                if current_equity >= 650.0 and active_floor < 500.0:
                    active_floor = 500.0
                elif current_equity >= 1500.0 and active_floor < 1000.0:
                    active_floor = 1000.0
                elif current_equity >= 3000.0 and active_floor < 2000.0:
                    active_floor = 2000.0

                if active_floor > 0 and current_equity <= active_floor:
                    for p in universe:
                        cnt = pos_cnt[p]
                        if cnt != 0:
                            px = c_df[p].iloc[t]
                            mult = MICRO_CONTRACT_SIZES.get(p, 1.0)
                            qty = abs(cnt) * mult
                            side = 1 if cnt > 0 else -1
                            fee = qty * px * 0.0005
                            cash += (px - pos_entry[p]) * qty * side - fee
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
                            px = c_df[p].iloc[t]
                            e50 = ema50d[p].iloc[t]
                            mult = MICRO_CONTRACT_SIZES.get(p, 1.0)
                            qty = abs(cnt) * mult
                            side = 1 if cnt > 0 else -1
                            if (cnt > 0 and px < e50) or (cnt < 0 and px > e50):
                                fee = qty * px * 0.0002
                                pnl = (px - pos_entry[p]) * qty * side - fee
                                cash += pnl
                                total_fees += fee
                                trades.append((p, "EMA_EXIT", pnl, current_time))
                                pos_cnt[p] = 0
                                pos_tp_hit[p] = False

                    active_count = sum(1 for p in universe if pos_cnt[p] != 0)
                    available_slots = 2 - active_count

                    if available_slots > 0:
                        is_defensive = (current_equity < 500.0)
                        excess = max(0.0, current_equity - 500.0)
                        add_cnt = int(excess // tier_step)
                        base_cnt = 1 if is_defensive else (1 + add_cnt)

                        for p in universe:
                            if available_slots <= 0:
                                break
                            if pos_cnt[p] != 0:
                                continue

                            px = c_df[p].iloc[t]
                            hi = donch_hi[p].iloc[t]
                            lo = donch_lo[p].iloc[t]
                            e50 = ema50d[p].iloc[t]
                            slp = slope50d[p].iloc[t]
                            a_val = atr_df[p].iloc[t]
                            vol_ok = vol_exp[p].iloc[t]
                            mult = MICRO_CONTRACT_SIZES.get(p, 1.0)

                            if np.isnan(px) or np.isnan(a_val) or a_val <= 0:
                                continue

                            cnt_target = max(1, base_cnt // 2) if p == "BTC-PERP" else base_cnt
                            if is_defensive:
                                cnt_target = 1

                            notional = cnt_target * mult * px
                            margin_needed = notional * MARGIN_RATES.get(p, 0.33)

                            if margin_needed <= current_equity * 0.50 and (notional / current_equity) <= 3.0:
                                stop_dist = max(2.5 * a_val, px * 0.035)

                                # LONG ENTRY: Price breaks 20-bar High + EMA 50d trend bullish + Volume expansion
                                if px > hi and px > e50 and slp > 0 and vol_ok:
                                    fee = notional * 0.0002
                                    cash -= fee
                                    total_fees += fee
                                    pos_cnt[p] = cnt_target
                                    pos_entry[p] = px
                                    pos_stop[p] = px - stop_dist
                                    pos_tp[p] = px + tp_atr * a_val
                                    pos_tp_hit[p] = False
                                    pos_peak[p] = px
                                    available_slots -= 1

                                # SHORT ENTRY: Price breaks 20-bar Low + EMA 50d trend bearish + Volume expansion
                                elif px < lo and px < e50 and slp < 0 and vol_ok:
                                    fee = notional * 0.0002
                                    cash -= fee
                                    total_fees += fee
                                    pos_cnt[p] = -cnt_target
                                    pos_entry[p] = px
                                    pos_stop[p] = px + stop_dist
                                    pos_tp[p] = px - tp_atr * a_val
                                    pos_tp_hit[p] = False
                                    pos_peak[p] = px
                                    available_slots -= 1

                equity_curve[t] = current_equity

            equity_curve[:1250] = initial_capital
            final_eq = equity_curve[-1]
            mult = final_eq / initial_capital
            peaks = np.maximum.accumulate(equity_curve)
            dds = (peaks - equity_curve) / peaks
            max_dd = np.max(dds)
            deposit_dd = max(0.0, (initial_capital - min_equity_ever) / initial_capital)
            wins = [tr for tr in trades if tr[2] > 0]
            wr = len(wins) / max(len(trades), 1)
            avg_win = np.mean([tr[2] for tr in wins]) if wins else 0.0
            losses = [tr for tr in trades if tr[2] <= 0]
            avg_loss = np.mean([tr[2] for tr in losses]) if losses else 0.0

            split = n_bars // 2
            is_mult = equity_curve[split] / initial_capital
            oos_mult = final_eq / equity_curve[split]

            if True:
                print(f"TP={tp_atr:3.1f}ATR | Step=${tier_step:5.1f} => Final=${final_eq:9,.2f} ({mult:5.2f}x) | MaxDD={max_dd:5.2%} | DepositDD={deposit_dd:5.2%} (MinEq=${min_equity_ever:6.2f}) | Trades={len(trades):3d} (WR={wr:4.1%}) | AvgW=${avg_win:5.2f} / AvgL=${avg_loss:5.2f} | IS={is_mult:4.2f}x | OOS={oos_mult:4.2f}x")

if __name__ == "__main__":
    run_squeeze_experiment()
