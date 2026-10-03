"""scratch/test_10x_nys_engine.py — 10x Micro Futures Engine for US / NYS Tradable Universe.

Specifications:
1. Starting Capital: $500.00.
2. Target: 10x ($5,000.00).
3. Tradable Universe: Assets with CFTC-regulated US / NYS futures contracts:
   - BTC (Nano BTI / Micro MBT = 0.01 BTC)
   - ETH (Micro MET / Nano ETI = 0.10 ETH)
   - SOL (Micro SOL = 1.0 SOL)
4. Non-Fractional Futures: Strictly whole integer contracts (1, 2, 3... contracts).
5. Allow Shorting: Both Long and Short trades with 50d EMA trend alignment.
6. Hard Capital Floor ($500.00):
   - Initial seed risk: strictly $15 per trade (3% of $500).
   - Once equity reaches $530+, the $500 floor activates.
   - Any trade stop is clamped so that the account balance NEVER dips below $500.00.
   - Only excess house money (Equity - $500) is risked for subsequent trades.
7. Realistic US Leverage: Maximum leverage 3.0x (CME / CFTC compliant).
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data

def run_nys_10x_sim():
    cfg = PortfolioConfig()
    bundle = load_portfolio_data(cfg, data_dir="data")
    
    # Restrict to US/NYS tradable crypto futures: BTC, ETH, SOL
    nys_universe = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
    common_idx = bundle.market_data_1h["BTC-PERP"].index

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in nys_universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in nys_universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in nys_universe}, index=common_idx).ffill().bfill()

    # 4h Resample
    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    contract_sizes = {
        "BTC-PERP": 0.01,  # 1 nano contract = 0.01 BTC ($600 - $1,000 notional)
        "ETH-PERP": 0.10,  # 1 micro contract = 0.10 ETH ($250 - $400 notional)
        "SOL-PERP": 1.00,  # 1 micro contract = 1.0 SOL ($100 - $250 notional)
    }

    # Trend Indicators
    # 20-day High/Low Donchian Breakout (120 4h bars)
    up_4h = h_4h.rolling(120, min_periods=60).max().shift(1)
    dn_4h = l_4h.rolling(120, min_periods=60).min().shift(1)

    # 50-day EMA and Slope (300 4h bars)
    ema50_4h = c_4h.ewm(span=300).mean().shift(1)
    slope_4h = ema50_4h.diff(6).shift(1)

    # 4-day Trailing Exits (24 4h bars)
    ex_l_4h = l_4h.rolling(24, min_periods=12).min().shift(1)
    ex_h_4h = h_4h.rolling(24, min_periods=12).max().shift(1)

    # ATR 14
    atr_4h = {}
    for p in nys_universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    # Kaufman Efficiency Ratio (ER)
    net_c = (c_4h - c_4h.shift(30)).abs()
    gross_c = (c_4h - c_4h.shift(1)).abs().rolling(30).sum()
    er_4h = (net_c / np.maximum(gross_c, 1e-8)).shift(1).fillna(0.0)

    # Forward to 1h
    up_1h = up_4h.reindex(common_idx).ffill()
    dn_1h = dn_4h.reindex(common_idx).ffill()
    ex_l_1h = ex_l_4h.reindex(common_idx).ffill()
    ex_h_1h = ex_h_4h.reindex(common_idx).ffill()
    ema50_1h = ema50_4h.reindex(common_idx).ffill()
    slope_1h = slope_4h.reindex(common_idx).ffill()
    atr_1h = atr_df.reindex(common_idx).ffill()
    er_1h = er_4h.reindex(common_idx).ffill()

    # Simulation across different house-money reinvestment rates
    for reinvest_rate in [0.10, 0.15, 0.20, 0.25, 0.30]:
        initial_capital = 500.0
        floor_capital = 500.0
        cash = initial_capital
        equity_curve = []
        trades = []
        min_equity_ever = cash
        floor_activated = False

        pos_cnt = {p: 0 for p in nys_universe}
        pos_entry = {p: 0.0 for p in nys_universe}
        pos_stop = {p: 0.0 for p in nys_universe}

        n_bars = len(common_idx)

        for t in range(1200, n_bars):
            current_time = common_idx[t]
            unrealized = 0.0

            # 1. Hourly Stop Loss Evaluation
            for p in nys_universe:
                cnt = pos_cnt[p]
                if cnt != 0:
                    px = c_df[p].iloc[t]
                    hi = h_df[p].iloc[t]
                    lo = l_df[p].iloc[t]
                    stop = pos_stop[p]
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

                    if hit_stop:
                        side = 1 if cnt > 0 else -1
                        fee = qty * exit_px * 0.0005
                        pnl = (exit_px - pos_entry[p]) * qty * side - fee
                        cash += (exit_px - pos_entry[p]) * qty * side - fee
                        trades.append((p, "STOP", pnl, current_time))
                        pos_cnt[p] = 0
                        pos_entry[p] = 0.0
                        pos_stop[p] = 0.0
                        continue

                    # Trailing stop ratchet
                    if cnt > 0:
                        ex_l = ex_l_1h[p].iloc[t]
                        if not np.isnan(ex_l) and ex_l > pos_stop[p]:
                            pos_stop[p] = ex_l
                    elif cnt < 0:
                        ex_h = ex_h_1h[p].iloc[t]
                        if not np.isnan(ex_h) and (pos_stop[p] == 0 or ex_h < pos_stop[p]):
                            pos_stop[p] = ex_h

                    # If floor activated, enforce that trailing stop NEVER drops cash + unrealized below $500!
                    if floor_activated:
                        # Required stop level to protect $500
                        # cash + (stop_req - entry) * qty * side >= 500
                        # (stop_req - entry) * qty * side >= 500 - cash
                        if cnt > 0:
                            hard_stop_px = pos_entry[p] + (floor_capital - cash) / (qty + 1e-9)
                            pos_stop[p] = max(pos_stop[p], hard_stop_px)
                        elif cnt < 0:
                            hard_stop_px = pos_entry[p] - (floor_capital - cash) / (qty + 1e-9)
                            pos_stop[p] = min(pos_stop[p], hard_stop_px)

                    side = 1 if cnt > 0 else -1
                    unrealized += (px - pos_entry[p]) * qty * side

            current_equity = cash + unrealized
            min_equity_ever = min(min_equity_ever, current_equity)
            equity_curve.append(current_equity)

            # Activate floor once equity builds above $530
            if current_equity >= 530.0:
                floor_activated = True

            # If floor is activated and current_equity touches $500, emergency close all to preserve initial capital!
            if floor_activated and current_equity <= floor_capital:
                for p in nys_universe:
                    cnt = pos_cnt[p]
                    if cnt != 0:
                        px = c_df[p].iloc[t]
                        mult = contract_sizes[p]
                        qty = abs(cnt) * mult
                        side = 1 if cnt > 0 else -1
                        fee = qty * px * 0.0005
                        pnl = (px - pos_entry[p]) * qty * side - fee
                        cash += pnl
                        pos_cnt[p] = 0
                current_equity = cash
                unrealized = 0.0

            # 2. Entries on 4h boundary
            if current_time.hour % 4 == 0 and current_equity >= 450.0:
                # Signal exits (cross of 50d EMA)
                for p in nys_universe:
                    cnt = pos_cnt[p]
                    if cnt != 0:
                        px = c_df[p].iloc[t]
                        e50 = ema50_1h[p].iloc[t]
                        mult = contract_sizes[p]
                        qty = abs(cnt) * mult
                        side = 1 if cnt > 0 else -1

                        if cnt > 0 and px < e50:
                            fee = qty * px * 0.0002
                            cash += (px - pos_entry[p]) * qty - fee
                            trades.append((p, "LONG_EMA_EXIT", (px - pos_entry[p]) * qty - fee, current_time))
                            pos_cnt[p] = 0

                        elif cnt < 0 and px > e50:
                            fee = qty * px * 0.0002
                            cash += (pos_entry[p] - px) * qty - fee
                            trades.append((p, "SHORT_EMA_EXIT", (pos_entry[p] - px) * qty - fee, current_time))
                            pos_cnt[p] = 0

                active_count = sum(1 for p in nys_universe if pos_cnt[p] != 0)
                available_slots = 2 - active_count

                if available_slots > 0:
                    # Risk budget:
                    if not floor_activated:
                        # Starting seed risk: $15 per trade
                        risk_budget_per_trade = 15.0
                    else:
                        # HOUSE MONEY RISK BUDGET:
                        excess = max(0.0, current_equity - floor_capital)
                        # Risk a portion of excess capital
                        risk_budget_per_trade = max(15.0, (excess * reinvest_rate) / available_slots)
                        # Strictly cap risk so trade stop can never breach the $500 floor
                        risk_budget_per_trade = min(risk_budget_per_trade, excess * 0.45)

                    for p in nys_universe:
                        if available_slots <= 0:
                            break
                        if pos_cnt[p] != 0:
                            continue

                        px = c_df[p].iloc[t]
                        up = up_1h[p].iloc[t]
                        dn = dn_1h[p].iloc[t]
                        e50 = ema50_1h[p].iloc[t]
                        slp = slope_1h[p].iloc[t]
                        er_val = er_1h[p].iloc[t]
                        atr = atr_1h[p].iloc[t]
                        mult = contract_sizes[p]

                        if np.isnan(px) or np.isnan(atr) or atr <= 0:
                            continue

                        stop_distance = max(2.5 * atr, px * 0.03)
                        dollar_risk_per_contract = stop_distance * mult

                        # INTEGER CONTRACT SIZING:
                        n_cnt = int(math.floor(risk_budget_per_trade / dollar_risk_per_contract))

                        # Cap at 3x gross leverage (CME / CFTC limit)
                        notional = n_cnt * mult * px
                        if notional > current_equity * 3.0:
                            n_cnt = int(math.floor((current_equity * 3.0) / (mult * px)))

                        # LONG ENTRY
                        if px > up and px > e50 and slp > 0 and er_val >= 0.28:
                            if n_cnt >= 1:
                                fee = n_cnt * mult * px * 0.0002
                                cash -= fee
                                pos_cnt[p] = n_cnt
                                pos_entry[p] = px
                                pos_stop[p] = px - stop_distance
                                available_slots -= 1

                        # SHORT ENTRY (SHORTING ENABLED!)
                        elif px < dn and px < e50 and slp < 0 and er_val >= 0.28:
                            if n_cnt >= 1:
                                fee = n_cnt * mult * px * 0.0002
                                cash -= fee
                                pos_cnt[p] = -n_cnt
                                pos_entry[p] = px
                                pos_stop[p] = px + stop_distance
                                available_slots -= 1

        eq_arr = np.array(equity_curve)
        final_eq = eq_arr[-1]
        mult = final_eq / initial_capital
        peaks = np.maximum.accumulate(eq_arr)
        dds = (peaks - eq_arr) / peaks
        max_dd = np.max(dds)
        deposit_dd = max(0.0, (initial_capital - min_equity_ever) / initial_capital)
        wins = [tr for tr in trades if tr[2] > 0]
        wr = len(wins) / max(len(trades), 1)

        print(f"Reinvest={reinvest_rate:.2f} => Final=${final_eq:9,.2f} ({mult:5.2f}x) | Peak MaxDD={max_dd:5.2%} | Deposit DD={deposit_dd:5.2%} (MinEq=${min_equity_ever:.2f}) | Trades={len(trades)} | WR={wr:4.1%}")

if __name__ == "__main__":
    run_nys_10x_sim()
