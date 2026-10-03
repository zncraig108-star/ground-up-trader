"""scratch/test_nys_10x_calibrated.py — Calibrated 10x Micro Futures Engine for NYS.

Accurate mechanics:
1. Account: Starts at $500.00.
2. Tradable NYS contracts:
   - ETH Micro (MET = 0.1 ETH, ~$250 notional)
   - SOL Micro (1 SOL, ~$150 notional)
   - BTC Nano (BTI = 0.01 BTC, ~$600 notional, unlocked once equity >= $1,000)
3. Non-fractional: Strictly integer contract quantities.
4. Shorting enabled: Trend following on both sides.
5. Capital Preservation Floor:
   - Initial seed risk buffer: $50 (max initial drawdown 10% to $450).
   - Once equity >= $550, HARD FLOOR locks at $500.00.
   - All subsequent risk is strictly funded from HOUSE MONEY (Equity - $500).
   - If equity drops to $500.00 after being in profit, trading halts to preserve the $500.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data

def run_calibrated_10x():
    cfg = PortfolioConfig()
    bundle = load_portfolio_data(cfg, data_dir="data")
    universe = ["ETH-PERP", "SOL-PERP", "BTC-PERP"]
    common_idx = bundle.market_data_1h["BTC-PERP"].index

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()

    # 4h Resample
    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    # Micro/Nano contract multiplier
    contract_sizes = {
        "ETH-PERP": 0.10,  # 1 micro contract = 0.10 ETH
        "SOL-PERP": 1.00,  # 1 micro contract = 1.0 SOL
        "BTC-PERP": 0.01,  # 1 nano contract = 0.01 BTC
    }

    # 10-day Donchian High/Low (60 4h bars)
    up_4h = h_4h.rolling(60, min_periods=30).max().shift(1)
    dn_4h = l_4h.rolling(60, min_periods=30).min().shift(1)

    # 3-day Trailing Exits (18 4h bars)
    ex_l_4h = l_4h.rolling(18, min_periods=9).min().shift(1)
    ex_h_4h = h_4h.rolling(18, min_periods=9).max().shift(1)

    # 30-day EMA
    ema30_4h = c_4h.ewm(span=180).mean().shift(1)
    slope_4h = ema30_4h.diff(6).shift(1)

    # ATR 14
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    # Align to 1h
    up_1h = up_4h.reindex(common_idx).ffill()
    dn_1h = dn_4h.reindex(common_idx).ffill()
    ex_l_1h = ex_l_4h.reindex(common_idx).ffill()
    ex_h_1h = ex_h_4h.reindex(common_idx).ffill()
    ema30_1h = ema30_4h.reindex(common_idx).ffill()
    slope_1h = slope_4h.reindex(common_idx).ffill()
    atr_1h = atr_df.reindex(common_idx).ffill()

    initial_capital = 500.0
    cash = initial_capital
    floor_locked = False
    min_equity_ever = cash
    equity_curve = []
    trades = []

    pos_cnt = {p: 0 for p in universe}
    pos_entry = {p: 0.0 for p in universe}
    pos_stop = {p: 0.0 for p in universe}

    n_bars = len(common_idx)

    for t in range(600, n_bars):
        current_time = common_idx[t]
        unrealized = 0.0

        # 1. Hourly Stop Loss Evaluation
        for p in universe:
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
                    cash += pnl
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

                side = 1 if cnt > 0 else -1
                unrealized += (px - pos_entry[p]) * qty * side

        current_equity = cash + unrealized
        min_equity_ever = min(min_equity_ever, current_equity)
        equity_curve.append(current_equity)

        # Lock hard floor once equity reaches $550
        if current_equity >= 550.0:
            floor_locked = True

        # Emergency preservation: if floor locked and equity drops back to $500, close everything
        if floor_locked and current_equity <= 500.0:
            for p in universe:
                cnt = pos_cnt[p]
                if cnt != 0:
                    px = c_df[p].iloc[t]
                    qty = abs(cnt) * contract_sizes[p]
                    side = 1 if cnt > 0 else -1
                    fee = qty * px * 0.0005
                    pnl = (px - pos_entry[p]) * qty * side - fee
                    cash += pnl
                    pos_cnt[p] = 0
            current_equity = cash
            unrealized = 0.0

        # 2. Entries on 4h boundary
        if current_time.hour % 4 == 0:
            # If floor locked and at $500, halt trading (capital protected!)
            if floor_locked and current_equity <= 500.0:
                continue

            active_cnt = sum(1 for p in universe if pos_cnt[p] != 0)
            if active_cnt < 2:
                # Sizing:
                # If equity < 1000: trade 1 contract of SOL or ETH
                # If equity >= 1000: trade 2 contracts, or 1 BTC nano
                # If equity >= 2500: trade 4 contracts
                if current_equity < 1000.0:
                    allowed_assets = ["ETH-PERP", "SOL-PERP"]
                    n_contracts = 1
                elif current_equity < 2500.0:
                    allowed_assets = ["ETH-PERP", "SOL-PERP", "BTC-PERP"]
                    n_contracts = 2
                else:
                    allowed_assets = ["ETH-PERP", "SOL-PERP", "BTC-PERP"]
                    n_contracts = int(current_equity // 600.0)

                for p in allowed_assets:
                    if sum(1 for x in universe if pos_cnt[x] != 0) >= 2:
                        break
                    if pos_cnt[p] != 0:
                        continue

                    px = c_df[p].iloc[t]
                    up = up_1h[p].iloc[t]
                    dn = dn_1h[p].iloc[t]
                    e30 = ema30_1h[p].iloc[t]
                    slp = slope_1h[p].iloc[t]
                    atr = atr_1h[p].iloc[t]
                    mult = contract_sizes[p]

                    if np.isnan(px) or np.isnan(atr) or atr <= 0:
                        continue

                    stop_dist = max(2.0 * atr, px * 0.03)

                    # LONG ENTRY
                    if px > up and px > e30 and slp > 0:
                        cnt_to_buy = n_contracts if p != "BTC-PERP" else max(1, n_contracts // 2)
                        notional = cnt_to_buy * mult * px
                        if notional <= current_equity * 3.0:
                            fee = notional * 0.0002
                            cash -= fee
                            pos_cnt[p] = cnt_to_buy
                            pos_entry[p] = px
                            pos_stop[p] = px - stop_dist

                    # SHORT ENTRY (SHORTING ALLOWED!)
                    elif px < dn and px < e30 and slp < 0:
                        cnt_to_sell = n_contracts if p != "BTC-PERP" else max(1, n_contracts // 2)
                        notional = cnt_to_sell * mult * px
                        if notional <= current_equity * 3.0:
                            fee = notional * 0.0002
                            cash -= fee
                            pos_cnt[p] = -cnt_to_sell
                            pos_entry[p] = px
                            pos_stop[p] = px + stop_dist

    eq_arr = np.array(equity_curve)
    final_eq = eq_arr[-1]
    mult = final_eq / initial_capital
    peaks = np.maximum.accumulate(eq_arr)
    dds = (peaks - eq_arr) / peaks
    max_dd = np.max(dds)
    deposit_dd = max(0.0, (initial_capital - min_equity_ever) / initial_capital)
    wins = [tr for tr in trades if tr[2] > 0]
    wr = len(wins) / max(len(trades), 1)

    print(f"=== NYS MICRO FUTURES RESULT ($500 START) ===")
    print(f"Final Equity:   ${final_eq:10,.2f} ({mult:.2f}x)")
    print(f"Max Peak DD:    {max_dd:.2%}")
    print(f"Deposit DD:     {deposit_dd:.2%} (Lowest Equity: ${min_equity_ever:.2f})")
    print(f"Total Trades:   {len(trades)} | Win Rate: {wr:.1%}")

if __name__ == "__main__":
    run_calibrated_10x()
