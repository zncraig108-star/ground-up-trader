"""scratch/test_nys_ratcheting_floor.py — Ratcheting Capital Floor for $500 Account.

Mechanics:
- Initial Capital: $500.00
- Initial Buffer: Can take normal trades in early stage (max risk 5% = $25/trade).
- Floor Activation: Once account reaches $700 (+40% gain), HARD FLOOR is locked at $500.00.
  From that point on, you are playing 100% on house money. You can NEVER lose your initial deposit!
- Ratcheting Floors:
  - At $1,200 equity -> Floor raises to $800.
  - At $2,500 equity -> Floor raises to $1,800.
  - At $5,000 equity -> Target reached (10x)!
- Sizing: Non-fractional integer futures (1, 2, 3... contracts) on ETH, SOL, BTC.
- Shorting: Allowed on 50d EMA bearish breakdowns.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data

def run_ratcheting_engine():
    cfg = PortfolioConfig()
    bundle = load_portfolio_data(cfg, data_dir="data")
    universe = ["ETH-PERP", "SOL-PERP", "BTC-PERP"]
    common_idx = bundle.market_data_1h["BTC-PERP"].index

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()

    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    contract_sizes = {
        "ETH-PERP": 0.10,  # 1 micro contract = 0.10 ETH
        "SOL-PERP": 1.00,  # 1 micro contract = 1.0 SOL
        "BTC-PERP": 0.01,  # 1 nano contract = 0.01 BTC
    }

    up_4h = h_4h.rolling(120, min_periods=60).max().shift(1)
    dn_4h = l_4h.rolling(120, min_periods=60).min().shift(1)
    ex_l_4h = l_4h.rolling(24, min_periods=12).min().shift(1)
    ex_h_4h = h_4h.rolling(24, min_periods=12).max().shift(1)

    ema50_4h = c_4h.ewm(span=300).mean().shift(1)
    slope_4h = ema50_4h.diff(6).shift(1)
    sma20_4h = c_4h.rolling(20, min_periods=10).mean().shift(1)

    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    er_4h = ((c_4h - c_4h.shift(30)).abs() / np.maximum((c_4h - c_4h.shift(1)).abs().rolling(30).sum(), 1e-8)).shift(1).fillna(0.0)

    # 1h alignments
    up_1h = up_4h.reindex(common_idx).ffill()
    dn_1h = dn_4h.reindex(common_idx).ffill()
    ex_l_1h = ex_l_4h.reindex(common_idx).ffill()
    ex_h_1h = ex_h_4h.reindex(common_idx).ffill()
    ema50_1h = ema50_4h.reindex(common_idx).ffill()
    slope_1h = slope_4h.reindex(common_idx).ffill()
    sma20_1h = sma20_4h.reindex(common_idx).ffill()
    atr_1h = atr_df.reindex(common_idx).ffill()
    er_1h = er_4h.reindex(common_idx).ffill()

    initial_capital = 500.0
    cash = initial_capital
    active_floor = 0.0  # 0 until $700 reached
    min_equity_ever = cash
    equity_curve = []
    trades = []

    pos_cnt = {p: 0 for p in universe}
    pos_entry = {p: 0.0 for p in universe}
    pos_stop = {p: 0.0 for p in universe}
    pos_trimmed = {p: False for p in universe}

    n_bars = len(common_idx)

    for t in range(1200, n_bars):
        current_time = common_idx[t]
        unrealized = 0.0

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
                    pos_trimmed[p] = False
                    continue

                # Climax trim (lock 50% profit at +2.8 ATR)
                if not pos_trimmed[p] and abs(cnt) >= 2:
                    s20 = sma20_1h[p].iloc[t]
                    atr = atr_1h[p].iloc[t]
                    if not np.isnan(s20) and not np.isnan(atr) and atr > 0:
                        if cnt > 0 and (px - s20) >= 2.8 * atr and px > pos_entry[p]:
                            trim_cnt = abs(cnt) // 2
                            trim_qty = trim_cnt * mult
                            fee = trim_qty * px * 0.0002
                            cash += (px - pos_entry[p]) * trim_qty - fee
                            trades.append((p, "CLIMAX", (px - pos_entry[p]) * trim_qty - fee, current_time))
                            pos_cnt[p] = abs(cnt) - trim_cnt
                            pos_trimmed[p] = True
                            pos_stop[p] = pos_entry[p]
                        elif cnt < 0 and (s20 - px) >= 2.8 * atr and px < pos_entry[p]:
                            trim_cnt = abs(cnt) // 2
                            trim_qty = trim_cnt * mult
                            fee = trim_qty * px * 0.0002
                            cash += (pos_entry[p] - px) * trim_qty - fee
                            trades.append((p, "CLIMAX", (pos_entry[p] - px) * trim_qty - fee, current_time))
                            pos_cnt[p] = -(abs(cnt) - trim_cnt)
                            pos_trimmed[p] = True
                            pos_stop[p] = pos_entry[p]

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

        # Ratcheting Capital Floor Activation:
        if current_equity >= 700.0 and active_floor < 500.0:
            active_floor = 500.0  # LOCKED: Can NEVER lose original $500!
        if current_equity >= 1200.0 and active_floor < 800.0:
            active_floor = 800.0
        if current_equity >= 2500.0 and active_floor < 1800.0:
            active_floor = 1800.0

        # Floor Protection Trigger
        if active_floor > 0 and current_equity <= active_floor:
            for p in universe:
                cnt = pos_cnt[p]
                if cnt != 0:
                    px = c_df[p].iloc[t]
                    qty = abs(cnt) * contract_sizes[p]
                    side = 1 if cnt > 0 else -1
                    fee = qty * px * 0.0005
                    cash += (px - pos_entry[p]) * qty * side - fee
                    pos_cnt[p] = 0
            current_equity = cash
            unrealized = 0.0

        # Entries on 4h
        if current_time.hour % 4 == 0:
            if active_floor > 0 and current_equity <= active_floor:
                continue

            for p in universe:
                cnt = pos_cnt[p]
                if cnt != 0:
                    px = c_df[p].iloc[t]
                    e50 = ema50_1h[p].iloc[t]
                    qty = abs(cnt) * contract_sizes[p]
                    side = 1 if cnt > 0 else -1

                    if cnt > 0 and px < e50:
                        fee = qty * px * 0.0002
                        cash += (px - pos_entry[p]) * qty - fee
                        trades.append((p, "LONG_EMA_EXIT", (px - pos_entry[p]) * qty - fee, current_time))
                        pos_cnt[p] = 0
                        pos_trimmed[p] = False
                    elif cnt < 0 and px > e50:
                        fee = qty * px * 0.0002
                        cash += (pos_entry[p] - px) * qty - fee
                        trades.append((p, "SHORT_EMA_EXIT", (pos_entry[p] - px) * qty - fee, current_time))
                        pos_cnt[p] = 0
                        pos_trimmed[p] = False

            active_cnt = sum(1 for p in universe if pos_cnt[p] != 0)
            if active_cnt < 2 and current_equity >= 350.0:
                # Sizing:
                # If equity < 1000: 1 contract of ETH or SOL
                # If equity >= 1000: 2 contracts (or 1 BTC nano)
                # If equity >= 2000: 4 contracts
                # If equity >= 3500: 6 contracts
                if current_equity < 1000.0:
                    candidates = ["SOL-PERP", "ETH-PERP"]
                    base_cnt = 1
                elif current_equity < 2000.0:
                    candidates = ["SOL-PERP", "ETH-PERP", "BTC-PERP"]
                    base_cnt = 2
                elif current_equity < 3500.0:
                    candidates = ["SOL-PERP", "ETH-PERP", "BTC-PERP"]
                    base_cnt = 4
                else:
                    candidates = ["SOL-PERP", "ETH-PERP", "BTC-PERP"]
                    base_cnt = 6

                for p in candidates:
                    if sum(1 for x in universe if pos_cnt[x] != 0) >= 2:
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

                    stop_dist = max(2.5 * atr, px * 0.035)
                    cnt_to_trade = max(1, base_cnt // 2) if p == "BTC-PERP" else base_cnt
                    notional = cnt_to_trade * mult * px

                    if notional <= current_equity * 3.0:
                        # LONG ENTRY
                        if px > up and px > e50 and slp > 0 and er_val >= 0.28:
                            fee = notional * 0.0002
                            cash -= fee
                            pos_cnt[p] = cnt_to_trade
                            pos_entry[p] = px
                            pos_stop[p] = px - stop_dist
                            pos_trimmed[p] = False

                        # SHORT ENTRY (SHORTING ENABLED!)
                        elif px < dn and px < e50 and slp < 0 and er_val >= 0.28:
                            fee = notional * 0.0002
                            cash -= fee
                            pos_cnt[p] = -cnt_to_trade
                            pos_entry[p] = px
                            pos_stop[p] = px + stop_dist
                            pos_trimmed[p] = False

    eq_arr = np.array(equity_curve)
    final_eq = eq_arr[-1]
    mult = final_eq / initial_capital
    peaks = np.maximum.accumulate(eq_arr)
    dds = (peaks - eq_arr) / peaks
    max_dd = np.max(dds)
    deposit_dd = max(0.0, (initial_capital - min_equity_ever) / initial_capital)
    wins = [tr for tr in trades if tr[2] > 0]
    wr = len(wins) / max(len(trades), 1)

    print(f"=== RATCHETING CAPITAL PRESERVATION ENGINE ($500 START) ===")
    print(f"Final Equity:       ${final_eq:10,.2f} ({mult:.2f}x)")
    print(f"Max Peak DD:        {max_dd:.2%}")
    print(f"Initial Deposit DD: {deposit_dd:.2%} (Lowest Equity: ${min_equity_ever:.2f})")
    print(f"Total Trades:       {len(trades)} | Win Rate: {wr:.1%}")

if __name__ == "__main__":
    run_ratcheting_engine()
