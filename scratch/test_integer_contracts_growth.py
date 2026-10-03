"""scratch/test_integer_contracts_growth.py — Minimum Integer Contract Sizing for $500 Account.

Tests minimum whole contract trading (1 contract baseline, scaling with excess house money)
across BTC, ETH, and SOL futures with:
- Starting capital: $500.00
- Hard capital preservation floor: Once profitable, trailing stops lock so equity NEVER falls below $500.00
- Long and Short trading enabled
- Strictly whole integer contracts (1, 2, 3...)
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data

def test_integer_growth():
    cfg = PortfolioConfig()
    bundle = load_portfolio_data(cfg, data_dir="data")
    universe = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
    common_idx = bundle.market_data_1h["BTC-PERP"].index

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()

    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    contract_sizes = {
        "BTC-PERP": 0.01,  # 1 nano contract = 0.01 BTC
        "ETH-PERP": 0.10,  # 1 micro contract = 0.10 ETH
        "SOL-PERP": 1.00,  # 1 micro contract = 1.0 SOL
    }

    # Donchian 20d High/Low breakout
    up_4h = h_4h.rolling(120, min_periods=60).max().shift(1)
    dn_4h = l_4h.rolling(120, min_periods=60).min().shift(1)

    # 4-day Trailing Stop Channels
    ex_l_4h = l_4h.rolling(24, min_periods=12).min().shift(1)
    ex_h_4h = h_4h.rolling(24, min_periods=12).max().shift(1)

    # 50-day EMA & Slope
    ema50_4h = c_4h.ewm(span=300).mean().shift(1)
    slope_4h = ema50_4h.diff(6).shift(1)

    # 20-period SMA for Climax
    sma20_4h = c_4h.rolling(20, min_periods=10).mean().shift(1)

    # ATR 14
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    # Kaufman ER
    net_c = (c_4h - c_4h.shift(30)).abs()
    gross_c = (c_4h - c_4h.shift(1)).abs().rolling(30).sum()
    er_4h = (net_c / np.maximum(gross_c, 1e-8)).shift(1).fillna(0.0)

    # Momentum
    roc_30d = (c_4h / c_4h.shift(180) - 1.0).shift(1)

    # Align to 1h
    up_1h = up_4h.reindex(common_idx).ffill()
    dn_1h = dn_4h.reindex(common_idx).ffill()
    ex_l_1h = ex_l_4h.reindex(common_idx).ffill()
    ex_h_1h = ex_h_4h.reindex(common_idx).ffill()
    ema50_1h = ema50_4h.reindex(common_idx).ffill()
    slope_1h = slope_4h.reindex(common_idx).ffill()
    sma20_1h = sma20_4h.reindex(common_idx).ffill()
    atr_1h = atr_df.reindex(common_idx).ffill()
    er_1h = er_4h.reindex(common_idx).ffill()
    roc_1h = roc_30d.reindex(common_idx).ffill()

    for excess_scaling in [0.002, 0.005, 0.010, 0.015, 0.020]:
        initial_capital = 500.0
        cash = initial_capital
        equity_curve = []
        trades = []
        min_equity_ever = cash
        floor_activated = False

        pos_cnt = {p: 0 for p in universe}
        pos_entry = {p: 0.0 for p in universe}
        pos_stop = {p: 0.0 for p in universe}
        pos_trimmed = {p: False for p in universe}

        n_bars = len(common_idx)

        for t in range(1200, n_bars):
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
                        pos_trimmed[p] = False
                        continue

                    # Climax trim (take half off on vertical blowoffs)
                    if not pos_trimmed[p] and abs(cnt) >= 2:
                        s20 = sma20_1h[p].iloc[t]
                        atr = atr_1h[p].iloc[t]
                        if not np.isnan(s20) and not np.isnan(atr) and atr > 0:
                            if cnt > 0 and (px - s20) >= 2.5 * atr and px > pos_entry[p]:
                                trim_cnt = abs(cnt) // 2
                                trim_qty = trim_cnt * mult
                                fee = trim_qty * px * 0.0002
                                pnl = (px - pos_entry[p]) * trim_qty - fee
                                cash += pnl
                                trades.append((p, "CLIMAX_TRIM", pnl, current_time))
                                pos_cnt[p] = abs(cnt) - trim_cnt
                                pos_trimmed[p] = True
                                pos_stop[p] = pos_entry[p]
                            elif cnt < 0 and (s20 - px) >= 2.5 * atr and px < pos_entry[p]:
                                trim_cnt = abs(cnt) // 2
                                trim_qty = trim_cnt * mult
                                fee = trim_qty * px * 0.0002
                                pnl = (pos_entry[p] - px) * trim_qty - fee
                                cash += pnl
                                trades.append((p, "CLIMAX_TRIM", pnl, current_time))
                                pos_cnt[p] = -(abs(cnt) - trim_cnt)
                                pos_trimmed[p] = True
                                pos_stop[p] = pos_entry[p]

                    # Trailing Stop
                    if cnt > 0:
                        ex_l = ex_l_1h[p].iloc[t]
                        if not np.isnan(ex_l) and ex_l > pos_stop[p]:
                            pos_stop[p] = ex_l
                    elif cnt < 0:
                        ex_h = ex_h_1h[p].iloc[t]
                        if not np.isnan(ex_h) and (pos_stop[p] == 0 or ex_h < pos_stop[p]):
                            pos_stop[p] = ex_h

                    # Floor protection: if floor is active, clamp stop so exit never drops equity < $500
                    if floor_activated:
                        if cnt > 0:
                            hard_stop = pos_entry[p] + (500.0 - cash) / (qty + 1e-9)
                            pos_stop[p] = max(pos_stop[p], hard_stop)
                        elif cnt < 0:
                            hard_stop = pos_entry[p] - (500.0 - cash) / (qty + 1e-9)
                            pos_stop[p] = min(pos_stop[p], hard_stop)

                    side = 1 if cnt > 0 else -1
                    unrealized += (px - pos_entry[p]) * qty * side

            current_equity = cash + unrealized
            min_equity_ever = min(min_equity_ever, current_equity)
            equity_curve.append(current_equity)

            if current_equity >= 530.0:
                floor_activated = True

            # Emergency floor stop: if touched $500 after being profitable, close out
            if floor_activated and current_equity <= 500.0:
                for p in universe:
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
            if current_time.hour % 4 == 0 and current_equity >= 400.0:
                # 50d EMA exit
                for p in universe:
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
                            pos_trimmed[p] = False
                        elif cnt < 0 and px > e50:
                            fee = qty * px * 0.0002
                            cash += (pos_entry[p] - px) * qty - fee
                            trades.append((p, "SHORT_EMA_EXIT", (pos_entry[p] - px) * qty - fee, current_time))
                            pos_cnt[p] = 0
                            pos_trimmed[p] = False

                active_count = sum(1 for p in universe if pos_cnt[p] != 0)
                available_slots = 2 - active_count

                if available_slots > 0:
                    # Number of contracts allowed:
                    # Baseline = 1 contract
                    # Excess scaling: For every $500 of excess equity above $500, add contracts
                    excess = max(0.0, current_equity - 500.0)
                    extra_contracts = int(math.floor(excess * excess_scaling))

                    # Rank universe by 30d momentum
                    rocs = {p: roc_1h[p].iloc[t] for p in universe if not np.isnan(roc_1h[p].iloc[t])}
                    if len(rocs) >= 3:
                        ranked_desc = sorted(rocs.keys(), key=lambda x: rocs[x], reverse=True)

                        # Best leader for LONG
                        p_long = ranked_desc[0]
                        if pos_cnt[p_long] == 0 and available_slots > 0:
                            px = c_df[p_long].iloc[t]
                            up = up_1h[p_long].iloc[t]
                            e50 = ema50_1h[p_long].iloc[t]
                            slp = slope_1h[p_long].iloc[t]
                            er_val = er_1h[p_long].iloc[t]
                            atr = atr_1h[p_long].iloc[t]
                            mult = contract_sizes[p_long]

                            if px > up and px > e50 and slp > 0 and er_val >= 0.28 and rocs[p_long] > 0:
                                n_contracts = 1 + extra_contracts
                                notional = n_contracts * mult * px
                                if notional <= current_equity * 3.0:
                                    stop_dist = max(2.5 * atr, px * 0.04)
                                    fee = notional * 0.0002
                                    cash -= fee
                                    pos_cnt[p_long] = n_contracts
                                    pos_entry[p_long] = px
                                    pos_stop[p_long] = px - stop_dist
                                    pos_trimmed[p_long] = False
                                    available_slots -= 1

                        # Worst laggard for SHORT (Shorting allowed!)
                        p_short = ranked_desc[-1]
                        if pos_cnt[p_short] == 0 and available_slots > 0:
                            px = c_df[p_short].iloc[t]
                            dn = dn_1h[p_short].iloc[t]
                            e50 = ema50_1h[p_short].iloc[t]
                            slp = slope_1h[p_short].iloc[t]
                            er_val = er_1h[p_short].iloc[t]
                            atr = atr_1h[p_short].iloc[t]
                            mult = contract_sizes[p_short]

                            if px < dn and px < e50 and slp < 0 and er_val >= 0.28 and rocs[p_short] < 0:
                                n_contracts = 1 + extra_contracts
                                notional = n_contracts * mult * px
                                if notional <= current_equity * 3.0:
                                    stop_dist = max(2.5 * atr, px * 0.04)
                                    fee = notional * 0.0002
                                    cash -= fee
                                    pos_cnt[p_short] = -n_contracts
                                    pos_entry[p_short] = px
                                    pos_stop[p_short] = px + stop_dist
                                    pos_trimmed[p_short] = False
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

        print(f"ExcessScaling={excess_scaling:.3f} => Final=${final_eq:8,.2f} ({mult:5.2f}x) | Peak MaxDD={max_dd:5.2%} | Deposit DD={deposit_dd:5.2%} (MinEq=${min_equity_ever:.2f}) | Trades={len(trades)} | WR={wr:4.1%}")

if __name__ == "__main__":
    test_integer_growth()
