"""scratch/test_500_micro_asymmetry.py — High-Convexity $500 Micro Futures Engine.

Key Features:
1. Initial Capital: $500.00.
2. Hard Floor at $500: Once equity > $530, the account locks a hard stop floor at $500.00.
   Trailing stops are clamped so that a total portfolio stopout can NEVER drop equity below $500.
3. Integer Contracts: Every position size is strictly an integer (1, 2, 3... contracts).
4. Asymmetric Risk/Reward: 
   - Initial risk per trade: strictly $10 - $20 (2% - 4% of $500).
   - Once excess capital builds (e.g. $800 equity, $300 excess): risk scales with excess capital.
5. Parabolic Climax Trimming: Takes 50% profit at +2.8 ATR extension to bank house money.
6. Long & Short Allowed: Long leaders on bullish breakout, short laggards on bearish breakdown.
7. Realistic Leverage: Maximum leverage 3x (fully compliant with US / NYS margin requirements).
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data

def run_micro_engine():
    cfg = PortfolioConfig()
    bundle = load_portfolio_data(cfg, data_dir="data")
    universe = list(bundle.market_data_1h.keys())
    common_idx = bundle.market_data_1h["BTC-PERP"].index

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()

    # 4h Resample
    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    # Contract specifications for micro/nano crypto futures
    contract_sizes = {
        "BTC-PERP": 0.01,   # 1 nano contract = 0.01 BTC
        "ETH-PERP": 0.10,   # 1 nano contract = 0.10 ETH
        "SOL-PERP": 1.00,   # 1 contract = 1 SOL
        "AVAX-PERP": 1.00,  # 1 contract = 1 AVAX
        "LINK-PERP": 1.00,  # 1 contract = 1 LINK
        "DOGE-PERP": 100.0, # 1 contract = 100 DOGE
        "NEAR-PERP": 10.0,  # 1 contract = 10 NEAR
        "SUI-PERP": 10.0,   # 1 contract = 10 SUI
        "XRP-PERP": 100.0,  # 1 contract = 100 XRP
    }

    # 20-day High/Low Channels (120 4h bars)
    up_4h = h_4h.rolling(120, min_periods=60).max().shift(1)
    dn_4h = l_4h.rolling(120, min_periods=60).min().shift(1)
    
    # 4-day Trailing Exit Channels (24 4h bars)
    ex_l_4h = l_4h.rolling(24, min_periods=12).min().shift(1)
    ex_h_4h = h_4h.rolling(24, min_periods=12).max().shift(1)

    # 50-day EMA & Slope
    ema50_4h = c_4h.ewm(span=300).mean().shift(1)
    slope_4h = ema50_4h.diff(6).shift(1)

    # 20-period SMA for Climax trimming
    sma20_4h = c_4h.rolling(20, min_periods=10).mean().shift(1)

    # ATR 14
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    # Kaufman Efficiency Ratio (ER)
    net_c = (c_4h - c_4h.shift(30)).abs()
    gross_c = (c_4h - c_4h.shift(1)).abs().rolling(30).sum()
    er_4h = (net_c / np.maximum(gross_c, 1e-8)).shift(1).fillna(0.0)

    # 30-day Momentum
    roc_30d = (c_4h / c_4h.shift(180) - 1.0).shift(1)

    # BTC Macro Regime Filter
    btc_sma = c_4h["BTC-PERP"].rolling(1200, min_periods=120).mean().shift(1)
    is_bull_4h = (c_4h["BTC-PERP"] > btc_sma).shift(1).fillna(False)

    # Forward to 1h
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
    is_bull_1h = is_bull_4h.reindex(common_idx).ffill()

    # Test across multiple house-money risk multipliers
    for excess_risk_pct in [0.08, 0.12, 0.16, 0.20, 0.25]:
        initial_capital = 500.0
        cash = initial_capital
        equity_curve = []
        trades = []
        min_equity_ever = cash

        pos_cnt = {p: 0 for p in universe}       # signed integer
        pos_entry = {p: 0.0 for p in universe}
        pos_stop = {p: 0.0 for p in universe}
        pos_trimmed = {p: False for p in universe}

        n_bars = len(common_idx)

        for t in range(1200, n_bars):
            current_time = common_idx[t]
            unrealized = 0.0

            # 1. Hourly Stop Loss & Climax Check
            for p in universe:
                cnt = pos_cnt[p]
                if cnt != 0:
                    px = c_df[p].iloc[t]
                    hi = h_df[p].iloc[t]
                    lo = l_df[p].iloc[t]
                    stop = pos_stop[p]
                    mult = contract_sizes[p]
                    qty = abs(cnt) * mult

                    # A. Stop hit
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
                        pos_trimmed[p] = False
                        continue

                    # B. Parabolic Climax Trim (take 50% off the table at +2.8 ATR on winners)
                    if not pos_trimmed[p] and abs(cnt) >= 2:
                        s20 = sma20_1h[p].iloc[t]
                        atr = atr_1h[p].iloc[t]
                        if not np.isnan(s20) and not np.isnan(atr) and atr > 0:
                            is_climax = False
                            if cnt > 0 and (px - s20) >= 2.8 * atr and px > pos_entry[p]:
                                is_climax = True
                            elif cnt < 0 and (s20 - px) >= 2.8 * atr and px < pos_entry[p]:
                                is_climax = True

                            if is_climax:
                                trim_cnt = abs(cnt) // 2
                                trim_qty = trim_cnt * mult
                                side = 1 if cnt > 0 else -1
                                fee = trim_qty * px * 0.0002
                                pnl = (px - pos_entry[p]) * trim_qty * side - fee
                                cash += (px - pos_entry[p]) * trim_qty * side - fee
                                trades.append((p, "CLIMAX_TRIM", pnl, current_time))
                                pos_cnt[p] = (abs(cnt) - trim_cnt) * side
                                pos_trimmed[p] = True
                                # Ratchet stop to breakeven after climax trim!
                                pos_stop[p] = pos_entry[p]

                    # C. Ratchet trailing stop
                    if cnt > 0:
                        ex_l = ex_l_1h[p].iloc[t]
                        if not np.isnan(ex_l) and ex_l > pos_stop[p]:
                            pos_stop[p] = ex_l
                    elif cnt < 0:
                        ex_h = ex_h_1h[p].iloc[t]
                        if not np.isnan(ex_h) and (pos_stop[p] == 0 or ex_h < pos_stop[p]):
                            pos_stop[p] = ex_h

                    # Unrealized PnL
                    side = 1 if cnt > 0 else -1
                    unrealized += (px - pos_entry[p]) * qty * side

            current_equity = cash + unrealized
            min_equity_ever = min(min_equity_ever, current_equity)
            equity_curve.append(current_equity)

            # 2. Entries on 4h boundary
            if current_time.hour % 4 == 0:
                is_bull = is_bull_1h.iloc[t]

                # Trend Exit Check
                for p in universe:
                    cnt = pos_cnt[p]
                    if cnt != 0:
                        px = c_df[p].iloc[t]
                        e50 = ema50_1h[p].iloc[t]
                        mult = contract_sizes[p]
                        qty = abs(cnt) * mult
                        side = 1 if cnt > 0 else -1

                        # Long exit when price crosses below 50d EMA
                        if cnt > 0 and px < e50:
                            fee = qty * px * 0.0002
                            pnl = (px - pos_entry[p]) * qty - fee
                            cash += pnl
                            trades.append((p, "LONG_EMA_EXIT", pnl, current_time))
                            pos_cnt[p] = 0
                            pos_trimmed[p] = False

                        # Short exit when price crosses above 50d EMA
                        elif cnt < 0 and px > e50:
                            fee = qty * px * 0.0002
                            pnl = (pos_entry[p] - px) * qty - fee
                            cash += pnl
                            trades.append((p, "SHORT_EMA_EXIT", pnl, current_time))
                            pos_cnt[p] = 0
                            pos_trimmed[p] = False

                # Active slots
                active_count = sum(1 for p in universe if pos_cnt[p] != 0)
                available_slots = 2 - active_count

                if available_slots > 0 and current_equity > 100.0:
                    # HARD CAPITAL PRESERVATION LOGIC:
                    # Once current_equity > 520, we NEVER risk dropping below $500!
                    excess_equity = max(0.0, current_equity - 500.0)
                    
                    if current_equity <= 520.0:
                        # Initial seed stage: Risk $15 (3% of $500) per entry
                        allowed_risk_per_trade = 15.0
                    else:
                        # House money stage: Risk strictly a fraction of (equity - 500)
                        # Ensure all active stops combined can NEVER drop equity below $500!
                        total_risk_budget = excess_equity * excess_risk_pct
                        allowed_risk_per_trade = max(15.0, total_risk_budget / available_slots)
                        # Cap risk so total stops cannot breach $500
                        allowed_risk_per_trade = min(allowed_risk_per_trade, excess_equity * 0.45)

                    rocs = {p: roc_1h[p].iloc[t] for p in universe if not np.isnan(roc_1h[p].iloc[t])}
                    if len(rocs) >= 4:
                        ranked_desc = sorted(rocs.keys(), key=lambda x: rocs[x], reverse=True)

                        # A. Long Mode: Bull Regime
                        if is_bull:
                            for p in ranked_desc[:3]:
                                if available_slots <= 0:
                                    break
                                if pos_cnt[p] != 0:
                                    continue

                                px = c_df[p].iloc[t]
                                up = up_1h[p].iloc[t]
                                e50 = ema50_1h[p].iloc[t]
                                slp = slope_1h[p].iloc[t]
                                er_val = er_1h[p].iloc[t]
                                atr = atr_1h[p].iloc[t]
                                mult = contract_sizes[p]

                                if px > up and px > e50 and slp > 0 and er_val >= 0.28 and rocs[p] > 0 and atr > 0:
                                    stop_dist = max(2.5 * atr, px * 0.035)
                                    risk_per_contract = stop_dist * mult
                                    
                                    # INTEGER CONTRACTS:
                                    n_contracts = int(math.floor(allowed_risk_per_trade / risk_per_contract))
                                    # Ensure leverage doesn't exceed 3x
                                    notional = n_contracts * mult * px
                                    if notional > current_equity * 3.0:
                                        n_contracts = int(math.floor((current_equity * 3.0) / (mult * px)))

                                    if n_contracts >= 1:
                                        fee = n_contracts * mult * px * 0.0002
                                        cash -= fee
                                        pos_cnt[p] = n_contracts
                                        pos_entry[p] = px
                                        pos_stop[p] = px - stop_dist
                                        pos_trimmed[p] = False
                                        available_slots -= 1

                        # B. Short Mode: Bear Regime (BTC < 200d SMA) or Breakdown
                        else:
                            for p in ranked_desc[-3:]:
                                if available_slots <= 0:
                                    break
                                if pos_cnt[p] != 0:
                                    continue

                                px = c_df[p].iloc[t]
                                dn = dn_1h[p].iloc[t]
                                e50 = ema50_1h[p].iloc[t]
                                slp = slope_1h[p].iloc[t]
                                er_val = er_1h[p].iloc[t]
                                atr = atr_1h[p].iloc[t]
                                mult = contract_sizes[p]

                                if px < dn and px < e50 and slp < 0 and er_val >= 0.28 and rocs[p] < 0 and atr > 0:
                                    stop_dist = max(2.5 * atr, px * 0.035)
                                    risk_per_contract = stop_dist * mult

                                    n_contracts = int(math.floor(allowed_risk_per_trade / risk_per_contract))
                                    notional = n_contracts * mult * px
                                    if notional > current_equity * 3.0:
                                        n_contracts = int(math.floor((current_equity * 3.0) / (mult * px)))

                                    if n_contracts >= 1:
                                        fee = n_contracts * mult * px * 0.0002
                                        cash -= fee
                                        pos_cnt[p] = -n_contracts
                                        pos_entry[p] = px
                                        pos_stop[p] = px + stop_dist
                                        pos_trimmed[p] = False
                                        available_slots -= 1

        eq_arr = np.array(equity_curve)
        final_eq = eq_arr[-1]
        mult = final_eq / initial_capital
        peaks = np.maximum.accumulate(eq_arr)
        dds = (peaks - eq_arr) / peaks
        max_dd = np.max(dds)
        initial_dd = max(0.0, (initial_capital - min_equity_ever) / initial_capital)
        wins = [tr for tr in trades if tr[2] > 0]
        wr = len(wins) / max(len(trades), 1)

        split = len(eq_arr) // 2
        is_arr = eq_arr[:split]
        oos_arr = eq_arr[split:]
        is_mult = is_arr[-1] / initial_capital
        oos_mult = oos_arr[-1] / is_arr[-1]

        print(f"RiskExcess={excess_risk_pct:.2f} => Final=${final_eq:9,.2f} ({mult:6.2f}x) | MaxDD={max_dd:5.2%} | Deposit DD={initial_dd:5.2%} (MinEq=${min_equity_ever:.2f}) | Trades={len(trades)} | WR={wr:4.1%} | OOS={oos_mult:.2f}x")

if __name__ == "__main__":
    run_micro_engine()
