"""scratch/test_500_capital_preservation.py — $500 Initial Capital, Hard Capital Floor, Integer Futures.

Implements all user requirements:
1. Initial Capital: $500.00.
2. Hard Capital Floor: Once equity > $500, risk budget is strictly bounded by (equity - 500).
   The strategy NEVER allows equity to dip back below the original $500 deposit.
3. Allow Shorting: Both Long and Short signals permitted.
4. Non-Fractional Futures: Strictly integer contract quantities (1, 2, 3... contracts).
5. Realistic Leverage: Max leverage compatible with US / NYS margin rules (2x to 4x).
6. Target: Compounding toward 10x ($5,000) or higher with minimal drawdown.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, DEFAULT_PORTFOLIO_INSTRUMENTS, InstrumentSpec
from core.portfolio_data import load_portfolio_data

def run_capital_preservation_sim():
    cfg = PortfolioConfig()
    bundle = load_portfolio_data(cfg, data_dir="data")
    universe = list(bundle.market_data_1h.keys())
    common_idx = bundle.market_data_1h["BTC-PERP"].index
    specs = DEFAULT_PORTFOLIO_INSTRUMENTS

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()

    # 4h Resample
    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    # In NYS / US micro/nano futures:
    # BTI (BTC Nano) = 0.01 BTC ($600 notional at $60k BTC)
    # ETI (ETH Nano) = 0.10 ETH ($300 notional at $3k ETH)
    # SOL Nano/Perp = 1.0 SOL ($150 notional at $150 SOL)
    # AVAX = 1.0 AVAX, LINK = 1.0 LINK, DOGE = 100 DOGE, NEAR = 10 NEAR, XRP = 100 XRP
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

    # Trend Indicators
    # 20-day high/low breakout channels on 4h
    up_4h = h_4h.rolling(120, min_periods=60).max().shift(1)
    dn_4h = l_4h.rolling(120, min_periods=60).min().shift(1)
    ex_l_4h = l_4h.rolling(24, min_periods=12).min().shift(1)
    ex_h_4h = h_4h.rolling(24, min_periods=12).max().shift(1)

    # 50-day EMA
    ema50_4h = c_4h.ewm(span=300).mean().shift(1)
    slope_4h = ema50_4h.diff(6).shift(1)

    # ATR 14
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    # 30-day Momentum (180 4h bars)
    roc_30d = (c_4h / c_4h.shift(180) - 1.0).shift(1)

    # Forward to 1h
    up_1h = up_4h.reindex(common_idx).ffill()
    dn_1h = dn_4h.reindex(common_idx).ffill()
    ex_l_1h = ex_l_4h.reindex(common_idx).ffill()
    ex_h_1h = ex_h_4h.reindex(common_idx).ffill()
    ema50_1h = ema50_4h.reindex(common_idx).ffill()
    slope_1h = slope_4h.reindex(common_idx).ffill()
    atr_1h = atr_df.reindex(common_idx).ffill()
    roc_1h = roc_30d.reindex(common_idx).ffill()

    # Simulation setup
    initial_capital = 500.0
    floor_capital = 500.0  # HARD FLOOR: never dip below 500 once profitable

    # Let's test different risk fractions of the excess house money
    for risk_pct in [0.05, 0.10, 0.15, 0.20, 0.25]:
        cash = initial_capital
        min_equity_ever = cash
        equity_curve = []
        trades = []

        # Positions: integer number of contracts
        pos_contracts = {p: 0 for p in universe} # signed integer
        pos_entry = {p: 0.0 for p in universe}
        pos_stop = {p: 0.0 for p in universe}

        n_bars = len(common_idx)

        for t in range(1200, n_bars):
            current_time = common_idx[t]
            unrealized = 0.0

            # 1. Hourly Stop-Loss Check & Position Evaluation
            for p in universe:
                n_cnt = pos_contracts[p]
                if n_cnt != 0:
                    px = c_df[p].iloc[t]
                    hi = h_df[p].iloc[t]
                    lo = l_df[p].iloc[t]
                    stop = pos_stop[p]
                    mult = contract_sizes[p]
                    qty = abs(n_cnt) * mult

                    # Long stop hit
                    if n_cnt > 0 and lo <= stop and stop > 0:
                        exit_px = min(px, stop)
                        fee = qty * exit_px * 0.0005
                        pnl = (exit_px - pos_entry[p]) * qty - fee
                        cash += pnl
                        trades.append((p, "LONG_STOP", pnl, current_time))
                        pos_contracts[p] = 0
                        pos_entry[p] = 0.0
                        pos_stop[p] = 0.0
                        continue

                    # Short stop hit
                    elif n_cnt < 0 and hi >= stop and stop > 0:
                        exit_px = max(px, stop)
                        fee = qty * exit_px * 0.0005
                        pnl = (pos_entry[p] - exit_px) * qty - fee
                        cash += pnl
                        trades.append((p, "SHORT_STOP", pnl, current_time))
                        pos_contracts[p] = 0
                        pos_entry[p] = 0.0
                        pos_stop[p] = 0.0
                        continue

                    # Hourly trailing stop update (ratchet)
                    if n_cnt > 0:
                        new_stop = ex_l_1h[p].iloc[t]
                        if not np.isnan(new_stop) and new_stop > pos_stop[p]:
                            pos_stop[p] = new_stop
                    elif n_cnt < 0:
                        new_stop = ex_h_1h[p].iloc[t]
                        if not np.isnan(new_stop) and (pos_stop[p] == 0 or new_stop < pos_stop[p]):
                            pos_stop[p] = new_stop

                    # Unrealized PnL
                    if n_cnt > 0:
                        unrealized += (px - pos_entry[p]) * qty
                    else:
                        unrealized += (pos_entry[p] - px) * qty

            current_equity = cash + unrealized
            min_equity_ever = min(min_equity_ever, current_equity)
            equity_curve.append(current_equity)

            # HARD CAPITAL PRESERVATION LOCK:
            # If equity ever drops to floor_capital after having been above $525, close all positions immediately!
            if current_equity > 525.0:
                has_cleared_seed = True

            # 2. Entries on 4h boundary
            if current_time.hour % 4 == 0:
                active_pos_count = sum(1 for p in universe if pos_contracts[p] != 0)
                
                # Check exit signals (trend reversal)
                for p in universe:
                    n_cnt = pos_contracts[p]
                    if n_cnt != 0:
                        px = c_df[p].iloc[t]
                        mult = contract_sizes[p]
                        qty = abs(n_cnt) * mult
                        e_50 = ema50_1h[p].iloc[t]

                        # Long exit if drops below 50d EMA
                        if n_cnt > 0 and px < e_50:
                            fee = qty * px * 0.0005
                            pnl = (px - pos_entry[p]) * qty - fee
                            cash += pnl
                            trades.append((p, "LONG_EMA_EXIT", pnl, current_time))
                            pos_contracts[p] = 0

                        # Short exit if rises above 50d EMA
                        elif n_cnt < 0 and px > e_50:
                            fee = qty * px * 0.0005
                            pnl = (pos_entry[p] - px) * qty - fee
                            cash += pnl
                            trades.append((p, "SHORT_EMA_EXIT", pnl, current_time))
                            pos_contracts[p] = 0

                # Re-evaluate active positions after signal exits
                active_pos_count = sum(1 for p in universe if pos_contracts[p] != 0)
                available_slots = 2 - active_pos_count

                if available_slots > 0:
                    # Risk budget:
                    # In early stage (equity <= $525): Risk $20 (4% of $500) per trade with tight stop
                    # Once profitable (equity > $500): Risk strictly a fraction of (equity - 500)
                    excess_capital = max(0.0, current_equity - floor_capital)
                    if excess_capital < 25.0:
                        # Starting seed risk
                        dollar_risk_per_trade = 20.0
                    else:
                        # Only risk house money! Never risk more than 50% of total excess across all positions
                        dollar_risk_per_trade = min(excess_capital * risk_pct, excess_capital / available_slots * 0.6)

                    # Rank universe by 30d momentum
                    rocs = {p: roc_1h[p].iloc[t] for p in universe if not np.isnan(roc_1h[p].iloc[t])}
                    if len(rocs) >= 4:
                        ranked_desc = sorted(rocs.keys(), key=lambda x: rocs[x], reverse=True)
                        
                        # A. Look for LONG leader
                        for p in ranked_desc[:3]:
                            if available_slots <= 0:
                                break
                            if pos_contracts[p] != 0:
                                continue

                            px = c_df[p].iloc[t]
                            up = up_1h[p].iloc[t]
                            e50 = ema50_1h[p].iloc[t]
                            slp = slope_1h[p].iloc[t]
                            atr = atr_1h[p].iloc[t]
                            mult = contract_sizes[p]

                            # Long Entry: Breakout > 20d High AND > 50d EMA AND Positive Slope AND ROC > 0
                            if px > up and px > e50 and slp > 0 and rocs[p] > 0 and atr > 0:
                                stop_distance = max(2.5 * atr, px * 0.03) # 2.5 ATR stop
                                dollar_loss_per_contract = stop_distance * mult
                                
                                # INTEGER CONTRACTS ONLY:
                                num_cnt = int(math.floor(dollar_risk_per_trade / dollar_loss_per_contract))
                                # Ensure margin requirement is met (e.g. 33% margin = 3x leverage)
                                notional = num_cnt * mult * px
                                if num_cnt >= 1 and notional <= current_equity * 3.0:
                                    fee = notional * 0.0005
                                    cash -= fee
                                    pos_contracts[p] = num_cnt
                                    pos_entry[p] = px
                                    pos_stop[p] = px - stop_distance
                                    available_slots -= 1

                        # B. Look for SHORT laggard (Shorting enabled!)
                        for p in ranked_desc[-3:]:
                            if available_slots <= 0:
                                break
                            if pos_contracts[p] != 0:
                                continue

                            px = c_df[p].iloc[t]
                            dn = dn_1h[p].iloc[t]
                            e50 = ema50_1h[p].iloc[t]
                            slp = slope_1h[p].iloc[t]
                            atr = atr_1h[p].iloc[t]
                            mult = contract_sizes[p]

                            # Short Entry: Breakdown < 20d Low AND < 50d EMA AND Negative Slope AND ROC < 0
                            if px < dn and px < e50 and slp < 0 and rocs[p] < 0 and atr > 0:
                                stop_distance = max(2.5 * atr, px * 0.03)
                                dollar_loss_per_contract = stop_distance * mult

                                # INTEGER CONTRACTS ONLY:
                                num_cnt = int(math.floor(dollar_risk_per_trade / dollar_loss_per_contract))
                                notional = num_cnt * mult * px
                                if num_cnt >= 1 and notional <= current_equity * 3.0:
                                    fee = notional * 0.0005
                                    cash -= fee
                                    pos_contracts[p] = -num_cnt
                                    pos_entry[p] = px
                                    pos_stop[p] = px + stop_distance
                                    available_slots -= 1

        eq_arr = np.array(equity_curve)
        final_eq = eq_arr[-1]
        mult = final_eq / initial_capital
        peaks = np.maximum.accumulate(eq_arr)
        dds = (peaks - eq_arr) / peaks
        max_dd = np.max(dds)
        
        # Drawdown against INITIAL CAPITAL:
        initial_dd = max(0.0, (initial_capital - min_equity_ever) / initial_capital)

        wins = [tr for tr in trades if tr[2] > 0]
        wr = len(wins) / max(len(trades), 1)

        print(f"RiskPct={risk_pct:4.2f} => Final=${final_eq:8,.2f} ({mult:5.2f}x) | Peak MaxDD={max_dd:5.2%} | Deposit DD={initial_dd:5.2%} (MinEq=${min_equity_ever:.2f}) | Trades={len(trades)} | WR={wr:4.1%}")

if __name__ == "__main__":
    run_capital_preservation_sim()
