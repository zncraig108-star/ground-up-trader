"""scratch/test_r_multiples.py — Rigorous R-Multiples Risk-Budgeted Convex Trend Engine.

Every trade risks strictly 1.0% (1R) of equity at entry.
Initial Stop is placed at swing low or ATR distance.
When trade reaches +1.5R, stop moves to Breakeven (0R risk).
When trade reaches +3.0R, trailing stop ratchets behind 1.5 ATR.
Zero lookahead: all signals and stops evaluated on closed bars.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data

def run_r_engine(
    bundle,
    risk_pct=0.01,       # 1.0% risk per trade (1R)
    atr_mult=2.5,        # Initial stop at entry - (atr_mult * ATR)
    breakeven_r=1.5,     # Move stop to BE at +1.5R
    trail_r=3.0,         # Start trailing ATR stop at +3.0R
    trail_atr_mult=2.0,  # Trailing stop distance
    max_active=3,        # Max concurrent positions
    lookback_breakout=40 # 40 4h bars = ~6.6 days breakout
):
    universe = list(bundle.market_data_1h.keys())
    common_idx = bundle.market_data_1h["BTC-PERP"].index
    n_bars = len(common_idx)
    inst_specs = DEFAULT_PORTFOLIO_INSTRUMENTS

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()

    # 4h Resample
    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    # Breakout channel (lookback_breakout bars, shifted by 1 to prevent lookahead)
    breakout_high_4h = h_4h.rolling(lookback_breakout, min_periods=20).max().shift(1)
    
    # 14-period ATR on 4h
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df_4h = pd.DataFrame(atr_4h, index=c_4h.index)

    # 30-day Momentum (180 4h bars)
    roc_4h = (c_4h / c_4h.shift(180) - 1.0).shift(1)

    # BTC Macro regime (200d SMA = 1200 4h bars)
    btc_sma = c_4h["BTC-PERP"].rolling(1200, min_periods=120).mean().shift(1)
    is_bull_4h = (c_4h["BTC-PERP"] > btc_sma).shift(1).fillna(False)

    # Reindex to 1h
    breakout_1h = breakout_high_4h.reindex(common_idx).ffill()
    atr_1h = atr_df_4h.reindex(common_idx).ffill()
    roc_1h = roc_4h.reindex(common_idx).ffill()
    is_bull_1h = is_bull_4h.reindex(common_idx).ffill()

    # Simulation
    equity = 10000.0
    cash = 10000.0
    equity_curve = np.zeros(n_bars)

    # State per asset
    pos_qty = {p: 0.0 for p in universe}
    pos_entry = {p: 0.0 for p in universe}
    pos_stop = {p: 0.0 for p in universe}
    pos_r_dist = {p: 0.0 for p in universe} # dollar distance of 1R
    pos_peak = {p: 0.0 for p in universe}

    trades = []

    for t in range(1200, n_bars):
        current_time = common_idx[t]

        # 1. Stop Management & Exits on 1h bars
        unrealized = 0.0
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                px = c_df[p].iloc[t]
                lo = l_df[p].iloc[t]
                hi = h_df[p].iloc[t]
                stop = pos_stop[p]
                r_dist = pos_r_dist[p]
                entry = pos_entry[p]

                # Update peak price for trailing
                if hi > pos_peak[p]:
                    pos_peak[p] = hi

                # Check if stop hit (intrabar low breaches stop)
                if lo <= stop:
                    exit_px = min(px, stop)
                    spec = inst_specs.get(p, InstrumentSpec(p))
                    fee = q * exit_px * spec.taker_fee_rate
                    pnl = (exit_px - entry) * q - fee
                    cash += (exit_px - entry) * q - fee
                    r_multiple = (exit_px - entry) / max(r_dist, 1e-6)
                    trades.append((p, current_time, entry, exit_px, pnl, r_multiple))
                    pos_qty[p] = 0.0
                    pos_stop[p] = 0.0
                    continue

                # Dynamic Stop Ratchets based on R-Multiples
                paper_gain_r = (hi - entry) / max(r_dist, 1e-6)

                # Ratchet 1: Breakeven (+0.1R buffer for fees) at breakeven_r
                if paper_gain_r >= breakeven_r:
                    be_stop = entry + 0.1 * r_dist
                    if be_stop > pos_stop[p]:
                        pos_stop[p] = be_stop

                # Ratchet 2: Trailing stop at trail_r
                if paper_gain_r >= trail_r:
                    curr_atr = atr_1h[p].iloc[t]
                    if not np.isnan(curr_atr) and curr_atr > 0:
                        trail_stop = pos_peak[p] - trail_atr_mult * curr_atr
                        if trail_stop > pos_stop[p]:
                            pos_stop[p] = trail_stop

                unrealized += (px - entry) * q

        current_equity = cash + unrealized
        equity_curve[t] = current_equity

        # 2. Entries on 4h boundary
        if current_time.hour % 4 == 0 and current_equity > 100.0:
            is_bull = is_bull_1h.iloc[t]

            # If bear market: close any remaining positions
            if not is_bull:
                for p in universe:
                    if abs(pos_qty[p]) > 1e-9:
                        px = c_df[p].iloc[t]
                        spec = inst_specs.get(p, InstrumentSpec(p))
                        fee = pos_qty[p] * px * spec.maker_fee_rate
                        pnl = (px - pos_entry[p]) * pos_qty[p] - fee
                        cash += (px - pos_entry[p]) * pos_qty[p] - fee
                        r_mult = (px - pos_entry[p]) / max(pos_r_dist[p], 1e-6)
                        trades.append((p, current_time, pos_entry[p], px, pnl, r_mult))
                        pos_qty[p] = 0.0
                        pos_stop[p] = 0.0
                unrealized = 0.0
                current_equity = cash

            else:
                # In Bull market: count active positions
                active_count = sum(1 for p in universe if abs(pos_qty[p]) > 1e-9)
                slots_available = max_active - active_count

                if slots_available > 0:
                    # Rank assets by momentum
                    rocs = {p: roc_1h[p].iloc[t] for p in universe}
                    ranked = sorted(universe, key=lambda x: rocs[x], reverse=True)

                    for p in ranked:
                        if slots_available <= 0:
                            break
                        if abs(pos_qty[p]) > 1e-9:
                            continue # already open

                        px = c_df[p].iloc[t]
                        hi_breakout = breakout_1h[p].iloc[t]
                        atr = atr_1h[p].iloc[t]

                        # Entry condition: Close broke out above lookback channel AND momentum positive
                        if px > hi_breakout and rocs[p] > 0 and not np.isnan(atr) and atr > 0:
                            # 1R Stop distance
                            stop_dist = atr_mult * atr
                            initial_stop = px - stop_dist

                            # Position size: risk strictly risk_pct * current_equity
                            dollar_risk = current_equity * risk_pct
                            q_target = dollar_risk / stop_dist

                            spec = inst_specs.get(p, InstrumentSpec(p))
                            q = spec.floor_to_lot(q_target)
                            notional = q * px

                            # Max leverage safety cap per position: 1.0x equity
                            if notional > current_equity * 1.0:
                                q = spec.floor_to_lot((current_equity * 1.0) / px)

                            if q * px >= spec.min_notional:
                                fee = q * px * spec.maker_fee_rate
                                cash -= fee
                                pos_qty[p] = q
                                pos_entry[p] = px
                                pos_stop[p] = initial_stop
                                pos_r_dist[p] = stop_dist
                                pos_peak[p] = px
                                slots_available -= 1

    eq_active = equity_curve[1200:]
    peaks = np.maximum.accumulate(eq_active)
    dds = (peaks - eq_active) / np.maximum(peaks, 1e-6)
    max_dd = float(np.max(dds))
    final_eq = float(eq_active[-1])
    mult = final_eq / 10000.0

    rets = pd.Series(eq_active).pct_change().fillna(0.0)
    sharpe = float((rets.mean() / (rets.std() + 1e-8)) * math.sqrt(8760.0))

    wins = [t for t in trades if t[4] > 0]
    wr = len(wins) / max(len(trades), 1)
    avg_r = np.mean([t[5] for t in trades]) if trades else 0.0

    return final_eq, mult, max_dd, sharpe, len(trades), wr, avg_r

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")
    print("=== TESTING R-MULTIPLES RISK-BUDGETED TREND ENGINE ===")
    
    for r_risk in [0.01, 0.015, 0.02]:
        for be_r in [1.2, 1.5, 2.0]:
            for tr_r in [2.5, 3.5]:
                for tr_mult in [1.5, 2.0]:
                    eq, mult, dd, sh, n, wr, ar = run_r_engine(
                        bundle,
                        risk_pct=r_risk,
                        atr_mult=2.0,
                        breakeven_r=be_r,
                        trail_r=tr_r,
                        trail_atr_mult=tr_mult,
                        max_active=3,
                        lookback_breakout=30
                    )
                    print(f"Risk={r_risk*100:.1f}% | BE={be_r}R | Trail={tr_r}R (x{tr_mult}) => Final=${eq:,.2f} ({mult:5.2f}x) | Max DD={dd:5.2%} | Sharpe={sh:4.2f} | Trades={n:3d} | WR={wr:4.1%} | AvgR={ar:+.2f}")

if __name__ == "__main__":
    main()
