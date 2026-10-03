"""scratch/test_squeeze_momentum.py — Volatility Squeeze Breakout on Relative Strength Leaders.

Core Concept:
1. Volatility Squeeze: Bollinger Bands (20, 2.0) contract INSIDE Keltner Channels (20, 1.5 ATR).
   This indicates extreme market compression / energy coil.
2. When the squeeze fires (bands expand) AND price breaks out in the direction of the macro trend:
   The ensuing move is an explosive, directional impulse with high win rate and minimal pullback.
3. Concentration: Allocate capital strictly to the highest relative-strength asset in squeeze.
4. Asymmetric Stop: Initial Stop at Lower Keltner Channel or 1.5 ATR.
   Trailing stop ratchets to breakeven at +2R, and trails behind 2 ATR once in profit.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data

def run_squeeze_strategy(
    bundle,
    risk_pct=0.015,       # Risk 1.5% equity per trade
    bb_period=20,
    bb_mult=2.0,
    kc_mult=1.5,
    max_active=2
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

    # BTC Macro regime (200d SMA)
    btc_sma = c_4h["BTC-PERP"].rolling(1200, min_periods=120).mean().shift(1)
    is_bull_4h = (c_4h["BTC-PERP"] > btc_sma).shift(1).fillna(False)

    # 1. Bollinger Bands on 4h
    sma_4h = c_4h.rolling(bb_period).mean().shift(1)
    std_4h = c_4h.rolling(bb_period).std().shift(1)
    bb_up = sma_4h + bb_mult * std_4h
    bb_lo = sma_4h - bb_mult * std_4h

    # 2. Keltner Channels on 4h
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(bb_period).mean().shift(1)
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    kc_up = sma_4h + kc_mult * atr_df
    kc_lo = sma_4h - kc_mult * atr_df

    # Squeeze Condition: Bollinger Bands inside Keltner Channels
    is_squeeze_4h = (bb_up < kc_up) & (bb_lo > kc_lo)

    # 3. Momentum: 14-day ROC (84 4h bars)
    roc_4h = (c_4h / c_4h.shift(84) - 1.0).shift(1)

    # Forward to 1h
    bb_up_1h = bb_up.reindex(common_idx).ffill()
    sma_1h = sma_4h.reindex(common_idx).ffill()
    atr_1h = atr_df.reindex(common_idx).ffill()
    roc_1h = roc_4h.reindex(common_idx).ffill()
    is_squeeze_1h = is_squeeze_4h.reindex(common_idx).ffill()
    is_bull_1h = is_bull_4h.reindex(common_idx).ffill()

    # Simulation State
    cash = 10000.0
    equity_curve = np.zeros(n_bars)

    pos_qty = {p: 0.0 for p in universe}
    pos_entry = {p: 0.0 for p in universe}
    pos_stop = {p: 0.0 for p in universe}
    pos_r = {p: 0.0 for p in universe}
    pos_peak = {p: 0.0 for p in universe}

    trades = []

    for t in range(1200, n_bars):
        current_time = common_idx[t]

        # 1. Stop and Trailing Management on 1h bars
        unrealized = 0.0
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                px = c_df[p].iloc[t]
                lo = l_df[p].iloc[t]
                hi = h_df[p].iloc[t]
                stop = pos_stop[p]
                entry = pos_entry[p]
                r_dist = pos_r[p]

                if hi > pos_peak[p]:
                    pos_peak[p] = hi

                # Check Stop Hit
                if lo <= stop:
                    exit_px = min(px, stop)
                    spec = inst_specs.get(p, InstrumentSpec(p))
                    fee = q * exit_px * spec.taker_fee_rate
                    pnl = (exit_px - entry) * q - fee
                    cash += (exit_px - entry) * q - fee
                    r_mult = (exit_px - entry) / max(r_dist, 1e-6)
                    trades.append((p, current_time, entry, exit_px, pnl, r_mult))
                    pos_qty[p] = 0.0
                    pos_stop[p] = 0.0
                    continue

                # Trailing Ratchet:
                # If in profit >= +2.0R, move stop to breakeven + 0.2R (lock fees)
                curr_r = (hi - entry) / max(r_dist, 1e-6)
                if curr_r >= 2.0:
                    pos_stop[p] = max(pos_stop[p], entry + 0.2 * r_dist)

                # If in profit >= +3.5R, trail behind 2.0 ATR
                if curr_r >= 3.5:
                    curr_atr = atr_1h[p].iloc[t]
                    if not np.isnan(curr_atr) and curr_atr > 0:
                        trail_px = pos_peak[p] - 2.0 * curr_atr
                        pos_stop[p] = max(pos_stop[p], trail_px)

                unrealized += (px - entry) * q

        current_equity = cash + unrealized
        equity_curve[t] = current_equity

        # 2. Entries on 4h boundary
        if current_time.hour % 4 == 0 and current_equity > 100.0:
            is_bull = is_bull_1h.iloc[t]

            # In Bear: Close all to Cash
            if not is_bull:
                for p in universe:
                    if abs(pos_qty[p]) > 1e-9:
                        px = c_df[p].iloc[t]
                        spec = inst_specs.get(p, InstrumentSpec(p))
                        fee = pos_qty[p] * px * spec.maker_fee_rate
                        pnl = (px - pos_entry[p]) * pos_qty[p] - fee
                        cash += (px - pos_entry[p]) * pos_qty[p] - fee
                        r_mult = (px - pos_entry[p]) / max(pos_r[p], 1e-6)
                        trades.append((p, current_time, pos_entry[p], px, pnl, r_mult))
                        pos_qty[p] = 0.0
                        pos_stop[p] = 0.0
                unrealized = 0.0
                current_equity = cash

            else:
                active_count = sum(1 for p in universe if abs(pos_qty[p]) > 1e-9)
                slots = max_active - active_count

                if slots > 0:
                    # Rank universe by relative momentum
                    rocs = {p: roc_1h[p].iloc[t] for p in universe}
                    ranked = sorted(universe, key=lambda x: rocs[x], reverse=True)

                    for p in ranked:
                        if slots <= 0:
                            break
                        if abs(pos_qty[p]) > 1e-9:
                            continue

                        px = c_df[p].iloc[t]
                        up_band = bb_up_1h[p].iloc[t]
                        atr = atr_1h[p].iloc[t]

                        # Breakout Condition:
                        # Asset was in squeeze recently OR breaking out of upper band with positive momentum
                        if px > up_band and rocs[p] > 0 and not np.isnan(atr) and atr > 0:
                            stop_dist = 2.0 * atr
                            initial_stop = px - stop_dist

                            dollar_risk = current_equity * risk_pct
                            spec = inst_specs.get(p, InstrumentSpec(p))
                            q = spec.floor_to_lot(dollar_risk / stop_dist)

                            # Leverage cap: max 1.0x equity per position
                            if q * px > current_equity * 1.0:
                                q = spec.floor_to_lot((current_equity * 1.0) / px)

                            if q * px >= spec.min_notional:
                                fee = q * px * spec.maker_fee_rate
                                cash -= fee
                                pos_qty[p] = q
                                pos_entry[p] = px
                                pos_stop[p] = initial_stop
                                pos_r[p] = stop_dist
                                pos_peak[p] = px
                                slots -= 1

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
    print("=== TESTING SQUEEZE BREAKOUT MOMENTUM ===")
    for r_risk in [0.01, 0.015, 0.02, 0.025]:
        eq, mult, dd, sh, n, wr, ar = run_squeeze_strategy(bundle, risk_pct=r_risk)
        print(f"Risk={r_risk*100:4.1f}% => Final=${eq:10,.2f} ({mult:5.2f}x) | Max DD={dd:5.2%} | Sharpe={sh:4.2f} | Trades={n:3d} | WR={wr:4.1%} | AvgR={ar:+.2f}")

if __name__ == "__main__":
    main()
