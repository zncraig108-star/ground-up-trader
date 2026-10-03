"""scratch/test_panic_reversion.py — Asymmetric "Buy Fear in Bull Regimes" Engine.

Hypothesis: Crypto bull markets climb a wall of worry. Leverage flushes (liquidation wicks)
produce violent, temporary dislocations where 4h RSI drops below 28-30 while BTC remains
in a macro bull regime (above 200d SMA). Buying these panic dips with tight structural stops
yields high win-rate, high-asymmetry rebounds.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data

def run_panic_engine(
    bundle,
    rsi_thresh=28,      # Entry when 4h RSI <= rsi_thresh
    risk_pct=0.015,     # Risk 1.5% equity per trade
    stop_atr_mult=2.0,  # Stop loss distance
    tp_rsi=55,          # Take profit when 4h RSI reaches tp_rsi
    max_hold_bars=18,   # Max hold 18 4h bars (72 hours)
    max_active=3        # Max concurrent positions
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

    # RSI 14 on 4h
    delta = c_4h.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / np.maximum(loss, 1e-8)
    rsi_4h = (100.0 - (100.0 / (1.0 + rs))).shift(1)

    # ATR 14 on 4h
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14).mean().shift(1)
    atr_df_4h = pd.DataFrame(atr_4h, index=c_4h.index)

    # BTC Macro regime (200d SMA = 1200 4h bars)
    btc_sma = c_4h["BTC-PERP"].rolling(1200, min_periods=120).mean().shift(1)
    is_bull_4h = (c_4h["BTC-PERP"] > btc_sma).shift(1).fillna(False)

    # Reindex to 1h
    rsi_1h = rsi_4h.reindex(common_idx).ffill()
    atr_1h = atr_df_4h.reindex(common_idx).ffill()
    is_bull_1h = is_bull_4h.reindex(common_idx).ffill()

    # Simulation State
    cash = 10000.0
    equity_curve = np.zeros(n_bars)

    pos_qty = {p: 0.0 for p in universe}
    pos_entry = {p: 0.0 for p in universe}
    pos_stop = {p: 0.0 for p in universe}
    pos_bars = {p: 0 for p in universe}
    pos_r_dist = {p: 0.0 for p in universe}

    trades = []

    for t in range(1200, n_bars):
        current_time = common_idx[t]

        # 1. Hourly Stop and Profit Evaluation
        unrealized = 0.0
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                px = c_df[p].iloc[t]
                lo = l_df[p].iloc[t]
                stop = pos_stop[p]
                entry = pos_entry[p]
                r_dist = pos_r_dist[p]
                bars_held = pos_bars[p] + 1
                pos_bars[p] = bars_held

                # Check Stop Loss
                if lo <= stop:
                    exit_px = min(px, stop)
                    spec = inst_specs.get(p, InstrumentSpec(p))
                    fee = q * exit_px * spec.taker_fee_rate
                    pnl = (exit_px - entry) * q - fee
                    cash += (exit_px - entry) * q - fee
                    r_mult = (exit_px - entry) / max(r_dist, 1e-6)
                    trades.append((p, current_time, entry, exit_px, pnl, r_mult, "STOP"))
                    pos_qty[p] = 0.0
                    continue

                # Check Take Profit on 4h boundary (RSI mean reverted) or Time Expiry
                if current_time.hour % 4 == 0:
                    curr_rsi = rsi_1h[p].iloc[t]
                    if curr_rsi >= tp_rsi or bars_held >= max_hold_bars * 4:
                        spec = inst_specs.get(p, InstrumentSpec(p))
                        fee = q * px * spec.maker_fee_rate
                        pnl = (px - entry) * q - fee
                        cash += (px - entry) * q - fee
                        r_mult = (px - entry) / max(r_dist, 1e-6)
                        trades.append((p, current_time, entry, px, pnl, r_mult, "TP" if curr_rsi >= tp_rsi else "TIME"))
                        pos_qty[p] = 0.0
                        continue

                unrealized += (px - entry) * q

        current_equity = cash + unrealized
        equity_curve[t] = current_equity

        # 2. Entries on 4h boundary
        if current_time.hour % 4 == 0 and current_equity > 100.0:
            is_bull = is_bull_1h.iloc[t]

            # In Bull regimes only
            if is_bull:
                active_count = sum(1 for p in universe if abs(pos_qty[p]) > 1e-9)
                slots_available = max_active - active_count

                if slots_available > 0:
                    # Find assets with RSI <= rsi_thresh
                    candidates = []
                    for p in universe:
                        if abs(pos_qty[p]) > 1e-9:
                            continue
                        r_val = rsi_1h[p].iloc[t]
                        if not np.isnan(r_val) and r_val <= rsi_thresh:
                            candidates.append((p, r_val))

                    # Sort by lowest RSI (deepest panic)
                    candidates.sort(key=lambda x: x[1])

                    for p, r_val in candidates:
                        if slots_available <= 0:
                            break

                        px = c_df[p].iloc[t]
                        atr = atr_1h[p].iloc[t]
                        if not np.isnan(atr) and atr > 0:
                            stop_dist = stop_atr_mult * atr
                            initial_stop = px - stop_dist

                            dollar_risk = current_equity * risk_pct
                            q_target = dollar_risk / stop_dist

                            spec = inst_specs.get(p, InstrumentSpec(p))
                            q = spec.floor_to_lot(q_target)
                            notional = q * px

                            # Max leverage safety cap: 1.0x equity per position
                            if notional > current_equity * 1.0:
                                q = spec.floor_to_lot((current_equity * 1.0) / px)

                            if q * px >= spec.min_notional:
                                fee = q * px * spec.maker_fee_rate
                                cash -= fee
                                pos_qty[p] = q
                                pos_entry[p] = px
                                pos_stop[p] = initial_stop
                                pos_r_dist[p] = stop_dist
                                pos_bars[p] = 0
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
    print("=== TESTING PANIC MEAN-REVERSION IN MACRO BULL REGIMES ===")
    
    for rsi_th in [25, 28, 30, 32]:
        for risk in [0.01, 0.015, 0.02]:
            for stop_atr in [1.5, 2.0, 2.5]:
                eq, mult, dd, sh, n, wr, ar = run_panic_engine(
                    bundle,
                    rsi_thresh=rsi_th,
                    risk_pct=risk,
                    stop_atr_mult=stop_atr,
                    tp_rsi=50,
                    max_hold_bars=12, # 48h max hold
                    max_active=3
                )
                print(f"RSI<={rsi_th:2d} | Risk={risk*100:.1f}% | Stop={stop_atr}ATR => Final=${eq:,.2f} ({mult:5.2f}x) | Max DD={dd:5.2%} | Sharpe={sh:4.2f} | Trades={n:3d} | WR={wr:4.1%} | AvgR={ar:+.2f}")

if __name__ == "__main__":
    main()
