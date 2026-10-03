"""scratch/test_high_frequency_asymmetry.py — Testing 1h Asymmetric R:R with 1.0% Fixed Risk."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data

def test_1h_asymmetric(bundle, risk_pct=0.01, stop_atr_mult=1.5, min_rr=3.5):
    universe = [p for p in bundle.market_data_1h.keys()]
    common_idx = bundle.market_data_1h["BTC-PERP"].index
    n_bars = len(common_idx)

    # 1. Compute 1h ATR and Donchian channels for each asset
    atrs = {}
    high_chans = {}
    low_chans = {}
    volumes = {}
    vol_emas = {}

    for p in universe:
        df = bundle.market_data_1h[p].reindex(common_idx)
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]

        tr = np.maximum(h - l, np.maximum(abs(h - c.shift(1)), abs(l - c.shift(1))))
        atrs[p] = tr.rolling(24).mean()
        high_chans[p] = h.rolling(48).max().shift(1) # 48h (2-day) breakout
        low_chans[p] = l.rolling(48).min().shift(1)
        volumes[p] = v
        vol_emas[p] = v.rolling(48).mean().shift(1)

    # BTC Macro filter: BTC close > 50-day (1200h) SMA
    btc_c = bundle.market_data_1h["BTC-PERP"]["close"]
    btc_sma50d = btc_c.rolling(1200).mean().shift(1)
    is_macro_bull = btc_c > btc_sma50d

    # Simulate portfolio
    cash = 10000.0
    equity = 10000.0
    peak_equity = 10000.0
    equity_curve = np.zeros(n_bars)

    pos_qty = {p: 0.0 for p in universe}
    pos_entry = {p: 0.0 for p in universe}
    pos_stop = {p: 0.0 for p in universe}
    pos_target = {p: 0.0 for p in universe}

    trades = []

    for t in range(1200, n_bars):
        # 1. Update open positions and check stops/targets on 1h bar
        unrealized = 0.0
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                px = bundle.market_data_1h[p]["close"].iloc[t]
                hi = bundle.market_data_1h[p]["high"].iloc[t]
                lo = bundle.market_data_1h[p]["low"].iloc[t]
                entry = pos_entry[p]
                stop = pos_stop[p]

                # Long check
                if q > 0:
                    # Check stop hit
                    if lo <= stop:
                        exit_px = min(px, stop)
                        pnl = (exit_px - entry) * q - abs(q) * exit_px * 0.0005
                        cash += (exit_px - entry) * q - abs(q) * exit_px * 0.0005
                        trades.append((p, 1, entry, exit_px, pnl))
                        pos_qty[p] = 0.0
                        continue
                    # Trail stop once in profit >= 2.0 * risk_dist
                    risk_dist = entry - stop
                    if hi >= entry + 2.0 * risk_dist:
                        # Ratchet stop to entry + 1.0 * risk_dist (lock 1R profit)
                        pos_stop[p] = max(pos_stop[p], entry + 1.0 * risk_dist)
                    if hi >= entry + 4.0 * risk_dist:
                        # Lock 2.5R profit
                        pos_stop[p] = max(pos_stop[p], entry + 2.5 * risk_dist)
                    if hi >= entry + 6.0 * risk_dist:
                        # Lock 4.5R profit
                        pos_stop[p] = max(pos_stop[p], entry + 4.5 * risk_dist)

                    # Exit if 1h channel breaches
                    if lo <= low_chans[p].iloc[t]:
                        exit_px = min(px, low_chans[p].iloc[t])
                        pnl = (exit_px - entry) * q - abs(q) * exit_px * 0.0005
                        cash += (exit_px - entry) * q - abs(q) * exit_px * 0.0005
                        trades.append((p, 1, entry, exit_px, pnl))
                        pos_qty[p] = 0.0
                        continue

                    unrealized += (px - entry) * q

        current_equity = cash + unrealized
        peak_equity = max(peak_equity, current_equity)

        # 2. Check entries on 1h bar (Macro Bull only)
        if is_macro_bull.iloc[t] and current_equity > 100.0:
            open_count = sum(1 for p in universe if abs(pos_qty[p]) > 1e-9)
            if open_count < 4: # Max 4 concurrent positions
                for p in universe:
                    if abs(pos_qty[p]) < 1e-9:
                        px = bundle.market_data_1h[p]["close"].iloc[t]
                        atr = atrs[p].iloc[t]
                        upper = high_chans[p].iloc[t]
                        vol = volumes[p].iloc[t]
                        vol_ema = vol_emas[p].iloc[t]

                        if np.isnan(px) or np.isnan(atr) or atr <= 0:
                            continue

                        # Breakout with Volume Confirmation: Close > 48h High AND Volume > 1.2x Volume EMA
                        if px > upper and vol > 1.2 * vol_ema:
                            stop_dist = stop_atr_mult * atr
                            stop_px = px - stop_dist
                            # Sized so that loss if stopped = risk_pct * current_equity
                            dollar_risk = current_equity * risk_pct
                            q = dollar_risk / stop_dist
                            # Cap max notional per position at 0.5x equity
                            max_q = (current_equity * 0.50) / px
                            q = min(q, max_q)

                            fee = q * px * 0.0002
                            cash -= fee
                            pos_qty[p] = q
                            pos_entry[p] = px
                            pos_stop[p] = stop_px
                            open_count += 1
                            if open_count >= 4:
                                break

        equity_curve[t] = max(0.0, current_equity)

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

    return final_eq, mult, max_dd, sharpe, len(trades), wr

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")

    print("--- TESTING 1H VOLUME-CONFIRMED ASYMMETRIC R:R FIXED-RISK ---")
    for r in [0.005, 0.01, 0.015, 0.02, 0.025]:
        for stop_m in [1.0, 1.2, 1.5, 2.0]:
            eq, mult, dd, sh, n, wr = test_1h_asymmetric(bundle, risk_pct=r, stop_atr_mult=stop_m)
            print(f"Risk={r:.1%} | Stop={stop_m:.1f}ATR => Final=${eq:,.2f} ({mult:6.2f}x) | Max DD={dd:6.2%} | Sharpe={sh:4.2f} | Trades={n:4d} | WR={wr:4.1%}")

if __name__ == "__main__":
    main()
