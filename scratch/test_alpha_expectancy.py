"""scratch/test_alpha_expectancy.py — Compare Expectancy (R-multiples, Win Rate, Profit Factor) across Core Archetypes:
1. Classical Breakout (Buy 20d High)
2. Trend Pullback (Buy Dip to 20 EMA when above 50 EMA)
3. Panic Mean Reversion (Buy when 4h RSI < 25 / Bollinger oversold)
4. Relative Strength Rotation (Buy top 10% relative momentum on pullback)
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data

def evaluate_setups():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")
    universe = list(bundle.market_data_1h.keys())
    common_idx = bundle.market_data_1h["BTC-PERP"].index

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()

    # 4h Resampling
    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    # Indicators on 4h
    btc_sma200 = c_4h["BTC-PERP"].rolling(1200, min_periods=120).mean().shift(1)
    is_bull_4h = (c_4h["BTC-PERP"] > btc_sma200).shift(1).fillna(False)

    # 1. 20-day High (120 4h bars)
    hi_20d = h_4h.rolling(120, min_periods=30).max().shift(1)

    # 2. EMAs: 20 and 50
    ema20 = c_4h.ewm(span=20).mean().shift(1)
    ema50 = c_4h.ewm(span=50).mean().shift(1)

    # 3. RSI 14
    delta = c_4h.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / np.maximum(loss, 1e-8)
    rsi_4h = (100.0 - (100.0 / (1.0 + rs))).shift(1)

    # 4. ATR 14
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14).mean().shift(1)
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    # Forward 12-bar (48h) return and forward 24-bar (96h) max favorable / adverse excursion
    fwd_ret_48h = (c_4h.shift(-12) / c_4h - 1.0)
    
    print("=== EXPECTANCY ANALYSIS ACROSS ARCHETYPES (4h BARS, ZERO LOOKAHEAD) ===")
    
    # Archetype 1: Donchian 20d Breakout in Bull Market
    b1_mask = (c_4h > hi_20d) & is_bull_4h.values[:, None]
    b1_rets = fwd_ret_48h[b1_mask].dropna().values.flatten()
    print(f"\n1. Donchian 20d Breakout (Bull):")
    print(f"   Signals: {len(b1_rets)} | Win Rate (48h > 0): {np.mean(b1_rets > 0):.1%} | Mean Ret: {np.mean(b1_rets):+.2%} | Win/Loss Ratio: {abs(np.mean(b1_rets[b1_rets > 0]) / min(np.mean(b1_rets[b1_rets <= 0]), -1e-6)):.2f}")

    # Archetype 2: Trend Pullback (Price touches 20 EMA while 20 EMA > 50 EMA in Bull Market)
    b2_mask = (l_4h <= ema20) & (c_4h > ema20) & (ema20 > ema50) & is_bull_4h.values[:, None]
    b2_rets = fwd_ret_48h[b2_mask].dropna().values.flatten()
    print(f"\n2. Trend Pullback to 20 EMA (Bull):")
    print(f"   Signals: {len(b2_rets)} | Win Rate (48h > 0): {np.mean(b2_rets > 0):.1%} | Mean Ret: {np.mean(b2_rets):+.2%} | Win/Loss Ratio: {abs(np.mean(b2_rets[b2_rets > 0]) / min(np.mean(b2_rets[b2_rets <= 0]), -1e-6)):.2f}")

    # Archetype 3: Deep Panic Reversion (RSI < 25 in Bull Market - 'Buy Fear')
    b3_mask = (rsi_4h < 25) & is_bull_4h.values[:, None]
    b3_rets = fwd_ret_48h[b3_mask].dropna().values.flatten()
    print(f"\n3. Panic Dip Reversion (RSI < 25 in Bull Market):")
    print(f"   Signals: {len(b3_rets)} | Win Rate (48h > 0): {np.mean(b3_rets > 0):.1%} | Mean Ret: {np.mean(b3_rets):+.2%} | Win/Loss Ratio: {abs(np.mean(b3_rets[b3_rets > 0]) / min(np.mean(b3_rets[b3_rets <= 0]), -1e-6)):.2f}")

    # Archetype 4: Panic Dip Reversion (RSI < 20 ANY Market - Extreme Capitulation)
    b4_mask = (rsi_4h < 20)
    b4_rets = fwd_ret_48h[b4_mask].dropna().values.flatten()
    print(f"\n4. Extreme Capitulation Reversion (RSI < 20 All Regimes):")
    print(f"   Signals: {len(b4_rets)} | Win Rate (48h > 0): {np.mean(b4_rets > 0):.1%} | Mean Ret: {np.mean(b4_rets):+.2%} | Win/Loss Ratio: {abs(np.mean(b4_rets[b4_rets > 0]) / min(np.mean(b4_rets[b4_rets <= 0]), -1e-6)):.2f}")

if __name__ == "__main__":
    evaluate_setups()
