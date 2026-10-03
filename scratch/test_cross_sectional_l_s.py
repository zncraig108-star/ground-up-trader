"""scratch/test_cross_sectional_l_s.py — Systematic Cross-Sectional Long/Short Spread Momentum.

Tests market-neutral and beta-hedged cross-sectional relative strength momentum across
the 9 Coinbase perp assets (BTC, ETH, SOL, AVAX, DOGE, LINK, SUI, NEAR, XRP).
Explores whether hedging market direction eliminates the 80%+ directional drawdowns
and allows safe compounding toward the 100x target.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from config import PortfolioConfig, DEFAULT_PORTFOLIO_INSTRUMENTS, InstrumentSpec
from core.portfolio_data import load_portfolio_data

def run_spread_experiment():
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

    # 1. Volatility (ATR 14 on 4h)
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    # Test different lookback horizons for Relative Strength ranking
    for lookback_days in [3, 5, 7, 10, 14, 21]:
        bars = lookback_days * 6
        roc = (c_4h / c_4h.shift(bars) - 1.0).shift(1)

        for lev in [0.5, 0.8, 1.0, 1.2, 1.5]:
            equity = 10000.0
            eq_curve = []
            
            # Position holding: Long top 2, Short bottom 2
            # Rebalance every 24h (6 bars) or every bar
            rebal_bars = 6
            weights = {p: 0.0 for p in universe}

            for t in range(bars + 1, len(c_4h)):
                # Daily rebalance of target weights
                if t % rebal_bars == 0:
                    r_slice = roc.iloc[t].dropna()
                    if len(r_slice) >= 6:
                        ranked = r_slice.sort_values(ascending=False)
                        top_2 = ranked.index[:2]
                        bot_2 = ranked.index[-2:]

                        # Volatility parity inverse-ATR weights
                        inv_atr = {}
                        for p in list(top_2) + list(bot_2):
                            a = atr_df[p].iloc[t]
                            px = c_4h[p].iloc[t]
                            norm_vol = (a / px) if px > 0 and not np.isnan(a) and a > 0 else 0.05
                            inv_atr[p] = 1.0 / max(norm_vol, 0.01)

                        sum_top = sum(inv_atr[p] for p in top_2)
                        sum_bot = sum(inv_atr[p] for p in bot_2)

                        new_w = {p: 0.0 for p in universe}
                        for p in top_2:
                            new_w[p] = (inv_atr[p] / sum_top) * (lev * 0.5)
                        for p in bot_2:
                            new_w[p] = -(inv_atr[p] / sum_bot) * (lev * 0.5)
                        weights = new_w

                # 4h return update
                bar_ret = sum(weights[p] * (c_4h[p].iloc[t] / c_4h[p].iloc[t-1] - 1.0) for p in universe)
                equity *= (1.0 + bar_ret)
                eq_curve.append(equity)

            eq_arr = np.array(eq_curve)
            peaks = np.maximum.accumulate(eq_arr)
            dds = (peaks - eq_arr) / peaks
            max_dd = np.max(dds)
            mult = eq_arr[-1] / 10000.0
            
            rets = pd.Series(eq_arr).pct_change().dropna()
            sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(2190)

            # Split IS / OOS
            split = len(eq_arr) // 2
            is_arr = eq_arr[:split]
            oos_arr = eq_arr[split:]
            is_mult = is_arr[-1] / 10000.0
            oos_mult = oos_arr[-1] / is_arr[-1]
            oos_p = np.maximum.accumulate(oos_arr)
            oos_dd = np.max((oos_p - oos_arr) / oos_p)

            if mult > 5.0 or max_dd < 0.20:
                print(f"Lookback={lookback_days:2d}d | Lev={lev:.1f}x => Mult={mult:6.2f}x | MaxDD={max_dd:5.2%} | Sharpe={sharpe:4.2f} | IS={is_mult:4.2f}x | OOS={oos_mult:4.2f}x (OOS DD={oos_dd:5.2%})")

if __name__ == "__main__":
    run_spread_experiment()
