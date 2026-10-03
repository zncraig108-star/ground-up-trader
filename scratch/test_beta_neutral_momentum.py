"""scratch/test_beta_neutral_momentum.py — Test Beta-Neutral Long/Short Momentum."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data

def compute_rolling_betas(market_data_1h, universe, lookback_bars=720):
    """Compute rolling 30-day (720h) beta of each asset to BTC-PERP."""
    btc_c = market_data_1h["BTC-PERP"]["close"]
    btc_ret = btc_c.pct_change().fillna(0.0)

    betas = {}
    for p in universe:
        if p == "BTC-PERP":
            betas[p] = pd.Series(1.0, index=btc_c.index)
            continue
        c = market_data_1h[p]["close"]
        r = c.pct_change().fillna(0.0)
        cov = r.rolling(lookback_bars, min_periods=120).cov(btc_ret)
        var = btc_ret.rolling(lookback_bars, min_periods=120).var()
        b = (cov / np.maximum(var, 1e-8)).fillna(1.0).clip(0.2, 3.0)
        betas[p] = b

    return betas

def main():
    cfg = PortfolioConfig()
    bundle = load_portfolio_data(cfg, data_dir="data")
    universe = cfg.universe
    common_idx = bundle.market_data_1h["BTC-PERP"].index

    print("Computing rolling betas to BTC...")
    betas = compute_rolling_betas(bundle.market_data_1h, universe)

    # 30-day Momentum (720h)
    rocs = {}
    for p in universe:
        c = bundle.market_data_1h[p]["close"]
        rocs[p] = (c / c.shift(720) - 1.0).fillna(-999.0)

    roc_df = pd.DataFrame(rocs, index=common_idx)
    beta_df = pd.DataFrame(betas, index=common_idx)

    # Simulation: Every 4h, Long Top 2 Altcoin Leaders, Short BTC to achieve Net Beta = 0
    # Long weight per leader = L. Short BTC weight = - (L * beta_1 + L * beta_2)
    # Total Gross Leverage = 2 * L + |w_btc|
    print("\nSimulating Beta-Neutral Long/Short Momentum Portfolio:")
    for long_weight in [0.20, 0.30, 0.40, 0.50, 0.70, 1.00]:
        cash = 10000.0
        equity = 10000.0
        n_bars = len(common_idx)
        equity_curve = np.zeros(n_bars)

        # Positions in notional
        # We track cash and simple hourly returns
        current_longs = []
        long_qtys = {}
        btc_short_qty = 0.0

        for t in range(720, n_bars):
            ts = common_idx[t]
            btc_px = bundle.market_data_1h["BTC-PERP"]["close"].iloc[t]

            # Rebalance every 24 hours (daily) or 4 hours
            if t % 24 == 0:
                # Rank altcoins (exclude BTC)
                alt_universe = [p for p in universe if p != "BTC-PERP"]
                alt_rocs = {p: roc_df[p].iloc[t-1] for p in alt_universe}
                # Sort descending
                ranked_alts = sorted(alt_universe, key=lambda x: alt_rocs[x], reverse=True)
                top_2 = ranked_alts[:2]

                # Target notional per leader
                leader_notional = equity * long_weight
                # Beta hedge
                b1 = beta_df[top_2[0]].iloc[t-1]
                b2 = beta_df[top_2[1]].iloc[t-1]
                btc_hedge_notional = leader_notional * (b1 + b2)

                # Set positions
                current_longs = top_2
                long_qtys = {
                    p: leader_notional / bundle.market_data_1h[p]["close"].iloc[t]
                    for p in top_2
                }
                btc_short_qty = btc_hedge_notional / btc_px

            # Compute portfolio hourly return
            if t > 720 and len(current_longs) == 2:
                # Hourly ret
                pnl = 0.0
                for p in current_longs:
                    px_curr = bundle.market_data_1h[p]["close"].iloc[t]
                    px_prev = bundle.market_data_1h[p]["close"].iloc[t-1]
                    pnl += long_qtys[p] * (px_curr - px_prev)

                btc_curr = btc_px
                btc_prev = bundle.market_data_1h["BTC-PERP"]["close"].iloc[t-1]
                pnl -= btc_short_qty * (btc_curr - btc_prev)

                # Funding drag approximation (0.01% per 8h ~ 0.00125% per h)
                funding_cost = (abs(btc_short_qty * btc_curr) + sum(abs(long_qtys[p] * bundle.market_data_1h[p]["close"].iloc[t]) for p in current_longs)) * 0.00001
                equity += (pnl - funding_cost)

            equity_curve[t] = max(0.0, equity)

        # Performance metrics
        eq_active = equity_curve[720:]
        peaks = np.maximum.accumulate(eq_active)
        dds = (peaks - eq_active) / np.maximum(peaks, 1e-6)
        max_dd = np.max(dds)
        final_eq = eq_active[-1]
        mult = final_eq / 10000.0

        rets = pd.Series(eq_active).pct_change().fillna(0.0)
        sharpe = (rets.mean() / (rets.std() + 1e-8)) * math.sqrt(8760.0)

        print(f"Long Weight={long_weight:.2f} => Final=${final_eq:,.2f} ({mult:6.2f}x) | Max DD={max_dd:6.2%} | Sharpe={sharpe:4.2f}")

if __name__ == "__main__":
    main()
