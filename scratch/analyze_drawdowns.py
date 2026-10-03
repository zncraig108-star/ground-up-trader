"""scratch/analyze_drawdowns.py — Inspect all drawdowns of the 100x baseline."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals
from core.portfolio_backtester import backtest_portfolio

def main():
    cfg = PortfolioConfig()
    bundle = load_portfolio_data(cfg, data_dir="data")
    signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, cfg)
    res = backtest_portfolio(signals, cfg)

    eq = pd.Series(res.equity_curve, index=res.timestamps)
    peaks = eq.cummax()
    dds = (peaks - eq) / peaks

    # Find distinct drawdown episodes (> 20%)
    # Print drawdown episodes
    in_dd = False
    dd_start = None
    max_d = 0.0
    episodes = []

    for t, dd in dds.items():
        if dd > 0.15 and not in_dd:
            in_dd = True
            dd_start = t
            max_d = dd
        elif in_dd:
            if dd > max_d:
                max_d = dd
            if dd < 0.05:
                in_dd = False
                episodes.append((dd_start, t, max_d, peaks[dd_start]))
                max_d = 0.0

    if in_dd:
        episodes.append((dd_start, eq.index[-1], max_d, peaks[dd_start]))

    print(f"Total drawdown episodes > 15%: {len(episodes)}")
    for start, end, depth, peak_val in episodes:
        print(f"  {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} | Max DD = {depth:6.2%} | Peak Equity = ${peak_val:10,.2f}")

if __name__ == "__main__":
    main()
