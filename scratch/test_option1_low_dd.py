"""scratch/test_option1_low_dd.py — Fine-tune Option 1 (Conservative Low-DD Trend Following)."""

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
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")

    print("--- EVALUATING CONSERVATIVE LOW-DRAWDOWN DESIGNS ---")
    for lev in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        for allow_shorts in [False, True]:
            for exit_b in [24, 36, 48, 72]:
                cfg = PortfolioConfig(
                    tranche_1_leverage=lev,
                    tranche_2_leverage=0.0,
                    tranche_3_leverage=0.0,
                    pyramid_enabled=False,
                    allow_bear_shorts=allow_shorts,
                    exit_lookback_bars=exit_b,
                )
                signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, cfg)
                res = backtest_portfolio(signals, cfg)
                mult = res.final_equity / 10000.0

                if res.max_drawdown <= 0.35:
                    print(f"Lev={lev:.2f} | Shorts={str(allow_shorts):<5} | Exit={exit_b*4:3d}h => Mult={mult:5.2f}x (${res.final_equity:,.0f}) | Max DD={res.max_drawdown:5.2%} | Sharpe={res.sharpe:4.2f} | Calmar={res.calmar:4.2f}")

if __name__ == "__main__":
    main()
