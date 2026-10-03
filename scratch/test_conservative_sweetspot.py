"""scratch/test_conservative_sweetspot.py — Fine-tune the conservative 15-20% DD sweet spot."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals
from core.portfolio_backtester import backtest_portfolio

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")

    print("--- CONSERVATIVE SWEET SPOT GRID (Targeting Max DD <= 20%) ---")
    for lev in [0.12, 0.14, 0.15, 0.16, 0.18, 0.20]:
        for pyr in [False, True]:
            t2 = 0.5 * lev if pyr else 0.0
            t3 = 0.3 * lev if pyr else 0.0
            for exit_b in [24, 30, 36, 42, 48]: # 96h to 192h
                cfg = PortfolioConfig(
                    tranche_1_leverage=lev,
                    tranche_2_leverage=t2,
                    tranche_3_leverage=t3,
                    pyramid_enabled=pyr,
                    allow_bear_shorts=False, # 100% Cash in Bear
                    exit_lookback_bars=exit_b,
                )
                signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, cfg)
                res = backtest_portfolio(signals, cfg)
                mult = res.final_equity / 10000.0

                if res.max_drawdown <= 0.20:
                    print(f"Lev={lev:.2f} (pyr={str(pyr):<5}) Exit={exit_b*4:3d}h => Mult={mult:5.2f}x (${res.final_equity:,.0f}) | Max DD={res.max_drawdown:5.2%} | Sharpe={res.sharpe:4.2f} | Calmar={res.calmar:4.2f}")

if __name__ == "__main__":
    main()
