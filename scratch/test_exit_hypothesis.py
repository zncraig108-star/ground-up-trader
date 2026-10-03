"""scratch/test_exit_hypothesis.py — Testing shorter exit lookbacks and tight trailing stops."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals
from core.portfolio_backtester import backtest_portfolio

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")

    # Test various exit lookbacks (in 4h bars: 12=2d, 18=3d, 24=4d, 36=6d, 72=12d)
    # and various leverage configurations
    print("Testing exit_lookback_bars and leverage settings:")
    for exit_bars in [12, 18, 24, 30, 36]:
        for t1, t2, t3 in [(0.70, 0.40, 0.30), (0.80, 0.50, 0.40), (1.00, 0.60, 0.40)]:
            cfg = PortfolioConfig(
                exit_lookback_bars=exit_bars,
                tranche_1_leverage=t1,
                tranche_2_leverage=t2,
                tranche_3_leverage=t3,
            )
            signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, cfg)
            res = backtest_portfolio(signals, cfg)
            mult = res.final_equity / 10000.0
            print(f"Exit={exit_bars*4}h ({exit_bars*4//24}d) | Lev=({t1:.2f},{t2:.2f},{t3:.2f}) => Final=${res.final_equity:,.2f} ({mult:.2f}x) | Max DD={res.max_drawdown:.2%} | Sharpe={res.sharpe:.2f}")

if __name__ == "__main__":
    main()
