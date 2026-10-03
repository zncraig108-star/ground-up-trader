"""scratch/analyze_shorts.py — Check performance of bear shorts vs long-only."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals
from core.portfolio_backtester import backtest_portfolio

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")

    # Compare allow_bear_shorts = True vs False
    for allow_shorts in [True, False]:
        cfg = PortfolioConfig(allow_bear_shorts=allow_shorts)
        signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, cfg)
        res = backtest_portfolio(signals, cfg)
        mult = res.final_equity / 10000.0

        short_trades = [t for t in res.trades if t.side == -1]
        long_trades = [t for t in res.trades if t.side == 1]
        short_pnl = sum(t.pnl for t in short_trades)
        long_pnl = sum(t.pnl for t in long_trades)

        print(f"Allow Shorts = {allow_shorts}:")
        print(f"  Final Equity = ${res.final_equity:,.2f} ({mult:.2f}x) | Max DD = {res.max_drawdown:.2%} | Sharpe = {res.sharpe:.2f}")
        print(f"  Short Trades = {len(short_trades)} | Net Short PnL = ${short_pnl:+,.2f}")
        print(f"  Long Trades  = {len(long_trades)} | Net Long PnL  = ${long_pnl:+,.2f}")
        print()

if __name__ == "__main__":
    main()
