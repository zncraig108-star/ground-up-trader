"""scratch/test_dd_reduction.py — Analyze trades during the March-Oct 2024 drawdown."""

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

    # Filter trades exited between 2024-03-15 and 2024-10-26
    dd_trades = [t for t in res.trades if t.exit_time >= pd.Timestamp("2024-03-15") and t.exit_time <= pd.Timestamp("2024-10-26")]
    print(f"Number of trades exited during drawdown: {len(dd_trades)}")
    total_loss = sum(t.pnl for t in dd_trades)
    print(f"Total Net PnL of these trades: ${total_loss:,.2f}")
    for t in dd_trades:
        print(f"  {t.product_id} side={t.side:2d} entry={t.entry_time} exit={t.exit_time} pnl=${t.pnl:+10.2f} reason={t.exit_reason}")

if __name__ == "__main__":
    main()
