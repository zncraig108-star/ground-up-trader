"""scratch/inspect_peak.py — Inspect portfolio holdings on 2024-03-15 peak."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

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
    peak_date = pd.Timestamp("2024-03-15 02:00:00")
    print(f"Equity at peak ({peak_date}): ${eq.loc[peak_date]:,.2f}")

    # Inspect trades active around peak_date
    for t in res.trades:
        if t.entry_time <= peak_date and t.exit_time >= peak_date:
            print(f"Active Trade: {t.product_id} side={t.side} entry={t.entry_time} exit={t.exit_time} entry_px={t.entry_px:.4f} exit_px={t.exit_px:.4f} pnl=${t.pnl:+,.2f} exit_reason={t.exit_reason}")

if __name__ == "__main__":
    main()
