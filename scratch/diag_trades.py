"""scratch/diag_trades.py — Detailed trade inspection.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from config import MicroFuturesConfig, PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.micro_futures_strategy import compute_micro_futures_signals
from core.micro_futures_backtester import backtest_micro_futures

p_cfg = PortfolioConfig()
bundle = load_portfolio_data(p_cfg, data_dir="data")

cfg_1h = MicroFuturesConfig(signal_timeframe_hours=1)
sig_1h = compute_micro_futures_signals(bundle.market_data_1h, cfg_1h)
res_1h = backtest_micro_futures(sig_1h, cfg_1h)

df = pd.DataFrame([{
    "asset": t.product_id,
    "side": "LONG" if t.side > 0 else "SHORT",
    "entry_time": t.entry_time,
    "exit_time": t.exit_time,
    "pnl": t.pnl,
    "fees": t.fees,
    "reason": t.exit_reason
} for t in res_1h.trades])

print("Total trades:", len(df))
print("\nPnL by Asset:")
print(df.groupby("asset")["pnl"].agg(["count", "sum", "mean"]))

print("\nPnL by Side:")
print(df.groupby("side")["pnl"].agg(["count", "sum", "mean"]))

print("\nPnL by Reason:")
print(df.groupby("reason")["pnl"].agg(["count", "sum", "mean"]))

print("\nPnL by Asset & Side:")
print(df.groupby(["asset", "side"])["pnl"].agg(["count", "sum", "mean"]))
