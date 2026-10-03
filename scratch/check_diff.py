"""scratch/check_diff.py — Find the exact difference.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from config import MicroFuturesConfig, PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.micro_futures_strategy import compute_micro_futures_signals
from core.micro_futures_backtester import backtest_micro_futures

p_cfg = PortfolioConfig()
bundle = load_portfolio_data(p_cfg, data_dir="data")

# Let's test with climax_trim_enabled = False vs True
for trim in [False, True]:
    cfg = MicroFuturesConfig(
        signal_timeframe_hours=1,
        atr_stop_mult=2.0,
        be_lock_enabled=True,
        be_lock_mult_atr=2.0,
        compounding_tier_step=250.0,
        climax_trim_enabled=trim,
    )
    sig = compute_micro_futures_signals(bundle.market_data_1h, cfg)
    res = backtest_micro_futures(sig, cfg)
    print(f"Trim={trim} => Final=${res.final_equity:.2f} ({res.final_equity/500:.2f}x) | MaxDD={res.max_peak_drawdown:.1%} | DepDD={res.deposit_drawdown:.1%} | Trades={len(res.trades)}")
