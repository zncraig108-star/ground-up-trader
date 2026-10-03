"""scratch/check_cooldown.py — Test cooldown impact.
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

# Let's see: in core/micro_futures_backtester.py:
# Line 345:
# if p == last_stop_asset and (t - last_stop_bar) < (24 // tf_hours):
#     continue
# What if cooldown is 0?
