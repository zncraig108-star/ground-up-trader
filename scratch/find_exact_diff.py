"""scratch/find_exact_diff.py — Find the exact difference between scratch and backtester.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from config import MicroFuturesConfig, PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.micro_futures_strategy import compute_micro_futures_signals
from core.micro_futures_backtester import backtest_micro_futures, MICRO_CONTRACT_SIZES, MARGIN_RATES

p_cfg = PortfolioConfig()
bundle = load_portfolio_data(p_cfg, data_dir="data")

# In scratch/test_momentum_ranking.py:
# Look at line 147:
# if (active_floor == 0 or eq > active_floor) and (eq >= 450.0):
# Wait! In core/micro_futures_backtester.py:
# Line 317: can_enter = (active_floor == 0 or current_equity > active_floor) and (current_equity >= soft_floor)
# Wait! In scratch/test_momentum_ranking.py:
# Look at line 178:
# notional = cnt_target * mult * px
# margin = notional * MARGIN_RATES[p]
# if margin <= eq * 0.45 and (notional / eq) <= 2.8:
# In core/micro_futures_backtester.py:
# cfg.max_portfolio_leverage is in config.py! What is cfg.max_portfolio_leverage?
cfg = MicroFuturesConfig()
print("cfg.max_portfolio_leverage =", cfg.max_portfolio_leverage)
print("cfg.compounding_tier_step =", cfg.compounding_tier_step)
print("cfg.be_lock_enabled =", cfg.be_lock_enabled)
print("cfg.be_lock_mult_atr =", cfg.be_lock_mult_atr)
print("cfg.short_er_boost =", cfg.short_er_boost)
print("cfg.atr_stop_mult =", cfg.atr_stop_mult)

# What about the cooldown?
# In core/micro_futures_backtester.py:
# if p == last_stop_asset and (t - last_stop_bar) < (24 // tf_hours): continue
# If tf_hours = 1, 24 // 1 = 24 bars cooldown!
# Did scratch have cooldown?
# In scratch/test_momentum_ranking.py:
# There was NO cooldown check!
