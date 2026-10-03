"""scratch/test_aggressive_tier_compounding.py — Test Aggressive Integer Sizing with Floor Protection.

Investigate compounding tier steps: $40, $50, $75, $100, $150, $250.
Enforce invariant:
Total stop loss of all active positions must NEVER allow account to drop below active floor!
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import MicroFuturesConfig
from core.portfolio_data import load_portfolio_data
from core.micro_futures_strategy import compute_micro_futures_signals
from core.micro_futures_backtester import backtest_micro_futures, MICRO_CONTRACT_SIZES, MARGIN_RATES

def test_tier_steps():
    b = load_portfolio_data(MicroFuturesConfig(), data_dir="data")
    
    print("=========================================================================================")
    print("TESTING AGGRESSIVE TIER COMPOUNDING ON 1H NYS ENGINE ($500 START)")
    print("=========================================================================================\n")

    for tier_step in [30.0, 40.0, 50.0, 60.0, 75.0, 100.0, 150.0, 200.0, 250.0]:
        for exit_d in [4, 10, 20]:
            cfg = MicroFuturesConfig(
                signal_timeframe_hours=1,
                compounding_tier_step=tier_step,
                exit_lookback_days=exit_d,
            )
            sig = compute_micro_futures_signals(b.market_data_1h, cfg)
            res = backtest_micro_futures(sig, cfg)
            mult = res.final_equity / 500.0
            print(f"Step=${tier_step:5.1f} | Exit={exit_d:2d}d => Final=${res.final_equity:10,.2f} ({mult:6.2f}x) | MaxDD={res.max_peak_drawdown:5.2%} | DepositDD={res.deposit_drawdown:5.2%} (MinEq=${res.min_equity_ever:6.2f}) | Trades={len(res.trades):3d} | WR={res.win_rate:4.1%} | IS={res.is_mult:5.2f}x | OOS={res.oos_mult:5.2f}x")

if __name__ == "__main__":
    test_tier_steps()
