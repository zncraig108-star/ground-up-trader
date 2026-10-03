"""run_micro_comparison.py — Direct Comparison of 4h Macro vs 1h Intraday Micro Futures Engines.

Compares:
1. Engine A: 4h Macro Signals (4h breakout + 50d EMA + 1h stop execution)
2. Engine B: 1h Intraday Signals (1h breakout + 50d EMA + 1h stop execution)

Both run on real 5.25-year Coinbase hourly data with:
- $500 starting capital
- Whole integer contracts (MET, SOL, BTI)
- $450 soft floor with defensive posture
- $500 / $1k / $2k ratchets
- Shorting enabled
- 5% APY USDC yield on idle cash
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from config import MicroFuturesConfig, PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.micro_futures_strategy import compute_micro_futures_signals
from core.micro_futures_backtester import backtest_micro_futures

def main():
    print("===========================================================================")
    print("NYS REGULATED MICRO FUTURES COMPARISON ($500 INITIAL CAPITAL)")
    print("Tradable Universe: ETH Micro (MET), SOL Micro, BTC Nano (BTI)")
    print("Soft Floor: $450 | Ratchets: $500, $1,000, $2,000 | Non-Fractional Contracts")
    print("===========================================================================\n")

    p_cfg = PortfolioConfig()
    bundle = load_portfolio_data(p_cfg, data_dir="data")

    # 1. Test 4h Macro Engine
    print("--- [1] RUNNING 4-HOUR MACRO ENGINE ---")
    cfg_4h = MicroFuturesConfig(signal_timeframe_hours=4)
    sig_4h = compute_micro_futures_signals(bundle.market_data_1h, cfg_4h)
    res_4h = backtest_micro_futures(sig_4h, cfg_4h)

    print(f"  Final Equity:       ${res_4h.final_equity:10,.2f} ({res_4h.final_equity / 500.0:.2f}x)")
    print(f"  Total Return:       {res_4h.total_return:+.2%}")
    print(f"  CAGR:               {res_4h.cagr:+.2%}")
    print(f"  Sharpe Ratio:       {res_4h.sharpe:.2f}")
    print(f"  Sortino Ratio:      {res_4h.sortino:.2f}")
    print(f"  Max Peak DD:        {res_4h.max_peak_drawdown:.2%}")
    print(f"  Deposit DD:         {res_4h.deposit_drawdown:.2%} (Min Equity: ${res_4h.min_equity_ever:.2f} >= $450)")
    print(f"  Total Trades:       {len(res_4h.trades)} (Win Rate: {res_4h.win_rate:.1%})")
    print(f"  Total Fees Paid:    ${res_4h.total_fees:.2f}")
    print(f"  USDC Yield Earned:  ${res_4h.usdc_yield_earned:.2f}")
    print(f"  In-Sample (2021-23):{res_4h.is_mult:.2f}x (Peak DD: {res_4h.is_dd:.1%})")
    print(f"  Out-of-Sample (24+):{res_4h.oos_mult:.2f}x (Peak DD: {res_4h.oos_dd:.1%})\n")

    # 2. Test 1h Intraday Engine
    print("--- [2] RUNNING 1-HOUR INTRADAY ENGINE ---")
    cfg_1h = MicroFuturesConfig(signal_timeframe_hours=1)
    sig_1h = compute_micro_futures_signals(bundle.market_data_1h, cfg_1h)
    res_1h = backtest_micro_futures(sig_1h, cfg_1h)

    print(f"  Final Equity:       ${res_1h.final_equity:10,.2f} ({res_1h.final_equity / 500.0:.2f}x)")
    print(f"  Total Return:       {res_1h.total_return:+.2%}")
    print(f"  CAGR:               {res_1h.cagr:+.2%}")
    print(f"  Sharpe Ratio:       {res_1h.sharpe:.2f}")
    print(f"  Sortino Ratio:      {res_1h.sortino:.2f}")
    print(f"  Max Peak DD:        {res_1h.max_peak_drawdown:.2%}")
    print(f"  Deposit DD:         {res_1h.deposit_drawdown:.2%} (Min Equity: ${res_1h.min_equity_ever:.2f} >= $450)")
    print(f"  Total Trades:       {len(res_1h.trades)} (Win Rate: {res_1h.win_rate:.1%})")
    print(f"  Total Fees Paid:    ${res_1h.total_fees:.2f}")
    print(f"  USDC Yield Earned:  ${res_1h.usdc_yield_earned:.2f}")
    print(f"  In-Sample (2021-23):{res_1h.is_mult:.2f}x (Peak DD: {res_1h.is_dd:.1%})")
    print(f"  Out-of-Sample (24+):{res_1h.oos_mult:.2f}x (Peak DD: {res_1h.oos_dd:.1%})\n")

if __name__ == "__main__":
    main()
