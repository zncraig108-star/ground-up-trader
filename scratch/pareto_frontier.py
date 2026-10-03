"""scratch/pareto_frontier.py — Map the Pareto Frontier of Return vs Max Drawdown."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals
from core.portfolio_backtester import backtest_portfolio

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")

    # Grid over:
    # 1. Base leverage per leader (t1): 0.15 to 1.00
    # 2. Pyramiding additions (t2, t3): 0.0 to 0.50
    # 3. Trailing exit lookback (exit_lookback_bars): 12, 24, 36, 48, 72
    # 4. ATR stop multiplier: 1.5, 2.0, 2.5, 3.5

    results = []

    print("Running Pareto Frontier Search...")
    for t1 in [0.20, 0.35, 0.50, 0.70, 1.00]:
        for pyramid in [False, True]:
            t2 = 0.5 * t1 if pyramid else 0.0
            t3 = 0.3 * t1 if pyramid else 0.0
            for exit_b in [18, 24, 36, 48, 72]:
                for stop_m in [2.0, 2.5, 3.5]:
                    cfg = PortfolioConfig(
                        tranche_1_leverage=t1,
                        tranche_2_leverage=t2,
                        tranche_3_leverage=t3,
                        pyramid_enabled=pyramid,
                        exit_lookback_bars=exit_b,
                        atr_stop_mult=stop_m,
                    )
                    signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, cfg)
                    res = backtest_portfolio(signals, cfg)
                    mult = res.final_equity / 10000.0
                    results.append({
                        "t1": t1, "pyramid": pyramid, "exit_b": exit_b, "stop_m": stop_m,
                        "final_eq": res.final_equity, "multiplier": mult,
                        "max_dd": res.max_drawdown, "sharpe": res.sharpe, "calmar": res.calmar
                    })

    df = pd.DataFrame(results)
    df.sort_values(by="multiplier", ascending=False, inplace=True)

    print("\n--- TOP 10 HIGHEST RETURN CONFIGURATIONS ---")
    for _, r in df.head(10).iterrows():
        print(f"Lev={r['t1']:.2f} (pyr={r['pyramid']}) Exit={r['exit_b']*4}h Stop={r['stop_m']:.1f}ATR => Mult={r['multiplier']:6.2f}x | Max DD={r['max_dd']:6.2%} | Sharpe={r['sharpe']:4.2f} | Calmar={r['calmar']:4.2f}")

    print("\n--- TOP 10 LOWEST DRAWDOWN CONFIGURATIONS ---")
    df_low_dd = df.sort_values(by="max_dd", ascending=True)
    for _, r in df_low_dd.head(10).iterrows():
        print(f"Lev={r['t1']:.2f} (pyr={r['pyramid']}) Exit={r['exit_b']*4}h Stop={r['stop_m']:.1f}ATR => Mult={r['multiplier']:6.2f}x | Max DD={r['max_dd']:6.2%} | Sharpe={r['sharpe']:4.2f} | Calmar={r['calmar']:4.2f}")

    print("\n--- CONFIGURATIONS WITH MAX DD <= 25% ---")
    df_25 = df[df["max_dd"] <= 0.25]
    if len(df_25) > 0:
        for _, r in df_25.head(10).iterrows():
            print(f"Lev={r['t1']:.2f} (pyr={r['pyramid']}) Exit={r['exit_b']*4}h Stop={r['stop_m']:.1f}ATR => Mult={r['multiplier']:6.2f}x | Max DD={r['max_dd']:6.2%} | Sharpe={r['sharpe']:4.2f} | Calmar={r['calmar']:4.2f}")
    else:
        print("None found with Max DD <= 25% in this grid.")

    print("\n--- CONFIGURATIONS WITH MAX DD <= 35% ---")
    df_35 = df[df["max_dd"] <= 0.35]
    if len(df_35) > 0:
        for _, r in df_35.head(10).iterrows():
            print(f"Lev={r['t1']:.2f} (pyr={r['pyramid']}) Exit={r['exit_b']*4}h Stop={r['stop_m']:.1f}ATR => Mult={r['multiplier']:6.2f}x | Max DD={r['max_dd']:6.2%} | Sharpe={r['sharpe']:4.2f} | Calmar={r['calmar']:4.2f}")

if __name__ == "__main__":
    main()
