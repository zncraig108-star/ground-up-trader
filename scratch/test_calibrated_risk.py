"""scratch/test_calibrated_risk.py — Test risk calibration and OOS generalization."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals, PortfolioSignals, AssetSignals
from core.portfolio_backtester import backtest_portfolio

def eval_split(signals, cfg):
    ts = signals.timestamps
    is_mask = (ts >= "2021-07-01") & (ts <= "2023-12-31")
    oos_mask = (ts >= "2024-01-01") & (ts <= "2026-10-02")

    def run_sub(mask):
        sub_ts = ts[mask]
        sub_sigs = {}
        for p in cfg.universe:
            asig = signals.asset_signals[p]
            sub_sigs[p] = AssetSignals(
                product_id=p,
                timestamps=sub_ts,
                close=asig.close[mask],
                high=asig.high[mask],
                low=asig.low[mask],
                regime=asig.regime[mask],
                target_weight=asig.target_weight[mask],
                stop_loss_px=asig.stop_loss_px[mask],
                atr=asig.atr[mask],
                funding_rate=asig.funding_rate[mask],
                tranche=asig.tranche[mask],
            )
        psig = PortfolioSignals(universe=cfg.universe, timestamps=sub_ts, asset_signals=sub_sigs)
        return backtest_portfolio(psig, cfg)

    full_res = backtest_portfolio(signals, cfg)
    is_res = run_sub(is_mask)
    oos_res = run_sub(oos_mask)

    return full_res, is_res, oos_res

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")

    # Test leverage scaling
    print("Testing leverage calibrations:")
    for scale in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        cfg = PortfolioConfig(
            tranche_1_leverage=1.00 * scale,
            tranche_2_leverage=0.75 * scale,
            tranche_3_leverage=0.55 * scale,
            bear_short_leverage=0.50 * scale,
        )
        signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, cfg)
        full, ins, oos = eval_split(signals, cfg)

        print(f"Scale {scale:.1f}x (Max Lev={2.30*scale:.2f}x):")
        print(f"  FULL 5Y: Mult={full.final_equity/10000:6.2f}x | Max DD={full.max_drawdown:6.2%} | Sharpe={full.sharpe:4.2f} | Calmar={full.calmar:4.2f}")
        print(f"  TRAIN  : Mult={ins.final_equity/10000:6.2f}x | Max DD={ins.max_drawdown:6.2%} | Sharpe={ins.sharpe:4.2f}")
        print(f"  TEST   : Mult={oos.final_equity/10000:6.2f}x | Max DD={oos.max_drawdown:6.2%} | Sharpe={oos.sharpe:4.2f}")
        print()

if __name__ == "__main__":
    main()
