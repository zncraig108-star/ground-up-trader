"""scratch/test_low_dd_oos.py — Check OOS for Lev=0.12 and 0.14."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals, PortfolioSignals, AssetSignals
from core.portfolio_backtester import backtest_portfolio

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")

    for lev, exit_b in [(0.12, 30), (0.13, 30), (0.14, 30)]:
        cfg = PortfolioConfig(
            tranche_1_leverage=lev,
            tranche_2_leverage=0.0,
            tranche_3_leverage=0.0,
            pyramid_enabled=False,
            allow_bear_shorts=False,
            exit_lookback_bars=exit_b,
        )
        signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, cfg)

        ts = signals.timestamps
        is_mask = (ts >= "2021-07-01") & (ts <= "2023-12-31")
        oos_mask = (ts >= "2024-01-01") & (ts <= "2026-10-02")

        def run_sub(mask):
            sub_ts = ts[mask]
            sub_sigs = {}
            for p in cfg.universe:
                asig = signals.asset_signals[p]
                sub_sigs[p] = AssetSignals(
                    product_id=p, timestamps=sub_ts,
                    close=asig.close[mask], high=asig.high[mask], low=asig.low[mask],
                    regime=asig.regime[mask], target_weight=asig.target_weight[mask],
                    stop_loss_px=asig.stop_loss_px[mask], atr=asig.atr[mask],
                    funding_rate=asig.funding_rate[mask], tranche=asig.tranche[mask],
                )
            psig = PortfolioSignals(universe=cfg.universe, timestamps=sub_ts, asset_signals=sub_sigs)
            return backtest_portfolio(psig, cfg)

        full = backtest_portfolio(signals, cfg)
        ins = run_sub(is_mask)
        oos = run_sub(oos_mask)

        print(f"Lev={lev:.2f} Exit={exit_b*4}h:")
        print(f"  FULL 5Y: Mult={full.final_equity/10000:.2f}x (${full.final_equity:,.0f}) | Max DD={full.max_drawdown:.2%} | Sharpe={full.sharpe:.2f}")
        print(f"  TRAIN  : Mult={ins.final_equity/10000:.2f}x | Max DD={ins.max_drawdown:.2%} | Sharpe={ins.sharpe:.2f}")
        print(f"  TEST   : Mult={oos.final_equity/10000:.2f}x | Max DD={oos.max_drawdown:.2%} | Sharpe={oos.sharpe:.2f}")
        print()

if __name__ == "__main__":
    main()
