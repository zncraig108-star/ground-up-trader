"""ui/portfolio_service.py — Service layer for Multi-Asset Macro CTA Portfolio.

Computes and caches the full 5.25-year backtest and in-sample vs out-of-sample
holdout evaluations for the interactive web dashboard.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import PortfolioConfig
from core.portfolio_data import load_portfolio_data, make_synthetic_portfolio, PortfolioDataBundle
from core.portfolio_strategy import compute_portfolio_signals, PortfolioSignals
from core.portfolio_backtester import backtest_portfolio, PortfolioBacktestResult


class PortfolioService:
    """Thread-safe lazy-loading service for macro portfolio results."""

    _instance: Optional["PortfolioService"] = None
    _lock = threading.Lock()

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir
        self.cfg = PortfolioConfig()
        self._bundle: Optional[PortfolioDataBundle] = None
        self._signals: Optional[PortfolioSignals] = None
        self._result: Optional[PortfolioBacktestResult] = None
        self._benchmarks: Dict[str, Any] = {}
        self._oos_metrics: Dict[str, Any] = {}
        self._is_computing = False

    @classmethod
    def get_instance(cls, data_dir: str = "data") -> "PortfolioService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(data_dir=data_dir)
            return cls._instance

    def ensure_computed(self) -> None:
        """Run backtest and OOS holdout analysis if not already cached."""
        if self._result is not None:
            return

        with self._lock:
            if self._result is not None:
                return

            self._bundle = load_portfolio_data(self.cfg, data_dir=self.data_dir)
            n_bars = len(self._bundle.market_data_1h[self.cfg.universe[0]])
            if n_bars < 240:
                self._bundle = make_synthetic_portfolio(self.cfg, n_days=365)

            # 1. Full 5.25-Year Simulation
            self._signals = compute_portfolio_signals(
                self._bundle.market_data_1h,
                self._bundle.funding_data_1h,
                self.cfg,
            )
            self._result = backtest_portfolio(self._signals, self.cfg)

            # 2. Benchmarks
            ts = self._signals.timestamps
            btc_s = self._bundle.market_data_1h["BTC-PERP"]["close"].reindex(ts).dropna()
            btc_ret = float(btc_s.iloc[-1] / btc_s.iloc[0] - 1.0) if len(btc_s) > 1 else 0.0

            sol_s = self._bundle.market_data_1h["SOL-PERP"]["close"].reindex(ts).dropna()
            sol_ret = float(sol_s.iloc[-1] / sol_s.iloc[0] - 1.0) if len(sol_s) > 1 else 0.0

            basket_rets = []
            for p in self.cfg.universe:
                s = self._bundle.market_data_1h[p]["close"].reindex(ts).dropna()
                if len(s) > 1:
                    basket_rets.append(float(s.iloc[-1] / s.iloc[0] - 1.0))
            basket_ret = float(np.mean(basket_rets)) if basket_rets else 0.0

            self._benchmarks = {
                "btc_return": btc_ret,
                "sol_return": sol_ret,
                "basket_return": basket_ret,
            }

            # 3. In-Sample vs Out-of-Sample Holdout Split
            self._compute_oos_split()

    def _compute_oos_split(self) -> None:
        """Evaluate Train (2021-07 to 2023-12) vs Test (2024-01 to 2026-10)."""
        ts = self._signals.timestamps
        is_mask = (ts >= "2021-07-01") & (ts <= "2023-12-31")
        oos_mask = (ts >= "2024-01-01") & (ts <= "2026-10-02")

        # In-Sample sub-signals
        def eval_sub(mask: np.ndarray, label: str) -> Dict[str, Any]:
            if not np.any(mask):
                return {}
            sub_ts = ts[mask]
            sub_sigs = {}
            for p in self.cfg.universe:
                asig = self._signals.asset_signals[p]
                from core.portfolio_strategy import AssetSignals
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
            sub_psig = PortfolioSignals(universe=self.cfg.universe, timestamps=sub_ts, asset_signals=sub_sigs)
            res = backtest_portfolio(sub_psig, self.cfg)
            wins = sum(1 for t in res.trades if t.pnl > 0)
            wr = (wins / len(res.trades)) if res.trades else 0.0
            return {
                "label": label,
                "start": str(sub_ts[0])[:10],
                "end": str(sub_ts[-1])[:10],
                "final_equity": res.final_equity,
                "multiplier": res.final_equity / float(self.cfg.initial_equity),
                "total_return": res.total_return,
                "cagr": res.cagr,
                "sharpe": res.sharpe,
                "sortino": res.sortino,
                "max_drawdown": res.max_drawdown,
                "trades": len(res.trades),
                "win_rate": wr,
            }

        self._oos_metrics = {
            "in_sample": eval_sub(is_mask, "In-Sample Training (2021–2023)"),
            "out_of_sample": eval_sub(oos_mask, "Out-of-Sample Holdout (2024–2026)"),
        }

    def get_summary(self) -> Dict[str, Any]:
        self.ensure_computed()
        res = self._result
        return {
            "initial_equity": self.cfg.initial_equity,
            "final_equity": res.final_equity,
            "multiplier": res.final_equity / float(self.cfg.initial_equity),
            "total_return": res.total_return,
            "cagr": res.cagr,
            "sharpe": res.sharpe,
            "sortino": res.sortino,
            "max_drawdown": res.max_drawdown,
            "calmar": res.calmar,
            "total_trades": len(res.trades),
            "total_fees": res.total_fees,
            "fee_pct_of_equity": res.fee_pct_of_equity,
            "total_funding": res.total_funding,
            "benchmarks": self._benchmarks,
            "universe": list(self.cfg.universe),
        }

    def get_oos_report(self) -> Dict[str, Any]:
        self.ensure_computed()
        return self._oos_metrics

    def get_equity_curve(self, max_points: int = 600) -> Dict[str, Any]:
        self.ensure_computed()
        ts = self._signals.timestamps
        eq = self._result.equity_curve
        lev = self._result.leverage_curve
        n = len(ts)
        step = max(1, n // max_points)

        idx_sampled = list(range(0, n, step))
        if idx_sampled[-1] != n - 1:
            idx_sampled.append(n - 1)

        # Baseline curves normalized to initial capital
        btc_s = self._bundle.market_data_1h["BTC-PERP"]["close"].reindex(ts).ffill().bfill().to_numpy()
        sol_s = self._bundle.market_data_1h["SOL-PERP"]["close"].reindex(ts).ffill().bfill().to_numpy()

        btc_norm = (btc_s / btc_s[0]) * float(self.cfg.initial_equity)
        sol_norm = (sol_s / sol_s[0]) * float(self.cfg.initial_equity)

        points = []
        for i in idx_sampled:
            points.append({
                "time": str(ts[i])[:16],
                "equity": float(eq[i]),
                "leverage": float(lev[i]),
                "btc": float(btc_norm[i]),
                "sol": float(sol_norm[i]),
            })

        return {
            "initial_equity": self.cfg.initial_equity,
            "points": points,
        }

    def get_trades(self, limit: int = 150) -> List[Dict[str, Any]]:
        self.ensure_computed()
        out = []
        for t in self._result.trades[-limit:]:
            out.append({
                "product_id": t.product_id,
                "side": "LONG" if t.side > 0 else "SHORT",
                "entry_time": str(t.entry_time)[:16],
                "exit_time": str(t.exit_time)[:16],
                "entry_px": float(t.entry_px),
                "exit_px": float(t.exit_px),
                "qty": float(t.qty),
                "pnl": float(t.pnl),
                "ret_pct": float((t.exit_px / t.entry_px - 1.0) * t.side * 100),
                "fees": float(t.fees),
                "funding": float(t.funding),
                "exit_reason": t.exit_reason,
            })
        return out
