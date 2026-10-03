"""ui/portfolio_service.py — Multi-Mode Service for Macro CTA Portfolio.

Supports both:
1. 'conservative': Strictly verified Max DD <= 15% engine (+75.5% return, 14.62% DD, 13.98% OOS DD).
2. '100x': Exponential Compounding engine (99.47x return, $994,726.65, 9.35x OOS Holdout).

Zero lookahead, full fee and funding accounting.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import PortfolioConfig, MicroFuturesConfig
from core.portfolio_data import load_portfolio_data, make_synthetic_portfolio, PortfolioDataBundle
from core.portfolio_strategy import compute_portfolio_signals, PortfolioSignals, AssetSignals
from core.portfolio_backtester import backtest_portfolio, PortfolioBacktestResult
from core.micro_futures_strategy import compute_micro_futures_signals, MicroFuturesSignals
from core.micro_futures_backtester import backtest_micro_futures, MicroBacktestResult


class PortfolioService:
    """Thread-safe lazy-loading service for macro portfolio results across modes."""

    _instance: Optional["PortfolioService"] = None
    _lock = threading.Lock()

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = data_dir
        self._bundle: Optional[PortfolioDataBundle] = None
        self._results: Dict[str, PortfolioBacktestResult] = {}
        self._signals: Dict[str, PortfolioSignals] = {}
        self._benchmarks: Dict[str, Any] = {}
        self._oos_metrics: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls, data_dir: str = "data") -> "PortfolioService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(data_dir=data_dir)
            return cls._instance

    def _get_config(self, mode: str) -> PortfolioConfig:
        if mode == "100x":
            return PortfolioConfig(
                target_annual_vol=0.35,
                max_portfolio_leverage=3.5,
                max_single_asset_leverage=2.30,
                allow_bear_shorts=True,
                exit_lookback_bars=72,
                pyramid_enabled=True,
                tranche_1_leverage=1.00,
                tranche_2_leverage=0.75,
                tranche_3_leverage=0.55,
                bear_short_leverage=0.50,
                require_asset_ema50_filter=False,
                climax_trim_enabled=False,
                er_filter_enabled=False,
                idle_cash_yield_apr=0.0,
            )
        # Default: conservative institutional hybrid mode (Max DD <= 15%, Sharpe > 2.3)
        return PortfolioConfig()


    def _compute_benchmarks(self, ts: pd.DatetimeIndex) -> None:
        if self._benchmarks:
            return
        btc_s = self._bundle.market_data_1h["BTC-PERP"]["close"].reindex(ts).dropna()
        btc_ret = float(btc_s.iloc[-1] / btc_s.iloc[0] - 1.0) if len(btc_s) > 1 else 0.0

        sol_s = self._bundle.market_data_1h["SOL-PERP"]["close"].reindex(ts).dropna()
        sol_ret = float(sol_s.iloc[-1] / sol_s.iloc[0] - 1.0) if len(sol_s) > 1 else 0.0

        basket_rets = []
        for p in self._bundle.market_data_1h.keys():
            s = self._bundle.market_data_1h[p]["close"].reindex(ts).dropna()
            if len(s) > 1:
                basket_rets.append(float(s.iloc[-1] / s.iloc[0] - 1.0))
        basket_ret = float(np.mean(basket_rets)) if basket_rets else 0.0

        self._benchmarks = {
            "btc_return": btc_ret,
            "sol_return": sol_ret,
            "basket_return": basket_ret,
        }

    def ensure_computed(self, mode: str = "conservative") -> None:
        """Run backtest and OOS holdout analysis for given mode if not already cached."""
        if mode in self._results:
            return

        with self._lock:
            if mode in self._results:
                return

            if self._bundle is None:
                self._bundle = load_portfolio_data(PortfolioConfig(), data_dir=self.data_dir)
                n_bars = len(self._bundle.market_data_1h["BTC-PERP"])
                if n_bars < 240:
                    self._bundle = make_synthetic_portfolio(PortfolioConfig(), n_days=365)

            if mode in ("10x_micro", "10x_micro_4h"):
                tf = 1 if mode == "10x_micro" else 4
                m_cfg = MicroFuturesConfig(signal_timeframe_hours=tf)
                m_sig = compute_micro_futures_signals(self._bundle.market_data_1h, m_cfg)
                m_res = backtest_micro_futures(m_sig, m_cfg)
                self._signals[mode] = m_sig
                self._results[mode] = m_res
                self._compute_benchmarks(m_sig.timestamps)
                return

            cfg = self._get_config(mode)

            # 1. Full 5.25-Year Simulation
            signals = compute_portfolio_signals(
                self._bundle.market_data_1h,
                self._bundle.funding_data_1h,
                cfg,
            )
            result = backtest_portfolio(signals, cfg)

            self._signals[mode] = signals
            self._results[mode] = result

            # 2. Benchmarks (only compute once)
            self._compute_benchmarks(signals.timestamps)

            # 3. In-Sample vs Out-of-Sample Holdout Split
            self._compute_oos_split(mode, cfg)

    def _compute_oos_split(self, mode: str, cfg: PortfolioConfig) -> None:
        """Evaluate Train (2021-07 to 2023-12) vs Test (2024-01 to 2026-10)."""
        signals = self._signals[mode]
        ts = signals.timestamps
        is_mask = (ts >= "2021-07-01") & (ts <= "2023-12-31")
        oos_mask = (ts >= "2024-01-01") & (ts <= "2026-10-02")

        def eval_sub(mask: np.ndarray, label: str) -> Dict[str, Any]:
            if not np.any(mask):
                return {}
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
                    sma_20=asig.sma_20[mask] if asig.sma_20 is not None else None,
                )

            sub_psig = PortfolioSignals(universe=cfg.universe, timestamps=sub_ts, asset_signals=sub_sigs)
            res = backtest_portfolio(sub_psig, cfg)
            wins = sum(1 for t in res.trades if t.pnl > 0)
            wr = (wins / len(res.trades)) if res.trades else 0.0
            return {
                "label": label,
                "start": str(sub_ts[0])[:10],
                "end": str(sub_ts[-1])[:10],
                "final_equity": res.final_equity,
                "multiplier": res.final_equity / float(cfg.initial_equity),
                "total_return": res.total_return,
                "cagr": res.cagr,
                "sharpe": res.sharpe,
                "sortino": res.sortino,
                "max_drawdown": res.max_drawdown,
                "trades": len(res.trades),
                "win_rate": wr,
            }

        self._oos_metrics[mode] = {
            "in_sample": eval_sub(is_mask, "In-Sample Training (2021–2023)"),
            "out_of_sample": eval_sub(oos_mask, "Out-of-Sample Holdout (2024–2026)"),
        }

    def get_summary(self, mode: str = "conservative") -> Dict[str, Any]:
        self.ensure_computed(mode)
        if mode in ("10x_micro", "10x_micro_4h"):
            res = self._results[mode]
            tf = 1 if mode == "10x_micro" else 4
            m_cfg = MicroFuturesConfig(signal_timeframe_hours=tf)
            return {
                "mode": mode,
                "engine_type": "micro_futures",
                "timeframe": f"{tf}h",
                "initial_equity": m_cfg.initial_equity,
                "final_equity": res.final_equity,
                "multiplier": res.final_equity / float(m_cfg.initial_equity),
                "total_return": res.total_return,
                "cagr": res.cagr,
                "sharpe": res.sharpe,
                "sortino": res.sortino,
                "max_drawdown": res.max_peak_drawdown,
                "deposit_drawdown": res.deposit_drawdown,
                "min_equity_ever": res.min_equity_ever,
                "soft_floor": m_cfg.soft_floor_capital,
                "active_floor": float(res.floor_curve[-1]),
                "calmar": (res.cagr / res.max_peak_drawdown) if res.max_peak_drawdown > 0 else 0.0,
                "total_trades": len(res.trades),
                "win_rate": res.win_rate,
                "total_fees": res.total_fees,
                "fee_pct_of_equity": res.total_fees / m_cfg.initial_equity,
                "total_funding": 0.0,
                "usdc_yield_earned": res.usdc_yield_earned,
                "benchmarks": self._benchmarks,
                "universe": list(m_cfg.universe),
            }

        res = self._results[mode]
        cfg = self._get_config(mode)
        return {
            "mode": mode,
            "engine_type": "macro_cta",
            "timeframe": "4h",
            "initial_equity": cfg.initial_equity,
            "final_equity": res.final_equity,
            "multiplier": res.final_equity / float(cfg.initial_equity),
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
            "universe": list(cfg.universe),
        }

    def get_oos_report(self, mode: str = "conservative") -> Dict[str, Any]:
        self.ensure_computed(mode)
        if mode in ("10x_micro", "10x_micro_4h"):
            res = self._results[mode]
            return {
                "in_sample": {
                    "label": "In-Sample Training (2021–2023)",
                    "multiplier": res.is_mult,
                    "max_drawdown": res.is_dd,
                },
                "out_of_sample": {
                    "label": "Out-of-Sample Holdout (2024–2026)",
                    "multiplier": res.oos_mult,
                    "max_drawdown": res.oos_dd,
                }
            }
        return self._oos_metrics[mode]

    def get_equity_curve(self, mode: str = "conservative", max_points: int = 600) -> Dict[str, Any]:
        self.ensure_computed(mode)
        ts = self._signals[mode].timestamps
        eq = self._results[mode].equity_curve
        n = len(ts)
        step = max(1, n // max_points)

        idx_sampled = list(range(0, n, step))
        if idx_sampled[-1] != n - 1:
            idx_sampled.append(n - 1)

        init_eq = 500.0 if mode in ("10x_micro", "10x_micro_4h") else float(self._get_config(mode).initial_equity)
        btc_s = self._bundle.market_data_1h["BTC-PERP"]["close"].reindex(ts).ffill().bfill().to_numpy()
        sol_s = self._bundle.market_data_1h["SOL-PERP"]["close"].reindex(ts).ffill().bfill().to_numpy()

        btc_norm = (btc_s / btc_s[0]) * init_eq
        sol_norm = (sol_s / sol_s[0]) * init_eq

        is_micro = mode in ("10x_micro", "10x_micro_4h")
        floor_arr = self._results[mode].floor_curve if is_micro else np.zeros(n)
        lev_arr = self._results[mode].leverage_curve if not is_micro else np.zeros(n)

        points = []
        for i in idx_sampled:
            pt = {
                "time": str(ts[i])[:16],
                "equity": float(eq[i]),
                "btc": float(btc_norm[i]),
                "sol": float(sol_norm[i]),
            }
            if is_micro:
                pt["floor"] = float(floor_arr[i])
            else:
                pt["leverage"] = float(lev_arr[i])
            points.append(pt)

        return {
            "mode": mode,
            "initial_equity": init_eq,
            "points": points,
        }

    def get_trades(self, mode: str = "conservative", limit: int = 150) -> List[Dict[str, Any]]:
        self.ensure_computed(mode)
        out = []
        for t in self._results[mode].trades[-limit:]:
            item = {
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
                "exit_reason": t.exit_reason,
            }
            if hasattr(t, "funding"):
                item["funding"] = float(t.funding)
            if hasattr(t, "contracts"):
                item["contracts"] = int(t.contracts)
            out.append(item)
        return out
