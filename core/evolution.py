"""core/evolution.py — Genetic Algorithm + Walk-Forward Optimization + deployment gate.

Pipeline
--------
1. **Walk-forward windows** (``build_windows``): sliding ``IS``/``OOS`` hour windows over
   the 1h series. The GA optimizes on the in-sample window; genomes are then scored on
   the untouched out-of-sample window.
2. **GA** (``EvolutionEngine``): tournament selection, uniform crossover, Gaussian
   mutation (continuous genes) + bit-flip (``regime_mode``), elitism (top 5% carried
   unchanged), and clamp back to valid ranges. Fully deterministic under a seed.
3. **Deployment gate**: a genome graduates to the deployment pool only after passing the
   forward-only validation ledger (``ChampionValidationConfig``): non-overlapping periods
   that start *after* the genome's discovery window, with per-period Sharpe/drawdown/trade
   requirements. There is no shortcut from raw in-sample fitness to deployment.
4. **Cross-window consistency + global champion**: every candidate's OOS Sharpe is
   tracked per window. The best genome by mean OOS Sharpe (seen in >= 2 windows) becomes
   the *global champion* and is re-evaluated on **all** windows to produce the
   full-history total-equity curve shown in the UI.
5. **Checkpointing**: population, generation, RNG state, and pool are saved to disk.

``run(n)`` runs at most ``n`` generations total, then stops (``n <= 0`` == run until
cancelled). In walk-forward mode the ``n`` generations are distributed across windows,
cycling if necessary.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, asdict, field, fields, replace
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from config import Config, GAConfig, GENES, ChampionValidationConfig, WFOConfig
from core.backtester import BacktestResult, backtest
from core.ensemble import EnsembleMember, weights_from_members
from core.features import compute_features
from core.metrics import (HOURS_PER_YEAR, compute_metrics, deflated_sharpe_ratio,
                          downside_penalized_score, multi_window_score,
                          probabilistic_sharpe_ratio, probability_of_backtest_overfitting,
                          sharpe_shrinkage)
from core.strategy import (Genome, SignalFeatures, SignalIntents, clamp_genomes,
                           compute_signals, random_population)
from core.xp import asnumpy, get_backend_info, xp

__all__ = [
    "WindowSpec",
    "Snapshot",
    "build_windows",
    "EvolutionEngine",
]

_FLOAT = xp.float64


@dataclass(frozen=True)
class WindowSpec:
    """IS/OOS window boundaries in **1h bar index** units (half-open ``[is_h0, is_h1)``)."""

    is_h0: int
    is_h1: int
    oos_h1: int


def build_windows(n_hours: int, wfo: WFOConfig) -> List[WindowSpec]:
    """Build sliding walk-forward windows over ``n_hours`` 1h bars.

    ``wfo.holdout_days`` trailing days are reserved: no window may see them, so the
    final holdout can be used as a once-only check of the deployed champion.
    """
    is_hours = wfo.is_days * 24
    oos_hours = wfo.oos_days * 24
    step = max(1, wfo.step_days * 24)
    usable = max(0, n_hours - max(0, wfo.holdout_days) * 24)
    windows = []
    for h0 in range(0, usable - is_hours - oos_hours + 1, step):
        windows.append(WindowSpec(is_h0=h0, is_h1=h0 + is_hours, oos_h1=h0 + is_hours + oos_hours))
    return windows


@dataclass
class Snapshot:
    """Per-generation state snapshot, matching DEEPSEEK_BUILD_PROMPT.md §8 payload."""

    generation: int
    best_fitness: float
    wfo_window: Dict[str, str]
    is_equity_curve: List[List[float]]
    oos_equity_curve: List[List[float]]
    top_genomes: List[Dict[str, Any]]
    deployment_pool_size: int
    backend: str
    window_index: int = 0
    n_windows: int = 1
    cycle: int = 0
    aggregate_equity_curve: List[List[float]] = field(default_factory=list)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)
    window_board: List[Dict[str, Any]] = field(default_factory=list)
    slices: List[Dict[str, Any]] = field(default_factory=list)
    gens_per_s: float = 0.0
    champion_id: Optional[str] = None
    champion_score: Optional[float] = None
    champion_status: Optional[str] = None
    champion_validation: Dict[str, Any] = field(default_factory=dict)
    champion_candidates: List[Dict[str, Any]] = field(default_factory=list)
    benchmark_equity_curve: List[List[float]] = field(default_factory=list)
    window_benchmark_equity_curve: List[List[float]] = field(default_factory=list)
    run_status: str = "idle"
    completed_generations: int = 0
    planned_generations: int = 0
    validation_jobs_completed: int = 0
    validation_jobs_total: int = 0
    coverage_metadata: Dict[str, Any] = field(default_factory=dict)
    full_history_benchmark_equity_curve: List[List[float]] = field(default_factory=list)
    ensemble: List[Dict[str, Any]] = field(default_factory=list)
    holdout_report: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


class EvolutionEngine:
    """Runs the GA (single-window or walk-forward) and maintains the deployment pool."""

    def __init__(
        self,
        config: Config,
        ohlcv_1m: pd.DataFrame,
        ohlcv_1h: pd.DataFrame,
        funding: pd.DataFrame,
        instrument: Any,
        window: Optional[WindowSpec] = None,
        seed: Optional[int] = None,
        resume: bool = False,
        walk_forward: bool = False,
    ) -> None:
        self.config = config
        self.ga: GAConfig = config.ga
        self.wfo: WFOConfig = config.wfo
        self.instrument = instrument
        self.ohlcv_1m = ohlcv_1m
        self.ohlcv_1h = ohlcv_1h
        self.funding = funding

        self.seed = config.data.global_seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)

        self.n_hours = len(ohlcv_1h)
        self.hour_ts = ohlcv_1h.index  # naive-UTC DatetimeIndex
        # Scale the forward-validation ladder to the available history: with the
        # configured 30-day periods a short dataset could never qualify a champion.
        # We never lengthen the configured period, only shorten it to fit.
        self.config = replace(
            config, champion_validation=self._scaled_validation_policy())

        feat_set = compute_features(ohlcv_1h, funding)
        tf_mins = 60.0
        if len(ohlcv_1h) >= 2:
            dt_s = (ohlcv_1h.index[1] - ohlcv_1h.index[0]).total_seconds()
            if dt_s > 0:
                tf_mins = dt_s / 60.0
        self.features = SignalFeatures(
            open=ohlcv_1h["open"].to_numpy(),
            high=ohlcv_1h["high"].to_numpy(),
            low=ohlcv_1h["low"].to_numpy(),
            close=ohlcv_1h["close"].to_numpy(),
            volume_zscore=feat_set.volume_zscore,
            funding_zscore=feat_set.funding_zscore,
            timeframe_minutes=tf_mins,
        )

        windows = build_windows(self.n_hours, self.wfo)
        if not windows:
            raise ValueError(
                f"not enough data for WFO: need is_days({self.wfo.is_days}) + "
                f"oos_days({self.wfo.oos_days}) <= {self.n_hours // 24} days"
            )
        self.windows = windows
        self.window = window if window is not None else windows[0]
        self.window_index = 0
        self.walk_forward = walk_forward
        self.cycle = 0
        # Set by the UI/runner: continuously cycle through all windows while serving.
        self.walk_forward_repeat = False
        self.walk_forward_max_cycles = 0  # 0 == unlimited when repeating

        self.population: np.ndarray
        self.generation: int = 0
        self.best_fitness_history: List[float] = []
        # Last generation's in-sample fitness (used to warm-start the next WFO window).
        self._last_is_fitness: Optional[np.ndarray] = None
        self.deployment_pool: List[Dict[str, Any]] = []
        self._pool_keys: set = set()
        # This ledger is intentionally independent of the historical deployment pool.
        # A candidate has no validation evidence until a post-discovery period completes.
        self.candidate_registry: Dict[str, Dict[str, Any]] = {}
        # OOS-tested genomes accumulate here during a window.  Admission happens only
        # after its final generation, so repeated generations cannot flood validation.
        self._window_leaderboard: Dict[str, Dict[str, Any]] = {}
        self._validation_signal_cache: Dict[str, SignalIntents] = {}
        self._validation_curves: Dict[str, Dict[int, List[List[float]]]] = {}
        self._feat_h0: int = 0
        self._feat_h1: int = self.n_hours
        self.window_features: SignalFeatures = self.features
        self._cached_agg_curve: Optional[List[List[float]]] = None
        self._cached_agg_metrics: Optional[Dict[str, Any]] = None
        self._cached_benchmark_curve: Optional[List[List[float]]] = None
        self._cached_slices: Optional[List[Dict[str, Any]]] = None
        self._update_window_features()
        self.run_status = "idle"
        self.completed_generations = 0
        self.planned_generations = 0
        self.validation_jobs_completed = 0
        self.validation_jobs_total = 0
        self._cancelled = False

        # Optional progress sink (set by the UI). ``_emit`` is time-throttled.
        self.progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None
        self._last_emit: float = 0.0
        self.progress_interval_s: float = 0.10  # 10 updates/sec max (prevents UI queue flooding)
        self._progress_every = 0
        self._last_persist: float = 0.0

        # Walk-forward history (for the UI's full-history total-equity chart).
        self.oos_segments: Dict[int, Dict[str, Any]] = {}  # key: window is_h0
        self.window_board: Dict[int, Dict[str, Any]] = {}
        self.consistency: Dict[str, Dict[str, Any]] = {}  # key: genome id
        self._recent_ts: deque = deque(maxlen=100)  # throughput (gens/s)
        # Static chart data (computed once; rebuilt only if data changes).
        self._close_by_ts: Optional[Dict[int, float]] = None
        self._full_history_benchmark: Optional[List[List[float]]] = None
        # Once-only holdout evaluation of the deployed champion (see _evaluate_holdout_once).
        self.holdout_hours = max(0, int(getattr(self.wfo, "holdout_days", 0)) * 24)
        self.holdout_report: Dict[str, Any] = {}
        self._holdout_done = False

        # Global champion (best genome across ALL windows) + its full-history curve.
        # The champion is selected by cross-window OOS robustness (mean OOS Sharpe over
        # the windows it has been scored on), NOT raw in-sample fitness — that avoids
        # crowning an overfit single-window genome. See ``_select_champion``.
        self.champion_genome_id: Optional[str] = None
        self.champion_equity_curve: Optional[List[List[float]]] = None
        self.champion_slices: Dict[int, Dict[str, Any]] = {}
        self.champion_fitness: float = -np.inf
        self.champion_score: float = -np.inf  # champion's mean-OOS-Sharpe score
        self._last_champion_eval: float = 0.0
        # ``_champion_dirty`` forces a re-eval on the next opportunity even if the id is
        # unchanged (e.g. a new window extended the champion's coverage). This fixes the
        # stale-curve bug where a rate-limited skip was never retried.
        self._champion_dirty: bool = True
        # Genome vectors seen (keyed by id) so a champion chosen by id can be re-decoded.
        self._genome_vecs: Dict[str, np.ndarray] = {}
        # Best genome by IS fitness seen so far (the "top genome" — a fallback used only
        # while no genome has yet been scored on OOS).
        self.best_is_genome: Optional[np.ndarray] = None
        self.best_is_fitness: float = -np.inf
        # In-sample curve of the initial window (starting at t=0) used to anchor the spliced walk-forward curve.
        self._initial_is_curve: List[List[float]] = []

        if resume and os.path.exists(self.ga.checkpoint_path):
            self._restore_checkpoint()
        else:
            self.population = random_population(self.rng, self.ga.population_size)

    # ------------------------------------------------------------------ progress
    def _emit(self, phase: str, fraction: Optional[float] = None) -> None:
        """Emit a progress event to ``progress_cb`` (time-throttled; phase edges pass)."""
        if self.progress_cb is None:
            return
        now = time.monotonic()
        is_edge = fraction is None
        if not is_edge and (now - self._last_emit) < self.progress_interval_s:
            return
        self._last_emit = now
        self.progress_cb({
            "generation": self.generation,
            "window_index": self.window_index,
            "n_windows": len(self.windows),
            "cycle": self.cycle,
            "phase": phase,
            "fraction": fraction,
            "run_status": self.run_status,
            "completed_generations": self.completed_generations,
            "planned_generations": self.planned_generations,
            "validation_jobs_completed": self.validation_jobs_completed,
            "validation_jobs_total": self.validation_jobs_total,
        })

    # ------------------------------------------------------------------ windows
    def _scaled_validation_policy(self) -> ChampionValidationConfig:
        """Shrink the forward-validation ladder so short datasets can still qualify.

        Keeps the configured non-overlapping period/step when the history after the
        discovery window is long enough; otherwise halves the window budget down to a
        minimum of one day. Never lengthens the configured period.
        """
        policy = self.config.champion_validation
        history_days = self.n_hours / 24.0
        post = max(1.0, history_days - (self.wfo.is_days + self.wfo.oos_days))
        period = max(1, min(policy.validation_period_days, int(post // 2) or 1))
        step = max(1, min(policy.validation_step_days, period))
        capacity = max(1, int(post // step))
        required = max(1, min(policy.required_periods, capacity))
        return replace(policy, validation_period_days=period,
                       validation_step_days=step, required_periods=required)

    def _window_bounds(self, win: WindowSpec) -> Dict[str, str]:
        return {
            "is_start": str(self.hour_ts[win.is_h0]),
            "is_end": str(self.hour_ts[win.is_h1 - 1]),
            "oos_start": str(self.hour_ts[win.is_h1]),
            "oos_end": str(self.hour_ts[win.oos_h1 - 1]),
        }

    def _update_window_features(self) -> None:
        win = self.window
        warmup = min(240, win.is_h0)
        self._feat_h0 = win.is_h0 - warmup
        self._feat_h1 = win.oos_h1
        h0, h1 = self._feat_h0, self._feat_h1
        self.window_features = SignalFeatures(
            open=self.features.open[h0:h1],
            high=self.features.high[h0:h1],
            low=self.features.low[h0:h1],
            close=self.features.close[h0:h1],
            volume_zscore=self.features.volume_zscore[h0:h1],
            funding_zscore=self.features.funding_zscore[h0:h1],
            timeframe_minutes=self.features.timeframe_minutes,
        )

    def _slice_signals(self, signals: SignalIntents, row_indices: Any, h0: int, h1: int) -> SignalIntents:
        r = xp.asarray(row_indices)
        if signals.regime.shape[1] == self.n_hours:
            rel_h0, rel_h1 = h0, h1
        else:
            offset = getattr(self, "_feat_h0", 0)
            rel_h0, rel_h1 = h0 - offset, h1 - offset
        return SignalIntents(
            regime=signals.regime[r][:, rel_h0:rel_h1],
            atr=signals.atr[r][:, rel_h0:rel_h1],
            long_entry=signals.long_entry[r][:, rel_h0:rel_h1],
            short_entry=signals.short_entry[r][:, rel_h0:rel_h1],
            entry_price=signals.entry_price[r][:, rel_h0:rel_h1],
            is_breakout=signals.is_breakout[r][:, rel_h0:rel_h1] if signals.is_breakout is not None else None,
            breakout_price=signals.breakout_price[r][:, rel_h0:rel_h1] if signals.breakout_price is not None else None,
        )

    def _run_backtest(self, win: WindowSpec, signals: SignalIntents,
                      genome_rows: Any, row_indices: Any) -> BacktestResult:
        is_h0, is_h1 = win.is_h0, win.is_h1
        o1m = self.ohlcv_1m.iloc[is_h0 * 60:is_h1 * 60]
        o1h = self.ohlcv_1h.iloc[is_h0:is_h1]
        fund = self.funding.iloc[is_h0:is_h1]
        sig = self._slice_signals(signals, row_indices, is_h0, is_h1)
        m = (is_h1 - is_h0) * 60
        return backtest(o1m, o1h, fund, sig, genome_rows, self.instrument,
                        initial_equity=self.config.risk.initial_equity,
                        progress_cb=self._bt_progress("is_backtest"),
                        progress_every=max(1, m // 100),
                        use_numba=self.ga.use_numba, fill_model=self.config.fills,
                        **self._breaker_kwargs())

    def _run_oos_backtest(self, win: WindowSpec, signals: SignalIntents,
                          genome_rows: Any, row_indices: Any) -> BacktestResult:
        is_h1, oos_h1 = win.is_h1, win.oos_h1
        o1m = self.ohlcv_1m.iloc[is_h1 * 60:oos_h1 * 60]
        o1h = self.ohlcv_1h.iloc[is_h1:oos_h1]
        fund = self.funding.iloc[is_h1:oos_h1]
        sig = self._slice_signals(signals, row_indices, is_h1, oos_h1)
        m = (oos_h1 - is_h1) * 60
        return backtest(o1m, o1h, fund, sig, genome_rows, self.instrument,
                        initial_equity=self.config.risk.initial_equity,
                        progress_cb=self._bt_progress("oos_backtest"),
                        progress_every=max(1, m // 50),
                        use_numba=self.ga.use_numba, fill_model=self.config.fills,
                        **self._breaker_kwargs())

    def _bt_progress(self, phase: str):
        def _cb(done: int, total: int) -> None:
            self._emit(phase, done / total if total else 1.0)
        return _cb

    def _run_is_slices(self, win: WindowSpec, is_equity_1h: Any, k: int) -> Optional[np.ndarray]:
        """Extract ``k`` contiguous slices from the IS equity curve; return ``(P, k, T)`` returns.

        Used to require cross-sub-window consistency instead of rewarding a single
        lucky 60-day window. Returns ``None`` when slices would be too short to score.
        """
        span = win.is_h1 - win.is_h0
        if k < 2 or span // k < 3:
            return None
        eq = asnumpy(is_equity_1h)
        edges = np.linspace(0, span, k + 1).astype(int)
        rets: List[np.ndarray] = []
        for a, b in zip(edges[:-1], edges[1:]):
            sub = eq[:, a:b]
            rets.append(np.diff(sub, axis=1) / np.maximum(sub[:, :-1], 1e-12))
        t = min(r.shape[1] for r in rets)
        return np.stack([r[:, :t] for r in rets], axis=1)

    def _compute_metrics(self, equity: Any, trades: Any) -> Any:
        """Single metric funnel: robust-fitness shape with trade-count-shrunk inputs.

        Phase-3 accuracy: replace raw Sharpe/Sortino with their trade-count-shrunk
        versions so short, lucky samples cannot win selection. Fees and funding remain
        inside the equity curve (no double counting). Fitness stays finite and is 0 for
        negative-Sharpe strategies.
        """
        m = compute_metrics(equity, HOURS_PER_YEAR, trades,
                            min_trades=self.ga.min_trades,
                            trade_penalty_power=self.ga.trade_penalty_power)
        eq = xp.asarray(equity)
        n_obs = int(eq.shape[1]) if eq.ndim == 2 else 1
        shrunk_sharpe = np.asarray(asnumpy(sharpe_shrinkage(
            asnumpy(m.sharpe), n_obs, asnumpy(m.trades))), dtype=np.float64)
        shrunk_sortino = np.asarray(asnumpy(sharpe_shrinkage(
            asnumpy(m.sortino), n_obs, asnumpy(m.trades))), dtype=np.float64)
        # Lessen drawdown penalty: smooth curve prevents linear cliff that starves trend runners
        dd_val = np.asarray(asnumpy(m.max_drawdown), dtype=np.float64)
        dd_factor = 1.0 / (1.0 + 0.8 * np.clip(dd_val, 0.0, 1.0))
        # Reward capital growth directly so the GA actively pursues monster returns
        ret = np.asarray(asnumpy(m.total_return), dtype=np.float64)
        ret_factor = np.power(1.0 + np.maximum(ret, 0.0), 0.75)
        trade_factor = (np.minimum(1.0, np.asarray(asnumpy(m.trades), dtype=np.float64)
                                   / max(self.ga.min_trades, 1))
                        ** self.ga.trade_penalty_power)
        fitness = (np.sqrt(np.maximum(shrunk_sharpe, 0.0) * np.maximum(shrunk_sortino, 0.0))
                   * ret_factor * dd_factor * trade_factor)
        m.fitness = xp.asarray(np.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0))
        return m

    def _breaker_kwargs(self) -> Dict[str, Any]:
        """Forward the shared risk policy to the backtester when explicitly enabled."""
        if not self.config.risk.enforce_breaker_in_backtest:
            return {}
        return {"breaker_pct": self.config.risk.drawdown_breaker_pct,
                "breaker_window_hours": self.config.risk.breaker_window_hours}

    def _seed_next_window_population(self) -> np.ndarray:
        """Warm-start the next WFO window: carry the previous window's best genomes and
        fill the rest with fresh random draws (knowledge transfer + diversity)."""
        p = self.ga.population_size
        n_carry = min(p, max(1, int(round(p * self.ga.carryover_ratio))))
        fit = self._last_is_fitness
        if fit is None or fit.shape[0] != self.population.shape[0]:
            return random_population(self.rng, p)
        carry = self.population[np.argsort(-fit)[:n_carry]].copy()
        rest = random_population(self.rng, p - n_carry)
        return np.vstack([carry, rest]).astype(np.float64)

    def _jitter_robustness(self, entry: Dict[str, Any]) -> float:
        """Median fitness of parameter-jittered variants divided by the original fitness.

        Robust plateaus score near 1.0; sharp, isolated peaks score far lower and are
        rejected at admission. Deterministic (draws from the engine RNG).
        """
        n = int(self.ga.robustness_jitter)
        base_fit = float(entry.get("oos_fitness", 0.0))
        if n <= 0:
            return 1.0
        if not np.isfinite(base_fit) or base_fit <= 0.0:
            return 0.0
        base_vec = np.asarray(entry["genome"], dtype=np.float64)[None, :]
        ranges = np.asarray([(g.high - g.low) for g in GENES], dtype=np.float64)[None, :]
        fits = []
        for _ in range(n):
            noise = self.rng.normal(0.0, self.ga.jitter_sigma, size=base_vec.shape)
            jittered = clamp_genomes(base_vec + noise * ranges)
            signals = compute_signals(self.window_features, jittered)
            res = self._run_oos_backtest(self.window, signals, jittered, np.arange(1))
            metrics = self._compute_metrics(res.equity_1h, res.num_trades)
            fits.append(float(asnumpy(metrics.fitness)[0]))
        return float(np.clip(float(np.median(fits)) / base_fit, 0.0, 1.0))

    # ------------------------------------------------------------------- GA step
    def step(self) -> Snapshot:
        """Run one generation: IS backtest -> OOS gate -> snapshot -> evolve."""
        win = self.window
        pop = self.population
        self._recent_ts.append(time.monotonic())

        self._emit("signals")
        signals = compute_signals(self.window_features, pop,
                                  progress_cb=lambda frac: self._emit("signals", frac))
        self._emit("is_start")
        all_rows = np.arange(pop.shape[0])
        is_result = self._run_backtest(win, signals, pop, all_rows)
        is_metrics = self._compute_metrics(is_result.equity_1h, is_result.num_trades)
        self._emit("is_done")

        is_fitness = asnumpy(is_metrics.fitness)
        is_sharpe = asnumpy(is_metrics.sharpe)

        # Cross-sub-window consistency: a genome must survive both overall robustness
        # and its worst contiguous IS slice, not just one lucky window.
        k_slices = int(getattr(self.ga, "multi_window_slices", 1))
        if k_slices > 1:
            slice_returns = self._run_is_slices(win, is_result.equity_1h, k_slices)
            if slice_returns is not None:
                mws = asnumpy(multi_window_score(slice_returns))
                mws = np.nan_to_num(mws, nan=0.0, posinf=0.0, neginf=0.0)
                base = np.maximum(is_fitness, 0.0)
                blended = np.sqrt(base * np.maximum(mws, 0.0))
                # Tiny additive base term keeps a selection gradient when no genome
                # clears the consistency bar. It is per-genome (population-independent),
                # so an elite's fitness is identical across generations (elitism holds).
                is_fitness = blended + 1e-9 * base

        best_idx = int(np.argmax(is_fitness))
        self.best_fitness_history.append(float(is_fitness[best_idx]))
        if float(is_fitness[best_idx]) > self.best_is_fitness:
            self.best_is_fitness = float(is_fitness[best_idx])
            self.best_is_genome = np.array(self.population[best_idx], dtype=np.float64)

        if win.is_h0 == 0 or not self._initial_is_curve:
            is_eq_best = asnumpy(is_result.equity_1h)[best_idx]
            is_ts_best = self.hour_ts[win.is_h0:win.is_h1]
            self._initial_is_curve = [[int(t.value // 10**6), float(v)] for t, v in zip(is_ts_best, is_eq_best)]

        k = min(self.ga.oos_candidates, pop.shape[0])
        topk = np.argsort(-is_fitness)[:k]

        oos_sharpe = np.full(pop.shape[0], np.nan)
        oos_trades = np.zeros(pop.shape[0], dtype=np.int64)
        oos_fitness = np.full(pop.shape[0], np.nan)
        oos_equity_best = None

        if k > 0:
            self._emit("oos_start")
            oos_result = self._run_oos_backtest(win, signals, pop[topk], topk)
            oos_metrics = self._compute_metrics(oos_result.equity_1h, oos_result.num_trades)
            oos_sharpe[topk] = asnumpy(oos_metrics.sharpe)
            oos_trades[topk] = asnumpy(oos_metrics.trades)
            oos_fitness[topk] = asnumpy(oos_metrics.fitness)
            # ``best_idx`` is selected on IS and need not be among ``topk`` if the
            # candidate set changes. The displayed OOS curve must belong to a genome
            # that was actually evaluated on this untouched OOS slice.
            oos_order = np.lexsort((-is_fitness[topk], -oos_sharpe[topk],
                                    -oos_fitness[topk]))
            oos_best_pos = int(oos_order[0])
            oos_best_idx = int(topk[oos_best_pos])
            oos_equity_best = asnumpy(oos_result.equity_1h)[oos_best_pos]
            self._record_oos_segment(win, oos_best_idx, oos_equity_best,
                                      float(is_sharpe[oos_best_idx]),
                                      float(oos_sharpe[oos_best_idx]),
                                      int(oos_trades[oos_best_idx]))
            self._collect_window_candidates(topk, oos_fitness, oos_sharpe)
            self._emit("oos_done")

        self._emit("breeding")
        snap = self._build_snapshot(best_idx, is_fitness, is_sharpe, oos_sharpe,
                                    oos_fitness, oos_trades, is_result, oos_equity_best)

        self.population = _next_generation(pop, is_fitness, self.ga, self.rng)
        self._last_is_fitness = is_fitness.copy()
        self.generation += 1
        self.completed_generations += 1
        self._maybe_persist()
        return snap

    # ------------------------------------------------------- walk-forward history
    def _record_oos_segment(self, win: WindowSpec, best_idx: int, oos_equity: np.ndarray,
                            is_sharpe: float, oos_sharpe: float, oos_trades: int) -> None:
        key = win.is_h0
        ts = self.hour_ts[win.is_h1:win.oos_h1]
        gid = _genome_id(self.population[best_idx])
        self._spliced_cache = None
        self._cached_agg_curve = None
        self._cached_agg_metrics = None
        self._cached_benchmark_curve = None
        self._cached_slices = None
        self.oos_segments[key] = {
            "timestamps": [int(t.value // 10**6) for t in ts],
            "equity": [float(v) for v in oos_equity],
            "genome_id": gid,
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "trades": oos_trades,
            "cycle": self.cycle,
        }
        self.window_board[key] = {
            "index": self.window_index,
            "is_start": str(self.hour_ts[win.is_h0]),
            "is_end": str(self.hour_ts[win.is_h1 - 1]),
            "oos_start": str(self.hour_ts[win.is_h1]),
            "oos_end": str(self.hour_ts[win.oos_h1 - 1]),
            # Retained for existing consumers: this is the window-local OOS winner.
            "champion_id": gid,
            "window_genome_id": gid,
            "oos_sharpe": oos_sharpe,
            "updated_cycle": self.cycle,
        }

    def _record_consistency(self, genome_id: str, oos_sharpe: float,
                            genome_vec: Optional[np.ndarray] = None) -> None:
        if genome_vec is not None:
            self._genome_vecs[genome_id] = np.asarray(genome_vec, dtype=np.float64)
        stat = self.consistency.setdefault(genome_id, {"oos_by_window": {}})
        stat["oos_by_window"][self.window_index] = oos_sharpe
        vals = np.asarray(list(stat["oos_by_window"].values()), dtype=np.float64)
        stat["windows_seen"] = int(vals.size)
        stat["windows_positive"] = int(np.sum(vals > 0.0))
        stat["oos_sharpe_mean"] = float(vals.mean())
        stat["oos_sharpe_std"] = float(vals.std())
        stat["oos_sharpe_min"] = float(vals.min())
        stat["consistency"] = float(stat["windows_positive"] / max(1, stat["windows_seen"]))
        stat["last_cycle"] = self.cycle

    def _consistency_summary(self, genome_id: str) -> Dict[str, Any]:
        s = self.consistency.get(genome_id, {})
        return {
            "windows_seen": int(s.get("windows_seen", 0)),
            "windows_positive": int(s.get("windows_positive", 0)),
            "oos_sharpe_mean": float(s.get("oos_sharpe_mean", 0.0)),
            "oos_sharpe_std": float(s.get("oos_sharpe_std", 0.0)),
            "oos_sharpe_min": float(s.get("oos_sharpe_min", 0.0)),
            "consistency": float(s.get("consistency", 0.0)),
        }

    # ------------------------------------------------ forward-only champion ledger
    def _collect_window_candidates(self, topk: np.ndarray, oos_fitness: np.ndarray,
                                   oos_sharpe: np.ndarray) -> None:
        """Keep the best OOS evidence for each genome until this WFO window completes."""
        count = min(self.config.champion_validation.candidates_per_wfo_round, len(topk))
        ranked = sorted((int(i) for i in topk if np.isfinite(oos_fitness[i])),
                        key=lambda i: (-float(oos_fitness[i]), -float(oos_sharpe[i]),
                                       _genome_id(self.population[i])))[:count]
        for i in ranked:
            vec = np.asarray(self.population[i], dtype=np.float64)
            gid = _genome_id(vec)
            candidate = {
                "id": gid, "genome": [float(x) for x in vec],
                "params": asdict(Genome.decode(vec)), "discovery_end_hour": self.window.oos_h1,
                "discovery_end": str(self.hour_ts[self.window.oos_h1 - 1]),
                "periods": [], "status": "pending_admission", "score": float(oos_fitness[i]),
                "oos_fitness": float(oos_fitness[i]), "oos_sharpe": float(oos_sharpe[i]),
                "discovery_oos_sharpe": float(oos_sharpe[i]),
            }
            old = self._window_leaderboard.get(gid)
            if old is None or candidate["oos_fitness"] > old["oos_fitness"]:
                self._window_leaderboard[gid] = candidate

    def _admit_window_candidates(self) -> None:
        """Admit a bounded set once per completed window, replacing weak active entries."""
        policy = self.config.champion_validation
        ranked = sorted(self._window_leaderboard.values(),
                        key=lambda e: (-float(e["oos_fitness"]), -float(e["oos_sharpe"]), e["id"]))
        for entry in ranked[:policy.candidates_per_wfo_round]:
            gid = entry["id"]
            if gid in self.candidate_registry:
                continue
            active = [e for e in self.candidate_registry.values() if e.get("status") != "rejected"]
            if len(active) >= policy.maximum_candidates:
                replaceable = [e for e in active if e.get("status") != "qualified"]
                weakest = min(replaceable, key=lambda e: (float(e.get("score", 0.0)), e["id"]), default=None)
                if weakest is None or float(entry["oos_fitness"]) <= float(weakest.get("score", 0.0)):
                    continue
                weakest.update(status="rejected", rejection_reasons=["replaced_by_stronger_candidate"],
                               rejection_reason="replaced_by_stronger_candidate")
                self._validation_signal_cache.pop(weakest["id"], None)
            ratio = self._jitter_robustness(entry)
            entry["robustness_ratio"] = ratio
            if ratio < self.ga.robustness_min_ratio:
                continue
            entry.update(status="provisional", score=0.0, rejection_reasons=[])
            self.candidate_registry[gid] = entry
            # Signals are full-history and causal; retain them for all future periods.
            vec = np.asarray(entry["genome"], dtype=np.float64)[None, :]
            self._validation_signal_cache[gid] = compute_signals(self.features, vec)
        # Keep candidate registry bounded to prevent snapshot/serialization bloat
        max_total = policy.maximum_candidates * 2
        if len(self.candidate_registry) > max_total:
            prune = [k for k, v in self.candidate_registry.items() if v.get("status") == "rejected"]
            for k in prune[:len(self.candidate_registry) - policy.maximum_candidates]:
                self.candidate_registry.pop(k, None)
                self._validation_signal_cache.pop(k, None)
        self._window_leaderboard.clear()
        self._evaluate_forward_validation()

    def _get_candidate_signals(self, entry: Dict[str, Any]) -> SignalIntents:
        gid = entry["id"]
        sig = self._validation_signal_cache.get(gid)
        if sig is None:
            vec = np.asarray(entry["genome"], dtype=np.float64)[None, :]
            sig = compute_signals(self.features, vec)
            self._validation_signal_cache[gid] = sig
        return sig

    def _evaluate_forward_validation(self) -> None:
        """Evaluate only complete anchored periods whose start is after discovery."""
        policy = self.config.champion_validation
        reached_end = max((w.oos_h1 for k, w in enumerate(self.windows)
                           if self.windows[k].is_h0 in self.window_board), default=0)
        period_hours, step_hours = policy.validation_period_days * 24, policy.validation_step_days * 24
        jobs: Dict[int, List[Dict[str, Any]]] = {}
        for entry in self.candidate_registry.values():
            if entry.get("status") == "rejected":
                continue
            if len(entry.get("periods", [])) >= policy.required_periods:
                continue
            done = {int(p["start_hour"]) for p in entry["periods"]}
            needed = policy.required_periods - len(done)
            for start in range(0, max(0, reached_end - period_hours + 1), step_hours):
                end = start + period_hours
                if start >= int(entry["discovery_end_hour"]) and end <= reached_end and start not in done:
                    jobs.setdefault(start, []).append(entry)
                    needed -= 1
                    if needed <= 0:
                        break
        self.validation_jobs_total = len(jobs)
        self.validation_jobs_completed = 0
        old_status = self.run_status
        if jobs:
            self.run_status = "validating"
        for start, raw_entries in jobs.items():
            entries = [e for e in raw_entries if e.get("status") != "rejected"]
            if not entries:
                continue
            end = start + period_hours
            genome = np.asarray([e["genome"] for e in entries], dtype=np.float64)
            entry_sigs = [self._get_candidate_signals(e) for e in entries]
            # Each candidate's cached full-history signals is sliced, then backtested together.
            signals = SignalIntents(
                regime=xp.concatenate([s.regime for s in entry_sigs]),
                atr=xp.concatenate([s.atr for s in entry_sigs]),
                long_entry=xp.concatenate([s.long_entry for s in entry_sigs]),
                short_entry=xp.concatenate([s.short_entry for s in entry_sigs]),
                entry_price=xp.concatenate([s.entry_price for s in entry_sigs]),
            )
            sig = self._slice_signals(signals, np.arange(len(entries)), start, end)
            res = backtest(self.ohlcv_1m.iloc[start * 60:end * 60], self.ohlcv_1h.iloc[start:end],
                           self.funding.iloc[start:end], sig, genome, self.instrument,
                           initial_equity=self.config.risk.initial_equity,
                           use_numba=self.ga.use_numba, fill_model=self.config.fills,
                           **self._breaker_kwargs())
            equities = asnumpy(res.equity_1h)
            metrics = self._compute_metrics(equities, asnumpy(res.num_trades))
            for row, entry in enumerate(entries):
                eq = equities[row]
                sharpe_value, sortino_value = float(asnumpy(metrics.sharpe)[row]), float(asnumpy(metrics.sortino)[row])
                dd, total_return = float(asnumpy(metrics.max_drawdown)[row]), float(asnumpy(metrics.total_return)[row])
                trades = int(asnumpy(res.num_trades)[row])
                reasons = []
                if sharpe_value < policy.minimum_period_sharpe: reasons.append("sharpe")
                if dd > policy.maximum_period_drawdown: reasons.append("max_drawdown")
                if trades < policy.minimum_trades_per_period: reasons.append("trades")
                entry["periods"].append({
                    "start_hour": start, "end_hour": end,
                    "start": str(self.hour_ts[start]), "end": str(self.hour_ts[end - 1]),
                    "sharpe": sharpe_value, "sortino": sortino_value, "max_drawdown": dd,
                    "total_return": total_return, "trades": trades,
                    "passed": not reasons, "failure_reasons": reasons,
                })
                self._validation_curves.setdefault(entry["id"], {})[start] = [
                    [int(t.value // 10**6), float(v)] for t, v in zip(self.hour_ts[start:end], eq)]
                self._refresh_candidate(entry)
            self.validation_jobs_completed += 1
            self._emit("validation", self.validation_jobs_completed / self.validation_jobs_total)
        if jobs:
            self.run_status = old_status if old_status != "validating" else "running"
        self._synchronize_deployment_pool()

    def _refresh_candidate(self, entry: Dict[str, Any]) -> None:
        periods = entry["periods"]
        if not periods:
            entry.update(status="provisional", score=0.0, rejection_reasons=[])
            return
        policy = self.config.champion_validation
        sharpes = np.asarray([p["sharpe"] for p in periods], dtype=float)
        returns = np.asarray([p["total_return"] for p in periods], dtype=float)
        dds = np.asarray([p["max_drawdown"] for p in periods], dtype=float)
        trades = np.asarray([p["trades"] for p in periods], dtype=float)
        positive_ratio = float(np.mean(returns > 0.0))
        rejected = any(not p["passed"] for p in periods)
        rejection_reasons = sorted({reason for period in periods
                                    for reason in period.get("failure_reasons", [])})
        # Turnover guard: reject candidates trading far above the policy's period budget.
        if float(np.median(trades)) > policy.maximum_trades_per_period:
            rejected = True
            rejection_reasons = sorted(set(rejection_reasons) | {"turnover"})
        qualified = (not rejected and len(periods) >= policy.required_periods
                     and positive_ratio >= policy.minimum_positive_period_ratio)
        # Downside/trade-aware component (median Sharpe x worst DD x trade confidence).
        dps = float(np.asarray(asnumpy(downside_penalized_score(
            np.asarray([float(np.median(sharpes))]),
            np.asarray([float(np.max(dds))]),
            np.asarray([float(np.median(trades))]),
            policy.minimum_trades_per_period))).ravel()[0])
        # Robustness rewards central performance and coverage, while making a bad tail costly.
        score = (float(np.median(sharpes)) + positive_ratio + 10.0 * float(np.median(returns))
                 + min(1.0, float(np.median(trades)) / max(1, policy.minimum_trades_per_period))
                 - float(np.std(sharpes)) - max(0.0, -float(np.min(sharpes)))
                 - 5.0 * float(np.max(dds)) + 2.0 * dps)
        # Down-weight candidates that sit on a sharp fitness peak (jitter-unstable).
        score *= float(entry.get("robustness_ratio", 1.0))
        # Discovery-to-forward gap: selection-window OOS Sharpe minus median forward
        # Sharpe (tracked per candidate, surfaced for the selected champion).
        entry["overfit_gap"] = (float(entry.get("discovery_oos_sharpe", 0.0))
                                - float(np.median(sharpes)))
        entry.update(period_count=len(periods), positive_period_ratio=positive_ratio,
                      median_sharpe=float(np.median(sharpes)), median_return=float(np.median(returns)),
                      worst_sharpe=float(np.min(sharpes)), worst_drawdown=float(np.max(dds)),
                      median_trades=float(np.median(trades)), total_trades=int(np.sum(trades)),
                      positive_periods=int(np.sum(returns > 0.0)), sharpe_std=float(np.std(sharpes)),
                       rejection_reasons=rejection_reasons, score=float(np.nan_to_num(score)),
                       status="rejected" if rejected else ("qualified" if qualified else "provisional"),
                       rejection_reason=(rejection_reasons[0] if rejection_reasons else None))
        if rejected:
            self._validation_signal_cache.pop(entry["id"], None)

    def _evaluate_holdout_once(self) -> None:
        """Evaluate the deployed champion exactly once on the reserved trailing holdout.

        The trailing ``wfo.holdout_days`` are excluded from every WFO window, so this is
        the only time that data is touched. It reports (never selects): the result is a
        one-shot generalization check, not another optimization signal.
        """
        if self._holdout_done or self.holdout_hours <= 0 \
                or self.n_hours <= self.holdout_hours:
            return
        entry = self.candidate_registry.get(self.champion_genome_id or "")
        if entry is not None:
            genome_vec = np.asarray(entry["genome"], dtype=np.float64)
            gid = str(entry["id"])
            signals = self._get_candidate_signals(entry)
        elif self.best_is_genome is not None:
            # No validated champion yet: check whatever the demo/executor would deploy
            # (the best in-sample genome) so the holdout is still reported.
            genome_vec = np.asarray(self.best_is_genome, dtype=np.float64)
            gid = "best_is_fallback"
            signals = None
        else:
            return
        self._holdout_done = True
        start = self.n_hours - self.holdout_hours
        genome = genome_vec[None, :]
        if signals is None:
            signals = compute_signals(self.features, genome)
        sig = self._slice_signals(signals, np.arange(1), start, self.n_hours)
        res = backtest(self.ohlcv_1m.iloc[start * 60:], self.ohlcv_1h.iloc[start:],
                       self.funding.iloc[start:], sig, genome, self.instrument,
                       initial_equity=self.config.risk.initial_equity,
                       use_numba=self.ga.use_numba, fill_model=self.config.fills,
                       **self._breaker_kwargs())
        equities = asnumpy(res.equity_1h)
        metrics = self._compute_metrics(equities, asnumpy(res.num_trades))
        self.holdout_report = {
            "genome_id": gid,
            "start": str(self.hour_ts[start]),
            "end": str(self.hour_ts[-1]),
            "days": int(self.holdout_hours // 24),
            "sharpe": float(asnumpy(metrics.sharpe)[0]),
            "max_drawdown": float(asnumpy(metrics.max_drawdown)[0]),
            "total_return": float(asnumpy(metrics.total_return)[0]),
            "trades": int(asnumpy(res.num_trades)[0]),
            "fees": float(asnumpy(res.total_fees)[0]),
            "funding": float(asnumpy(res.total_funding)[0]),
        }

    def _candidate_summary(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        # Period curves are retained privately only for the selected champion.
        periods = [{k: v for k, v in period.items() if k != "curve"}
                   for period in entry.get("periods", [])]
        summary = {k: entry.get(k) for k in ("id", "params", "genome", "discovery_end", "status", "score",
                "period_count", "positive_period_ratio", "median_sharpe", "median_return",
                "worst_sharpe", "worst_drawdown", "median_trades", "total_trades",
                "positive_periods", "sharpe_std", "rejection_reasons", "robustness_ratio",
                "discovery_oos_sharpe", "overfit_gap")}
        summary["periods"] = periods
        summary["rejection_reason"] = entry.get("rejection_reason")
        summary["periods_completed"] = int(entry.get("period_count", 0))
        summary["required_periods"] = self.config.champion_validation.required_periods
        summary["deployable"] = entry.get("status") == "qualified"
        return summary

    @property
    def deployable_candidates(self) -> List[Dict[str, Any]]:
        """Validation-qualified candidates suitable for the deployment API."""
        return list(self.deployment_pool)

    def _probability_of_backtest_overfitting(self) -> float:
        """CSCV PBO over the candidate x common-period return matrix.

        Returns 0.5 (not estimable) when fewer than two candidates share at least four
        completed validation periods.
        """
        rows = [e for e in self.candidate_registry.values() if e.get("periods")]
        if len(rows) < 2:
            return 0.5
        common: Optional[set] = None
        for e in rows:
            starts = {int(p["start_hour"]) for p in e["periods"]}
            common = starts if common is None else (common & starts)
        common = sorted(common or [])
        if len(common) < 4:
            return 0.5
        usable = [e for e in rows
                  if set(common) <= {int(p["start_hour"]) for p in e["periods"]}]
        if len(usable) < 2:
            return 0.5
        matrix = np.asarray(
            [[next(float(p["total_return"]) for p in e["periods"]
                   if int(p["start_hour"]) == s) for s in common]
             for e in usable], dtype=np.float64)
        return probability_of_backtest_overfitting(matrix)

    def deployment_ensemble(self, max_members: int = 5) -> List[Dict[str, Any]]:
        """Risk-weighted allocation view over the validated candidates.

        Weights come from candidates' validation-period total returns via inverse-vol
        (equal weights when candidates share no common periods). This is the allocation
        layer for ensemble trading; single-genome execution remains the default.
        """
        qualified = [e for e in self.candidate_registry.values()
                     if e.get("status") == "qualified"]
        qualified = sorted(qualified, key=lambda e: (-float(e.get("score", 0.0)), e["id"]))
        qualified = qualified[:max_members]
        if not qualified:
            return []
        common: Optional[set] = None
        for e in qualified:
            starts = {int(p["start_hour"]) for p in e.get("periods", [])}
            common = starts if common is None else (common & starts)
        common = sorted(common or [])
        if len(common) >= 2:
            matrix = np.asarray(
                [[next(float(p["total_return"]) for p in e["periods"]
                       if int(p["start_hour"]) == s) for e in qualified]
                 for s in common], dtype=np.float64)
        else:
            matrix = np.zeros((1, len(qualified)), dtype=np.float64)
        members = weights_from_members(
            [EnsembleMember(genome_id=e["id"],
                            genome=tuple(float(x) for x in e["genome"]))
             for e in qualified], matrix)
        by_id = {e["id"]: e for e in qualified}
        return [{"id": m.genome_id, "weight": float(m.weight),
                 "score": float(by_id[m.genome_id].get("score", 0.0)),
                 "median_sharpe": float(by_id[m.genome_id].get("median_sharpe", 0.0)),
                 "genome": list(m.genome),
                 "params": by_id[m.genome_id].get("params", {})}
                for m in members]

    def _synchronize_deployment_pool(self) -> None:
        qualified = [self._candidate_summary(e) for e in self.candidate_registry.values()
                     if e.get("status") == "qualified"]
        self.deployment_pool = sorted(qualified, key=lambda e: (-float(e["score"]), e["id"]))
        self._pool_keys = {tuple(np.round(np.asarray(e["genome"]), 6)) for e in self.deployment_pool}

    def _maybe_update_global_champion(self, force: bool = False) -> None:
        candidates = [e for e in self.candidate_registry.values() if e.get("period_count", 0) > 0]
        qualified = [e for e in candidates if e.get("status") == "qualified"]
        choices = qualified or [e for e in candidates if e.get("status") == "provisional"]
        if not choices:
            self.champion_genome_id, self.champion_equity_curve = None, []
            self.champion_score = -np.inf
            return
        selected = min(choices, key=lambda e: (-float(e["score"]), e["id"]))
        if not force and selected["id"] == self.champion_genome_id and self.champion_equity_curve:
            return
        self.champion_genome_id = selected["id"]
        self.champion_score = float(selected["score"])
        self._ensure_champion_curves(selected)
        self.champion_equity_curve = self._compound_validation_curves(selected)

    def _ensure_champion_curves(self, entry: Dict[str, Any]) -> None:
        """Reconstruct missing selected-champion curves from cached causal signals.

        Non-champion period curves are intentionally discarded to keep snapshots and
        checkpoints bounded. The full-history signal cache avoids indicator recomputation
        if a later score change promotes a different candidate.
        """
        gid = entry["id"]
        curves = self._validation_curves.setdefault(gid, {})
        signals = self._get_candidate_signals(entry)
        genome = np.asarray(entry["genome"], dtype=np.float64)[None, :]
        # Ensure pre-discovery period (0 to discovery_end_hour) is evaluated so the curve starts at t=0
        disc_end = int(entry.get("discovery_end_hour", 0))
        if disc_end > 0 and 0 not in curves:
            sig0 = self._slice_signals(signals, np.arange(1), 0, disc_end)
            res0 = backtest(self.ohlcv_1m.iloc[0:disc_end * 60], self.ohlcv_1h.iloc[0:disc_end],
                            self.funding.iloc[0:disc_end], sig0, genome, self.instrument,
                            initial_equity=self.config.risk.initial_equity,
                            use_numba=self.ga.use_numba, fill_model=self.config.fills,
                            **self._breaker_kwargs())
            eq0 = asnumpy(res0.equity_1h)[0]
            curves[0] = [[int(t.value // 10**6), float(v)]
                         for t, v in zip(self.hour_ts[0:disc_end], eq0)]

        for period in entry["periods"]:
            start, end = int(period["start_hour"]), int(period["end_hour"])
            if start in curves:
                continue
            sig = self._slice_signals(signals, np.arange(1), start, end)
            res = backtest(self.ohlcv_1m.iloc[start * 60:end * 60], self.ohlcv_1h.iloc[start:end],
                           self.funding.iloc[start:end], sig, genome, self.instrument,
                           initial_equity=self.config.risk.initial_equity,
                           use_numba=self.ga.use_numba, fill_model=self.config.fills,
                           **self._breaker_kwargs())
            eq = asnumpy(res.equity_1h)[0]
            curves[start] = [[int(t.value // 10**6), float(v)]
                             for t, v in zip(self.hour_ts[start:end], eq)]

    def _compound_validation_curves(self, entry: Dict[str, Any]) -> List[List[float]]:
        running, curve = self.config.risk.initial_equity, []
        curves_dict = self._validation_curves.get(entry["id"], {})
        if 0 in curves_dict:
            raw0 = curves_dict[0]
            if raw0:
                curve.extend([[t, float(v)] for t, v in raw0])
                running = raw0[-1][1]
        for period in sorted(entry["periods"], key=lambda p: p["start_hour"]):
            raw = curves_dict.get(int(period["start_hour"]), period.get("curve", []))
            if not raw:
                continue
            scale = running / raw[0][1] if raw[0][1] else 1.0
            curve.extend([[t, float(v * scale)] for t, v in raw])
            running = raw[-1][1] * scale
        return curve

    def _build_spliced_walk_forward_curve(self) -> List[List[float]]:
        """Construct a continuous walk-forward curve starting from t=0.

        Begins with Window 0's best in-sample equity curve (t=0 to is_h1),
        then chains each completed out-of-sample window segment, compounding
        seamlessly from the previous segment's ending equity.
        """
        if not hasattr(self, "_initial_is_curve") or not self._initial_is_curve:
            return []

        n_segs = len(self.oos_segments)
        if hasattr(self, "_spliced_cache") and getattr(self, "_spliced_cache_len", -1) == n_segs:
            return self._spliced_cache

        curve = [list(pt) for pt in self._initial_is_curve]
        running = curve[-1][1] if curve else self.config.risk.initial_equity

        for k in sorted(self.oos_segments.keys()):
            seg = self.oos_segments[k]
            timestamps = seg.get("timestamps", [])
            equities = seg.get("equity", [])
            if not timestamps or not equities:
                continue
            if timestamps[0] >= curve[-1][0]:
                scale = running / equities[0] if equities[0] > 0 else 1.0
                for t, eq in zip(timestamps, equities):
                    curve.append([t, float(eq * scale)])
                running = equities[-1] * scale

        self._spliced_cache = curve
        self._spliced_cache_len = n_segs
        return curve

    def _build_window_benchmark_equity(self, win: WindowSpec) -> List[List[float]]:
        """SOL benchmark curve normalized to $10,000 at the start of the active window."""
        start, end = win.is_h0, win.oos_h1
        if end > self.n_hours:
            end = self.n_hours
        if start >= end:
            return []
        closes = self.ohlcv_1h["close"].iloc[start:end].to_numpy(dtype=np.float64)
        if len(closes) == 0 or closes[0] <= 0:
            return []
        first_price = float(closes[0])
        initial = self.config.risk.initial_equity
        ts = self.hour_ts[start:end]
        return [[int(t.value // 10**6), float(initial * p / first_price)] for t, p in zip(ts, closes)]

    def _build_aggregate_equity(self) -> List[List[float]]:
        t0 = int(self.hour_ts[0].value // 10**6) if self.n_hours > 0 else 0
        # 1. If we have a validated champion curve starting at t=0, use it
        if (self.champion_equity_curve and len(self.champion_equity_curve) > 0
                and self.champion_equity_curve[0][0] == t0):
            return list(self.champion_equity_curve)

        # 2. Spliced continuous walk-forward curve starting from t=0
        spliced = self._build_spliced_walk_forward_curve()
        if spliced and len(spliced) > 0 and spliced[0][0] == t0:
            return spliced

        # 3. Fallback to Window 0 IS curve starting from t=0
        if hasattr(self, "_initial_is_curve") and self._initial_is_curve:
            return list(self._initial_is_curve)

        return list(self.champion_equity_curve or [])

    def _build_benchmark_equity(self, curve: List[List[float]]) -> List[List[float]]:
        if not curve:
            return []
        if self._close_by_ts is None:
            self._close_by_ts = {int(t.value // 10**6): float(v)
                                 for t, v in zip(self.hour_ts, self.ohlcv_1h["close"])}
        close_by_timestamp = self._close_by_ts
        first_price = close_by_timestamp.get(curve[0][0])
        if not first_price:
            return []
        initial = self.config.risk.initial_equity
        return [[timestamp, float(initial * close_by_timestamp[timestamp] / first_price)]
                for timestamp, _ in curve if timestamp in close_by_timestamp]

    def _build_full_history_benchmark_equity(self) -> List[List[float]]:
        """Return the contextual SOL buy-and-hold curve over all available 1h closes.

        Static for a given dataset, so it is computed once and cached.
        """
        if self._full_history_benchmark is not None:
            return self._full_history_benchmark
        if self.n_hours == 0:
            self._full_history_benchmark = []
            return self._full_history_benchmark
        closes = self.ohlcv_1h["close"].to_numpy(dtype=np.float64)
        first_price = float(closes[0])
        if not np.isfinite(first_price) or first_price == 0.0:
            self._full_history_benchmark = []
            return self._full_history_benchmark
        initial = self.config.risk.initial_equity
        self._full_history_benchmark = [
            [int(timestamp.value // 10**6), float(initial * price / first_price)]
            for timestamp, price in zip(self.hour_ts, closes)]
        return self._full_history_benchmark

    def _coverage_metadata(self, selected: Optional[Dict[str, Any]],
                           evidence_curve: List[List[float]]) -> Dict[str, Any]:
        discovery_end = None
        if selected is not None:
            discovery_hour = int(selected.get("discovery_end_hour", 0))
            if 0 < discovery_hour <= self.n_hours:
                discovery_end = int(self.hour_ts[discovery_hour - 1].value // 10**6)
        return {
            "history_start": int(self.hour_ts[0].value // 10**6),
            "history_end": int(self.hour_ts[-1].value // 10**6),
            "candidate_discovery_end": discovery_end,
            "validation_coverage_start": evidence_curve[0][0] if evidence_curve else None,
            "validation_coverage_end": evidence_curve[-1][0] if evidence_curve else None,
            "coverage_semantics": "forward_only_completed_validation",
            "validation_period_days": self.config.champion_validation.validation_period_days,
            "validation_step_days": self.config.champion_validation.validation_step_days,
        }

    def _aggregate_metrics(self, curve: List[List[float]]) -> Dict[str, Any]:
        if len(curve) < 2:
            return {}
        eq = np.asarray([v for _, v in curve], dtype=np.float64)
        # Aggregate points can be separated by WFO step gaps and overlapping windows.
        # They are not contiguous hourly observations, so annualizing them as 8760
        # samples/year would materially overstate risk-adjusted performance.
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        mean_return = float(returns.mean())
        return_std = float(returns.std())
        sharpe_value = mean_return / return_std if return_std > 0.0 else 0.0
        downside = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))
        sortino_value = mean_return / downside if downside > 0.0 else (10.0 if mean_return > 0.0 else 0.0)
        peak = np.maximum.accumulate(eq)
        drawdown = 1.0 - eq / np.maximum(peak, 1e-12)
        # Statistical significance of the reported Sharpe against selection over many
        # candidates (PSR) and against the expected max Sharpe under the null (DSR).
        psr_value = float(asnumpy(probabilistic_sharpe_ratio(returns[None, :], 0.0,
                                                             HOURS_PER_YEAR))[0])
        n_obs = int(returns.size)
        sd = float(returns.std())
        if n_obs >= 3 and sd > 0.0:
            z = (returns - float(returns.mean())) / sd
            skew = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4))
        else:
            skew, kurt = 0.0, 3.0
        trial_sr = [float(e.get("median_sharpe", 0.0))
                    for e in self.candidate_registry.values()]
        sr_var = float(np.var(trial_sr)) if len(trial_sr) >= 2 else 0.0
        dsr_value = float(np.asarray(asnumpy(deflated_sharpe_ratio(
            float(sharpe_value), max(2, len(self.candidate_registry)), n_obs,
            skew, kurt, sr_var, 0.0))).ravel()[0])
        return {
            "sharpe": float(sharpe_value),
            "sortino": float(sortino_value),
            "max_drawdown": float(np.clip(drawdown.max(), 0.0, 1.0)),
            "total_return": float(eq[-1] / max(eq[0], 1e-12) - 1.0),
            "annualization": "none (per forward-validation hourly interval)",
            "psr": psr_value,
            "dsr": dsr_value,
            "dsr_n_trials": int(max(2, len(self.candidate_registry))),
            "pbo": self._probability_of_backtest_overfitting(),
            "overfit_gap": float(
                (self.candidate_registry.get(self.champion_genome_id or "") or {})
                .get("overfit_gap", 0.0) or 0.0),
            "coverage_start": curve[0][0],
            "coverage_end": curve[-1][0],
            "history_start": int(self.hour_ts[0].value // 10**6),
            "history_end": int(self.hour_ts[-1].value // 10**6),
        }

    # ------------------------------------------------------------------ snapshot
    def build_dashboard_snapshot(self, last_snapshot: Optional[Snapshot] = None) -> Snapshot:
        """Build the authoritative current-state snapshot for dashboard consumers.

        ``last_snapshot`` preserves the most recent window's IS/OOS and top-genome detail,
        while all engine-wide fields are refreshed after candidate admission and validation.
        """
        agg_curve = self._build_aggregate_equity()
        selected = self.candidate_registry.get(self.champion_genome_id or "")
        candidate_summaries = [self._candidate_summary(entry) for entry in
                               sorted(self.candidate_registry.values(),
                                      key=lambda entry: (-float(entry.get("score", 0.0)), entry["id"]))]
        gens_per_s = 0.0
        if len(self._recent_ts) >= 2:
            span = self._recent_ts[-1] - self._recent_ts[0]
            gens_per_s = (len(self._recent_ts) - 1) / span if span > 0 else 0.0
        refreshed = {
            "generation": self.generation,
            "wfo_window": self._window_bounds(self.window),
            "deployment_pool_size": len(self.deployment_pool),
            "backend": get_backend_info()["backend"],
            "window_index": self.window_index,
            "n_windows": len(self.windows),
            "cycle": self.cycle,
            "aggregate_equity_curve": agg_curve,
            "aggregate_metrics": self._aggregate_metrics(agg_curve),
            "window_board": [self.window_board[key] for key in sorted(self.window_board.keys())],
            "slices": self._build_slices(),
            "gens_per_s": float(gens_per_s),
            "champion_id": self.champion_genome_id,
            "champion_score": None if not np.isfinite(self.champion_score) else float(self.champion_score),
            "champion_status": (selected or {}).get("status"),
            "champion_validation": self._candidate_summary(selected) if selected else {},
            "champion_candidates": candidate_summaries[:max(1, self.config.champion_validation.maximum_candidates)],
            "benchmark_equity_curve": self._build_benchmark_equity(agg_curve),
            "window_benchmark_equity_curve": self._build_window_benchmark_equity(self.window),
            "full_history_benchmark_equity_curve": self._build_full_history_benchmark_equity(),
            "ensemble": self.deployment_ensemble(),
            "holdout_report": self.holdout_report,
            "coverage_metadata": self._coverage_metadata(selected, agg_curve),
            "run_status": self.run_status,
            "completed_generations": self.completed_generations,
            "planned_generations": self.planned_generations,
            "validation_jobs_completed": self.validation_jobs_completed,
            "validation_jobs_total": self.validation_jobs_total,
        }
        if last_snapshot is not None:
            return replace(last_snapshot, **refreshed)
        return Snapshot(
            best_fitness=float(self.best_is_fitness) if np.isfinite(self.best_is_fitness) else 0.0,
            is_equity_curve=[], oos_equity_curve=[], top_genomes=[], **refreshed,
        )

    def _build_snapshot(self, best_idx: int, is_fitness: np.ndarray, is_sharpe: np.ndarray,
                        oos_sharpe: np.ndarray, oos_fitness: np.ndarray,
                        oos_trades: np.ndarray, is_result: BacktestResult,
                        oos_equity_best: Optional[np.ndarray]) -> Snapshot:
        win = self.window
        is_eq = asnumpy(is_result.equity_1h)[best_idx]
        is_ts = self.hour_ts[win.is_h0:win.is_h1]
        is_curve = [[int(t.value // 10**6), float(v)] for t, v in zip(is_ts, is_eq)]

        if oos_equity_best is not None and len(is_eq) > 0 and len(oos_equity_best) > 0:
            scale = float(is_eq[-1]) / max(float(oos_equity_best[0]), 1e-12)
            oos_ts = self.hour_ts[win.is_h1:win.oos_h1]
            oos_curve = [[int(t.value // 10**6), float(v * scale)] for t, v in zip(oos_ts, oos_equity_best)]
        else:
            oos_curve = []

        k = min(self.ga.oos_candidates, is_fitness.shape[0])
        order = np.argsort(-is_fitness)[:k]
        top_genomes = []
        for rank, i in enumerate(order, start=1):
            g = Genome.decode(self.population[i])
            is_s = float(is_sharpe[i])
            oos_s = None if np.isnan(oos_sharpe[i]) else float(oos_sharpe[i])
            wfe = min(max(oos_s / is_s, 0.0), 1.5) if (is_s > 0 and oos_s is not None) else 0.0
            top_genomes.append({
                "rank": rank,
                "params": asdict(g),
                "is_sharpe": is_s,
                "oos_sharpe": oos_s,
                "wfe": wfe,
                "fitness": float(is_fitness[i]),
                "trades": int(oos_trades[i]),
            })

        if self._cached_agg_curve is None:
            self._cached_agg_curve = self._build_aggregate_equity()
            self._cached_agg_metrics = self._aggregate_metrics(self._cached_agg_curve)
            self._cached_benchmark_curve = self._build_benchmark_equity(self._cached_agg_curve)
            self._cached_slices = self._build_slices()
        agg_curve = self._cached_agg_curve
        agg_metrics = self._cached_agg_metrics
        slices = self._cached_slices
        selected = self.candidate_registry.get(self.champion_genome_id or "")
        candidate_summaries = [self._candidate_summary(e) for e in
                               sorted(self.candidate_registry.values(),
                                      key=lambda e: (-float(e.get("score", 0.0)), e["id"]))]
        board = [self.window_board[k] for k in sorted(self.window_board.keys())]
        gens_per_s = 0.0
        if len(self._recent_ts) >= 2:
            span = self._recent_ts[-1] - self._recent_ts[0]
            gens_per_s = (len(self._recent_ts) - 1) / span if span > 0 else 0.0

        return Snapshot(
            generation=self.generation,
            best_fitness=float(is_fitness[best_idx]),
            wfo_window=self._window_bounds(win),
            is_equity_curve=is_curve,
            oos_equity_curve=oos_curve,
            top_genomes=top_genomes,
            deployment_pool_size=len(self.deployment_pool),
            backend=get_backend_info()["backend"],
            window_index=self.window_index,
            n_windows=len(self.windows),
            cycle=self.cycle,
            aggregate_equity_curve=agg_curve,
            aggregate_metrics=agg_metrics,
            window_board=board,
            slices=slices,
            gens_per_s=float(gens_per_s),
            champion_id=self.champion_genome_id,
            champion_score=(None if not np.isfinite(self.champion_score)
                            else float(self.champion_score)),
            champion_status=(selected or {}).get("status"),
            champion_validation=self._candidate_summary(selected) if selected else {},
            champion_candidates=candidate_summaries[:max(1, self.config.champion_validation.maximum_candidates)],
            benchmark_equity_curve=self._cached_benchmark_curve,
            window_benchmark_equity_curve=self._build_window_benchmark_equity(win),
            run_status=self.run_status,
            completed_generations=self.completed_generations,
            planned_generations=self.planned_generations,
            validation_jobs_completed=self.validation_jobs_completed,
            validation_jobs_total=self.validation_jobs_total,
            coverage_metadata=self._coverage_metadata(selected, agg_curve),
            full_history_benchmark_equity_curve=[],
            ensemble=self.deployment_ensemble(),
            holdout_report=self.holdout_report,
        )

    def _build_slices(self) -> List[Dict[str, Any]]:
        """Return each reached window's locally OOS-validated winner for the slice view."""
        if not hasattr(self, "_slice_cache"):
            self._slice_cache = {}
        for k in sorted(self.oos_segments.keys()):
            if k not in self._slice_cache:
                seg = self.oos_segments[k]
                board = self.window_board.get(k, {})
                curve = [[t, v] for t, v in zip(seg["timestamps"], seg["equity"])]
                ret = (curve[-1][1] / curve[0][1] - 1.0) if curve and curve[0][1] else 0.0
                self._slice_cache[k] = {
                    "index": board.get("index", 0),
                    "is_start": board.get("is_start", ""),
                    "is_end": board.get("is_end", ""),
                    "oos_start": board.get("oos_start", ""),
                    "oos_end": board.get("oos_end", ""),
                    "curve": curve,
                    "sharpe": float(seg.get("oos_sharpe", 0.0)),
                    "total_return": ret,
                    "trades": int(seg.get("trades", 0)),
                    "champion_id": board.get("champion_id", seg.get("genome_id")),
                    "complete": True,
                    "winner_scope": "window_oos",
                }
        return [self._slice_cache[k] for k in sorted(self.oos_segments.keys())]

    # ------------------------------------------------------------------- run loop
    def _budget_exhausted(self, ran: int, limit: int) -> bool:
        return limit > 0 and ran >= limit

    def run(self, n_generations: int,
            callback: Optional[Callable[[Snapshot], None]] = None) -> List[Snapshot]:
        """Run at most ``n_generations`` generations total, then stop.

        ``n_generations <= 0`` runs until cancelled. In walk-forward mode the budget is
        distributed across windows (cycling if needed).
        """
        if self.walk_forward:
            return self.run_walk_forward(n_generations, callback,
                                         repeat=self.walk_forward_repeat,
                                         max_cycles=self.walk_forward_max_cycles)
        self.run_status = "running"
        self.completed_generations = 0
        self.planned_generations = max(0, n_generations)
        self._update_window_features()
        snapshots: List[Snapshot] = []
        ran = 0
        while not self._cancelled and not self._budget_exhausted(ran, n_generations):
            snap = self.step()
            snapshots.append(snap)
            ran += 1
            if callback is not None:
                callback(snap)
        self._maybe_update_global_champion(force=True)
        self._evaluate_holdout_once()
        self.run_status = "cancelled" if self._cancelled else "persisting"
        self.flush()
        if not self._cancelled:
            self.run_status = "completed"
        if callback is not None:
            callback(self.build_dashboard_snapshot(snapshots[-1] if snapshots else None))
        return snapshots

    def run_walk_forward(self, total_generations: int,
                         callback: Optional[Callable[[Snapshot], None]] = None,
                         repeat: bool = False, max_cycles: int = 0) -> List[Snapshot]:
        """Run a clean one-pass WFO schedule unless explicit repeat mode is requested.

        The ordinary finite path deliberately ignores ``total_generations``: every window
        receives ``wfo.generations_per_window`` generations exactly once.
        """
        snapshots: List[Snapshot] = []
        gpw = max(1, int(getattr(self.wfo, "generations_per_window", 20)))
        infinite = repeat and total_generations <= 0
        self.run_status = "running"
        self.completed_generations = 0
        passes = max_cycles if repeat and max_cycles else 1
        self.planned_generations = 0 if infinite else len(self.windows) * gpw * passes
        used_initial = False

        while True:
            for wi in range(self.window_index, len(self.windows)):
                if self._cancelled:
                    break
                self.window = self.windows[wi]
                self.window_index = wi
                self._update_window_features()
                if used_initial:
                    self.population = self._seed_next_window_population()
                used_initial = True
                self._window_leaderboard.clear()
                self._emit("window_start")
                for _ in range(gpw):
                    if self._cancelled:
                        break
                    snap = self.step()
                    snapshots.append(snap)
                    if callback is not None:
                        callback(snap)
                if not self._cancelled:
                    self._admit_window_candidates()
                self._maybe_update_global_champion()
            if self._cancelled:
                break
            if not repeat:
                break
            if max_cycles and (self.cycle + 1) >= max_cycles:
                break
            self.window_index = 0
            self.cycle += 1

        self._maybe_update_global_champion(force=True)
        self._evaluate_holdout_once()
        self.run_status = "cancelled" if self._cancelled else "persisting"
        self.flush()
        if not self._cancelled:
            self.run_status = "completed"
        if callback is not None:
            callback(self.build_dashboard_snapshot(snapshots[-1] if snapshots else None))
        return snapshots

    def cancel(self) -> None:
        """Request a clean stop at the next generation boundary."""
        self._cancelled = True
        self.run_status = "cancelled"

    # -------------------------------------------------------------- persistence
    def _maybe_persist(self) -> None:
        now = time.monotonic()
        if now - self._last_persist >= self.ga.persist_interval_s:
            self._last_persist = now
            self.flush()

    def flush(self) -> None:
        self._save_checkpoint()
        self._save_pool()

    def _save_checkpoint(self) -> None:
        champ_curves = {self.champion_genome_id: self._validation_curves.get(self.champion_genome_id, {})} if self.champion_genome_id else {}
        data = {
            "generation": self.generation,
            "seed": self.seed,
            "walk_forward": self.walk_forward,
            "window_index": self.window_index,
            "cycle": self.cycle,
            "population": [list(map(float, row)) for row in self.population],
            "rng_state": _rng_to_dict(self.rng),
            "best_fitness_history": self.best_fitness_history,
            "deployment_pool": self.deployment_pool,
            "candidate_registry": self.candidate_registry,
            "validation_curves": champ_curves,
            "run_status": self.run_status,
            "completed_generations": self.completed_generations,
            "planned_generations": self.planned_generations,
        }
        _atomic_json(self.ga.checkpoint_path, data)

    def _restore_checkpoint(self) -> None:
        with open(self.ga.checkpoint_path) as f:
            data = json.load(f)
        self.generation = int(data["generation"])
        self.seed = int(data["seed"])
        self.walk_forward = bool(data.get("walk_forward", False))
        self.window_index = int(data.get("window_index", 0))
        self.cycle = int(data.get("cycle", 0))
        self.window = self.windows[self.window_index]
        self.population = np.asarray(data["population"], dtype=np.float64)
        self.rng = _rng_from_dict(data["rng_state"])
        self.best_fitness_history = list(data["best_fitness_history"])
        # Old checkpoints have a legacy pool without independent validation evidence.
        # Deliberately discard it rather than treating it as proof of deployability.
        raw_registry = data.get("candidate_registry")
        self.candidate_registry = dict(raw_registry) if isinstance(raw_registry, dict) else {}
        raw_curves = data.get("validation_curves", {})
        self._validation_curves = {
            str(gid): {int(start): curve for start, curve in periods.items()}
            for gid, periods in raw_curves.items() if isinstance(periods, dict)
        } if isinstance(raw_curves, dict) else {}
        self.run_status = str(data.get("run_status", "idle"))
        self.completed_generations = int(data.get("completed_generations", 0))
        self.planned_generations = int(data.get("planned_generations", 0))
        for entry in self.candidate_registry.values():
            # Older checkpoints stored curves in periods; migrate them to private data.
            gid = entry.get("id")
            if gid and entry.get("genome"):
                self._validation_signal_cache[gid] = compute_signals(
                    self.features, np.asarray(entry["genome"], dtype=np.float64)[None, :])
            for period in entry.get("periods", []):
                curve = period.pop("curve", None)
                if curve and gid:
                    self._validation_curves.setdefault(gid, {})[int(period["start_hour"])] = curve
            self._refresh_candidate(entry)
        self._synchronize_deployment_pool()

    def _save_pool(self) -> None:
        _atomic_json(self.ga.pool_path, self.deployment_pool)


# ---------------------------------------------------------------------------
# GA operators (module-level, deterministic, numpy)
# ---------------------------------------------------------------------------

def _tournament_select(population: np.ndarray, fitness: np.ndarray, n_slots: int,
                       tournament_size: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``n_slots`` parent indices via tournament selection."""
    p = population.shape[0]
    idx = rng.integers(0, p, size=(n_slots, max(1, tournament_size)))
    fit = fitness[idx]
    return idx[np.arange(n_slots), fit.argmax(axis=1)]


def _mutate(children: np.ndarray, rng: np.random.Generator, ga: GAConfig) -> np.ndarray:
    """Gaussian mutation for continuous genes, bit-flip for discrete genes."""
    n, g = children.shape
    mut_mask = rng.random((n, g)) < ga.mutation_rate
    out = children.copy()
    for j, gene in enumerate(GENES):
        if gene.discrete:
            flip = mut_mask[:, j]
            out[flip, j] = 1.0 - out[flip, j]
        else:
            sigma = ga.mutation_sigma * (gene.high - gene.low)
            out[:, j] = out[:, j] + mut_mask[:, j] * rng.normal(0.0, sigma, size=n)
    return clamp_genomes(out)


def _next_generation(population: np.ndarray, fitness: np.ndarray, ga: GAConfig,
                     rng: np.random.Generator) -> np.ndarray:
    """Produce the next population: elitism + tournament/selection/crossover/mutation."""
    p, _ = population.shape
    n_elite = max(1, int(round(p * ga.elitism_ratio)))
    order = np.argsort(-fitness)
    elite = population[order[:n_elite]].copy()

    n_children = p - n_elite
    parents = _tournament_select(population, fitness, n_children * 2, ga.tournament_size, rng)
    a = population[parents[:n_children]]
    b = population[parents[n_children:]]

    # Per-child crossover probability (ga.crossover_rate); uniform gene swap within a child.
    do_cross = rng.random(n_children) < ga.crossover_rate
    gene_mask = rng.random((n_children, population.shape[1])) < 0.5
    children = np.where(do_cross[:, None] & gene_mask, b, a)
    children = _mutate(children, rng, ga)

    pop = np.vstack([elite, children])
    # Random immigrants: replace the weakest slots with fresh draws to prevent the
    # population from collapsing onto a single peak.
    n_imm = int(round(p * ga.immigrant_ratio))
    if n_imm > 0:
        pop[-n_imm:] = random_population(rng, n_imm)
    return pop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _genome_id(genome_vec: np.ndarray) -> str:
    """Deterministic short id for a genome vector (used by the deploy API)."""
    import hashlib
    raw = np.round(genome_vec, 6).tobytes()
    return hashlib.sha1(raw).hexdigest()[:12]


def _rng_to_dict(rng: np.random.Generator) -> Dict[str, Any]:
    state = rng.bit_generator.state
    return {
        "bit_generator": str(state["bit_generator"]),
        "state": {"state": int(state["state"]["state"]), "inc": int(state["state"]["inc"])},
        "has_uint32": int(state.get("has_uint32", 0)),
        "uinteger": int(state.get("uinteger", 0)),
    }


def _rng_from_dict(d: Dict[str, Any]) -> np.random.Generator:
    g = np.random.Generator(np.random.PCG64(0))
    g.bit_generator.state = {
        "bit_generator": d["bit_generator"],
        "state": {"state": int(d["state"]["state"]), "inc": int(d["state"]["inc"])},
        "has_uint32": int(d.get("has_uint32", 0)),
        "uinteger": int(d.get("uinteger", 0)),
    }
    return g


def _atomic_json(path: str, obj: Any) -> None:
    """Write JSON atomically (tmp file + rename)."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
