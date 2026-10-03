"""config.py — single source of truth for all tunables, ranges, and instrument spec.

This module owns every configurable value in the SOL-PERP Evolutionary Trading
System. No other module hard-codes a tunable; they import from here.

Values are grouped into frozen dataclasses (cheap to snapshot, hash, and pass between
the decoupled sub-systems: data -> indicators -> strategy -> backtester -> evolution
-> ui/live). Environment variables (see ``env.example``) are read by
``build_config()`` so secrets and host/port overrides never live in code.

Money-math conventions used throughout the system:
  * ``*_bps`` fields are basis points: 1 bps == 0.01% == 1e-4 as a fraction.
  * All fill comparisons round to ``tick_size`` first (see ``InstrumentSpec`` helpers);
    vectorized kernels quantize to integer ticks to avoid float compare bugs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence, Tuple

__all__ = [
    "InstrumentSpec",
    "DataConfig",
    "GAConfig",
    "ChampionValidationConfig",
    "FillModelConfig",
    "WFOConfig",
    "RiskConfig",
    "UIConfig",
    "LiveConfig",
    "Config",
    "GeneSpec",
    "GENES",
    "GENE_NAMES",
    "NUM_GENES",
    "REGIME_EMA",
    "REGIME_SUPERTREND",
    "PortfolioConfig",
    "DEFAULT_PORTFOLIO_INSTRUMENTS",
    "default_config",
    "build_config",
]


# ---------------------------------------------------------------------------
# Instrument spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstrumentSpec:
    """Static contract spec for SOL-PERP.

    Fee/limit/margin values are documented defaults. All are
    ``# VERIFY vs live Coinbase spec`` — they are placeholders chosen to be
    internally consistent, not copied from a live market data feed.
    """

    product_id: str = "SOL-PERP"
    # Tick size: minimum price increment. SOL ~ $150-200, so 1c ticks are plausible.
    tick_size: float = 0.01  # VERIFY vs live Coinbase spec
    # Lot size: minimum order step in units of SOL (contract/step size).
    lot_size: float = 0.01  # VERIFY vs live Coinbase spec
    # Minimum notional (USD) for a valid order.
    min_notional: float = 5.0  # VERIFY vs live Coinbase spec
    # Fees in basis points (1 bps == 0.01%). Maker ~0.02%, taker ~0.05% are typical.
    maker_fee_bps: float = 2.0  # VERIFY vs live Coinbase spec
    taker_fee_bps: float = 5.0  # VERIFY vs live Coinbase spec
    # Funding settles hourly (per the spec).
    funding_interval_hours: int = 1
    # Hard leverage cap. The strategy's target_leverage must be <= this.
    max_leverage: float = 4.0
    # Maintenance margin rate: fraction of notional that must be backed by equity.
    maintenance_margin_rate: float = 0.05  # VERIFY vs live Coinbase spec

    # -- derived fee rates ----------------------------------------------------
    @property
    def maker_fee_rate(self) -> float:
        """Maker fee as a fraction of notional."""
        return self.maker_fee_bps / 10_000.0

    @property
    def taker_fee_rate(self) -> float:
        """Taker fee as a fraction of notional."""
        return self.taker_fee_bps / 10_000.0

    # -- money-math rounding helpers ------------------------------------------
    def round_tick(self, price: float) -> float:
        """Round a price to the nearest valid tick (quantize to integer ticks first
        to avoid raw-float comparison errors)."""
        ticks = round(price / self.tick_size)
        return ticks * self.tick_size

    def round_lot(self, qty: float) -> float:
        """Round a quantity to the nearest valid lot."""
        lots = round(qty / self.lot_size)
        return lots * self.lot_size

    def floor_to_lot(self, qty: float) -> float:
        """Floor a quantity down to a whole number of lots (used by dynamic sizing)."""
        lots = math.floor(qty / self.lot_size + 1e-9)
        return lots * self.lot_size

DEFAULT_PORTFOLIO_INSTRUMENTS: Mapping[str, InstrumentSpec] = {
    "BTC-PERP": InstrumentSpec("BTC-PERP", tick_size=0.1, lot_size=0.001, min_notional=10.0, maintenance_margin_rate=0.02),
    "ETH-PERP": InstrumentSpec("ETH-PERP", tick_size=0.01, lot_size=0.01, min_notional=10.0, maintenance_margin_rate=0.025),
    "SOL-PERP": InstrumentSpec("SOL-PERP", tick_size=0.01, lot_size=0.01, min_notional=5.0, maintenance_margin_rate=0.05),
    "AVAX-PERP": InstrumentSpec("AVAX-PERP", tick_size=0.01, lot_size=0.05, min_notional=5.0, maintenance_margin_rate=0.05),
    "DOGE-PERP": InstrumentSpec("DOGE-PERP", tick_size=0.00001, lot_size=10.0, min_notional=5.0, maintenance_margin_rate=0.05),
    "LINK-PERP": InstrumentSpec("LINK-PERP", tick_size=0.001, lot_size=0.1, min_notional=5.0, maintenance_margin_rate=0.05),
    "SUI-PERP": InstrumentSpec("SUI-PERP", tick_size=0.0001, lot_size=1.0, min_notional=5.0, maintenance_margin_rate=0.05),
    "NEAR-PERP": InstrumentSpec("NEAR-PERP", tick_size=0.001, lot_size=0.5, min_notional=5.0, maintenance_margin_rate=0.05),
    "XRP-PERP": InstrumentSpec("XRP-PERP", tick_size=0.0001, lot_size=5.0, min_notional=5.0, maintenance_margin_rate=0.05),
}


@dataclass(frozen=True)
class PortfolioConfig:
    """Configuration for multi-asset macro CTA trend & carry portfolio."""

    universe: Tuple[str, ...] = (
        "BTC-PERP", "ETH-PERP", "SOL-PERP", "AVAX-PERP", "DOGE-PERP",
        "LINK-PERP", "SUI-PERP", "NEAR-PERP", "XRP-PERP"
    )
    signal_timeframe_hours: int = 4  # 4h macro trend signals
    execution_timeframe_hours: int = 1  # 1h execution & stop resolution
    trend_lookback_bars: int = 120  # 120 * 4h = 20 days entry breakout
    exit_lookback_bars: int = 72  # 72 * 4h = 12 days exit channel
    atr_period_bars: int = 42  # 42 * 4h = 7 days ATR
    atr_stop_mult: float = 3.5  # 3.5x ATR trailing stop
    target_annual_vol: float = 0.35  # 35% target portfolio volatility
    max_portfolio_leverage: float = 3.5  # max portfolio leverage across leaders
    max_single_asset_leverage: float = 2.30  # max concentration per leader
    funding_filter_z: float = 2.5  # avoid entering into >2.5 z-score adverse funding
    allow_alt_shorts: bool = False  # Long-only on alts; shorts restricted to BTC/ETH macro hedges
    allow_bear_shorts: bool = True  # Enable BTC/ETH shorting when BTC < 200d SMA
    max_concurrent_leaders: int = 2  # Focus 100% of capital on Top 2 Relative Strength leaders
    momentum_lookback_bars: int = 180  # 30 days = 180 * 4h bars for RS ranking
    macro_btc_sma_bars: int = 1200  # 200 days = 1200 * 4h bars for BTC macro regime filter
    pyramid_enabled: bool = True  # Pyramiding onto winning trades using house money
    tranche_1_leverage: float = 1.00  # Base entry leverage per leader
    tranche_2_leverage: float = 0.75  # Added at +2.5 ATR profit
    tranche_3_leverage: float = 0.55  # Added at +5.0 ATR profit
    bear_short_leverage: float = 0.50  # Macro bear short leverage (BTC/ETH only)
    pyramid_milestone_1_atr: float = 2.5  # +2.5 ATR for Tranche 2
    pyramid_milestone_2_atr: float = 5.0  # +5.0 ATR for Tranche 3
    parabolic_threshold_atr: float = 8.0  # Climax threshold
    parabolic_stop_atr: float = 2.0  # Climax trailing stop distance
    initial_equity: float = 10_000.0


# ---------------------------------------------------------------------------
# Regime mode enum (single discrete gene)
# ---------------------------------------------------------------------------

REGIME_EMA = 0
REGIME_SUPERTREND = 1


# ---------------------------------------------------------------------------
# Genome gene ranges (single source of truth; order is canonical for encode/decode)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneSpec:
    """Range/type metadata for one genome gene.

    ``discrete=True`` genes are decoded to integers (0/1 for regime_mode).
    Continuous genes use ``(low, high)``; ``gaussian mutation`` uses
    ``sigma = mutation_sigma * (high - low)``.
    """

    name: str
    low: float
    high: float
    discrete: bool = False
    integer: bool = False  # period/bar counts that get rounded at use-site


# Canonical gene order — must match strategy.Genome.encode()/decode().
GENES: Tuple[GeneSpec, ...] = (
    GeneSpec("ema_fast", 5.0, 50.0, integer=True),
    GeneSpec("ema_slow", 20.0, 200.0, integer=True),
    GeneSpec("st_period", 7.0, 30.0, integer=True),
    GeneSpec("st_mult", 2.0, 4.0),
    GeneSpec("atr_period", 7.0, 28.0, integer=True),
    GeneSpec("entry_offset_atr_mult", 0.0, 2.5),
    GeneSpec("tp_atr_mult", 2.5, 20.0),    # tightened: 35x ATR TPs never fill in real markets
    GeneSpec("sl_atr_mult", 0.8, 3.0),      # tightened: 0.5 ATR fires on 1h noise
    GeneSpec("max_hold_bars", 12.0, 720.0, integer=True),
    GeneSpec("target_leverage", 1.0, 4.0),
    GeneSpec("regime_mode", float(REGIME_EMA), float(REGIME_SUPERTREND), discrete=True),
    GeneSpec("risk_per_trade_pct", 0.020, 0.200),
    GeneSpec("trailing_atr_mult", 1.5, 10.0),
    GeneSpec("funding_filter_z", 0.5, 3.5),
    GeneSpec("volume_filter_z", -1.0, 2.5),
    GeneSpec("er_threshold", 0.15, 0.65),
    GeneSpec("adx_threshold", 15.0, 40.0),  # Wilder ADX hurdle; low = always trade, high = trend-only
)

GENE_NAMES: Tuple[str, ...] = tuple(g.name for g in GENES)
NUM_GENES: int = len(GENES)


# ---------------------------------------------------------------------------
# Sub-system config groups
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Data paths, timeframes, seeds, and synthetic-generation defaults."""

    signal_timeframe: str = "1h"
    fill_timeframe: str = "1m"
    data_dir: str = "data"  # default cache/data directory (created lazily)
    ohlcv_path: str = ""  # optional explicit OHLCV path ("" -> synthetic in demo)
    funding_path: str = ""  # optional explicit funding path
    global_seed: int = 42
    # Synthetic series defaults (used by core.data.make_synthetic).
    synth_days: int = 90
    synth_start: str = "2024-01-01T00:00:00Z"  # UTC, naive
    synth_start_price: float = 100.0
    synth_annual_vol: float = 0.60  # base annualized vol
    synth_annual_drift: float = 0.05  # long-run drift (regime adds a signed delta)
    synth_regime_drift: float = 0.60  # added annualized drift per +1 regime step
    synth_regime_vol_mult: float = 0.35  # vol multiplier per |regime| step
    # Per-hour funding is derived from a typical 0.01% per 8h perp rate (1.25e-5/hr).
    # The previous 0.0001/hr (~88%/yr) was ~8x realistic and dominated learned P&L.
    synth_funding_mean: float = 1.25e-5  # VERIFY vs live Coinbase funding history
    synth_funding_std: float = 1.5e-5
    synth_funding_beta: float = 5e-5  # funding sensitivity to hourly return
    allow_gap_ffill: bool = True  # forward-fill gaps in loaded data (with warning)
    train_year: Optional[int] = None  # optional calendar year filter (e.g. 2024)


@dataclass(frozen=True)
class GAConfig:
    """Genetic algorithm tunables."""

    population_size: int = 64
    generations: int = 20
    elitism_ratio: float = 0.05  # top fraction carried over unchanged
    tournament_size: int = 4
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15  # per-gene mutation probability
    mutation_sigma: float = 0.10  # per-gene sigma as fraction of the gene's range
    immigrant_ratio: float = 0.05  # fraction of each new generation replaced by random draws
    carryover_ratio: float = 0.10  # top fraction carried into the next WFO window
    robustness_jitter: int = 3  # parameter-perturbation samples per admitted candidate
    jitter_sigma: float = 0.02  # jitter sigma as a fraction of each gene's range
    robustness_min_ratio: float = 0.70  # raised from 0.50: genome must keep 70% fitness after 2% param nudge
    multi_window_slices: int = 3  # contiguous IS sub-windows scored for consistency (1 = off)
    min_trades: int = 3  # trade-count penalty anchor (forwarded to metrics.compute_metrics)
    trade_penalty_power: float = 1.5
    oos_candidates: int = 10  # top-N genomes scored on OOS each generation
    use_numba: bool = True  # use the Numba-JIT multithreaded CPU backtester when available
    checkpoint_path: str = "data/ga_checkpoint.json"
    pool_path: str = "data/deployment_pool.json"
    persist_interval_s: float = 5.0  # min seconds between checkpoint/pool disk writes


@dataclass(frozen=True)
class ChampionValidationConfig:
    """Forward-only consistency policy for global champion and deployment selection.

    The GA still optimizes a local walk-forward IS window. This separate layer validates
    candidate genomes only on 30-day periods that finish after their discovery, preventing
    a one-window OOS outlier from being promoted as the global/deployable strategy.
    """

    validation_period_days: int = 30
    validation_step_days: int = 30  # non-overlapping independent periods
    candidates_per_wfo_round: int = 10
    required_periods: int = 6
    minimum_positive_period_ratio: float = 0.75  # raised from 0.60: ≥3/4 periods must profit
    minimum_trades_per_period: int = 3
    maximum_trades_per_period: int = 40  # turnover guard: reject over-trading candidates
    maximum_period_drawdown: float = 0.25  # tightened from 0.30
    minimum_period_sharpe: float = 0.10   # raised from -0.5: every period must show skill
    minimum_period_return: float = -0.05  # new: max -5% per 30d period; stops slow bleeders
    maximum_candidates: int = 100


@dataclass(frozen=True)
class FillModelConfig:
    """Exchange-emulation stress knobs shared by the backtester and the live executor.

    ``mode`` selects a preset: ``"optimistic"`` (legacy behavior: no slippage, no stale-
    order cancel, hold counted from signal), ``"base"`` (realistic: cancel stale orders,
    count hold from fill, 1-tick stop slippage, intrabar liquidation, same-bar stop check)
    and ``"pessimistic"`` (base + extra stop slippage and maker penetration requirement).
    Individual fields can be overridden regardless of mode.
    """

    mode: str = "base"  # "optimistic" | "base" | "pessimistic"
    stop_slippage_ticks: int = 1  # extra ticks paid beyond the stop trigger price
    stop_gap_through: bool = True  # if a bar opens beyond the stop, fill at the open
    maker_penetration_ticks: int = 1  # price must cross the limit by >= N ticks to fill
    cancel_stale_orders: bool = True  # flat/rejected signal cancels a resting entry
    max_hold_from_fill: bool = True  # time stop measured from the fill bar, not signal
    same_bar_stop_check: bool = True  # check SL/TP on the bar that fills the entry
    intrabar_liquidation: bool = True  # margin checked against the bar's adverse extreme
    taker_slippage_ticks: int = 0  # extra adverse ticks on taker (market) exits

    @classmethod
    def preset(cls, mode: str) -> "FillModelConfig":
        """Build a named fill-model preset (used by stress tests and the demo)."""
        if mode == "optimistic":
            return cls(mode=mode, stop_slippage_ticks=0, stop_gap_through=False,
                       maker_penetration_ticks=1, cancel_stale_orders=False,
                       max_hold_from_fill=False, same_bar_stop_check=False,
                       intrabar_liquidation=False, taker_slippage_ticks=0)
        if mode == "pessimistic":
            return cls(mode=mode, stop_slippage_ticks=3, stop_gap_through=True,
                       maker_penetration_ticks=2, cancel_stale_orders=True,
                       max_hold_from_fill=True, same_bar_stop_check=True,
                       intrabar_liquidation=True, taker_slippage_ticks=1)
        return cls(mode="base")


@dataclass(frozen=True)
class WFOConfig:
    """Walk-forward optimization schedule and window sizes.

    A normal walk-forward pass runs every window exactly once, with
    ``generations_per_window`` generations in each window.
    """

    is_days: int = 90
    oos_days: int = 30
    step_days: int = 30  # rolling step (default == oos_days)
    generations_per_window: int = 20
    holdout_days: int = 0  # final days reserved from WFO windows (0 = no holdout)


@dataclass(frozen=True)
class RiskConfig:
    """Risk guardrails for the live/paper executor."""

    max_leverage: float = 4.0  # hard cap (mirrors InstrumentSpec)
    target_leverage: float = 3.0  # default deploy leverage (<= max)
    drawdown_breaker_pct: float = 22.0  # unrealized DD within window -> halt
    breaker_window_hours: int = 24
    enforce_breaker_in_backtest: bool = False  # mirror the live breaker during training
    initial_equity: float = 10_000.0  # starting account equity (USD) for backtest/paper


@dataclass(frozen=True)
class UIConfig:
    """Web dashboard host/port."""

    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class LiveConfig:
    """Live/paper execution config. Secrets are env-only (see env.example)."""

    mode: str = "paper"  # "paper" (default) | "live"
    api_key_name_env: str = "CDP_API_KEY_NAME"
    api_key_private_env: str = "CDP_API_KEY_PRIVATE_KEY"
    heartbeat_timeout_s: float = 30.0
    reconnect_base_s: float = 1.0
    reconnect_max_s: float = 60.0


@dataclass(frozen=True)
class Config:
    """Aggregate root config passed between sub-systems."""

    instrument: InstrumentSpec = field(default_factory=InstrumentSpec)
    data: DataConfig = field(default_factory=DataConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    fills: FillModelConfig = field(default_factory=FillModelConfig)
    champion_validation: ChampionValidationConfig = field(default_factory=ChampionValidationConfig)
    wfo: WFOConfig = field(default_factory=WFOConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    live: LiveConfig = field(default_factory=LiveConfig)


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Best-effort load of a local .env file (never required)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - python-dotenv absent/optional
        pass


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    try:
        return int(env[key])
    except (KeyError, ValueError):
        return default


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env[key])
    except (KeyError, ValueError):
        return default


def default_config() -> Config:
    """Return the all-defaults Config (no env overrides)."""
    return Config()


def build_config(env: Mapping[str, str] | None = None) -> Config:
    """Build a Config from defaults + environment variables.

    ``env`` defaults to ``os.environ`` (a ``.env`` file is loaded best-effort first).
    Only a few safe knobs are env-overridable; secrets are read by the live layer
    directly, never surfaced into Config.
    """
    _load_dotenv()
    e = dict(os.environ) if env is None else dict(env)

    ui = UIConfig(host=e.get("UI_HOST", UIConfig.host), port=_env_int(e, "UI_PORT", UIConfig.port))
    live = LiveConfig(mode=e.get("TRADING_MODE", LiveConfig.mode))
    data = DataConfig(global_seed=_env_int(e, "GLOBAL_SEED", DataConfig.global_seed))
    return Config(ui=ui, live=live, data=data)


# ---------------------------------------------------------------------------
# Module-level convenience: default config available without any I/O
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Config = default_config()
