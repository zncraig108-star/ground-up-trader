"""core/strategy.py — genome -> signals (the concrete strategy template).

Implements the single well-defined strategy family from DEEPSEEK_BUILD_PROMPT.md §4,
parameterized by a ``Genome``. All signal computation is fully vectorized over the
population axis and consumes only *fully-closed* signal candles.

Strategy rule (documented exactly as implemented)
-------------------------------------------------
On the close of signal candle ``t`` (data ``<= t`` only):

* **Regime** ``d_t`` in ``{+1 long, -1 short, 0 flat}``:
    - ``regime_mode == EMA``: ``+1`` if ``EMA_fast > EMA_slow`` AND the fast EMA is
      rising (``EMA_fast[t] > EMA_fast[t-1]``); ``-1`` if ``EMA_fast < EMA_slow`` AND
      falling; ``0`` otherwise (ASSUMPTION: sideways => flat, no entry).
    - ``regime_mode == SUPERTREND``: ``+1`` if Supertrend is bullish, ``-1`` if bearish
      (never flat).
* **Entry (post-only maker limit)**, only in the regime's direction:
    - long regime  -> resting BUY  at ``close[t] - entry_offset_atr_mult * ATR[t]``
    - short regime -> resting SELL at ``close[t] + entry_offset_atr_mult * ATR[t]``
    - flat         -> no order.

The pullback condition of the spec is encoded by placing the limit *below* (long) /
*above* (short) the close: it only fills if price retraces to it. The limit acts on the
next hour's 1m candles (``T+1`` window); that shift is applied structurally by the
backtester, not here.

Exits (take-profit / stop-loss / time stop) are driven by ``tp_atr_mult``,
``sl_atr_mult`` and ``max_hold_bars`` at fill time in the backtester; the signals expose
the ``atr`` needed to size those distances.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, List

from config import GENES, GENE_NAMES, REGIME_EMA, REGIME_SUPERTREND
from core import indicators
from core.xp import xp

__all__ = [
    "Genome",
    "SignalFeatures",
    "SignalIntents",
    "GENE_INDEX",
    "encode",
    "decode",
    "random_genome",
    "random_population",
    "clamp_genomes",
    "round_integer_genes",
    "compute_signals",
]

GENE_INDEX = {name: i for i, name in enumerate(GENE_NAMES)}


@dataclass(frozen=True)
class Genome:
    """Parameter vector for the strategy template (field order == ``GENES``)."""

    ema_fast: float
    ema_slow: float
    st_period: float
    st_mult: float
    atr_period: float
    entry_offset_atr_mult: float
    tp_atr_mult: float
    sl_atr_mult: float
    max_hold_bars: float
    target_leverage: float
    regime_mode: int
    risk_per_trade_pct: float = 0.015
    trailing_atr_mult: float = 2.0
    funding_filter_z: float = 2.0
    volume_filter_z: float = 0.0

    def encode(self) -> np.ndarray:
        """Encode to a flat float64 array in canonical gene order."""
        return np.asarray(
            [
                self.ema_fast,
                self.ema_slow,
                self.st_period,
                self.st_mult,
                self.atr_period,
                self.entry_offset_atr_mult,
                self.tp_atr_mult,
                self.sl_atr_mult,
                self.max_hold_bars,
                self.target_leverage,
                float(self.regime_mode),
                self.risk_per_trade_pct,
                self.trailing_atr_mult,
                self.funding_filter_z,
                self.volume_filter_z,
            ],
            dtype=np.float64,
        )

    @classmethod
    def decode(cls, arr: Any) -> "Genome":
        """Decode a flat array (numpy or xp) into a Genome."""
        a = np.asarray(arr, dtype=np.float64).ravel()
        return cls(
            ema_fast=float(a[0]),
            ema_slow=float(a[1]),
            st_period=float(a[2]),
            st_mult=float(a[3]),
            atr_period=float(a[4]),
            entry_offset_atr_mult=float(a[5]),
            tp_atr_mult=float(a[6]),
            sl_atr_mult=float(a[7]),
            max_hold_bars=float(a[8]),
            target_leverage=float(a[9]),
            regime_mode=int(round(a[10])),
            risk_per_trade_pct=float(a[11]) if len(a) > 11 else 0.015,
            trailing_atr_mult=float(a[12]) if len(a) > 12 else 2.0,
            funding_filter_z=float(a[13]) if len(a) > 13 else 2.0,
            volume_filter_z=float(a[14]) if len(a) > 14 else 0.0,
        )


def encode(genome: Genome) -> np.ndarray:
    """Flat-array encoding of a single genome (see ``Genome.encode``)."""
    return genome.encode()


def decode(arr: Any) -> Genome:
    """Flat-array -> Genome (see ``Genome.decode``)."""
    return Genome.decode(arr)


def _sample_gene(rng: np.random.Generator, g: Any) -> float:
    """Sample one gene uniformly from its range; discrete/integer genes are integral."""
    if g.discrete:
        return float(rng.integers(int(g.low), int(g.high) + 1))
    if g.integer:
        return float(rng.integers(int(g.low), int(g.high) + 1))
    return float(rng.uniform(g.low, g.high))


def random_genome(rng: np.random.Generator) -> Genome:
    """Draw a uniformly random genome from the configured gene ranges."""
    vals = [_sample_gene(rng, g) for g in GENES]
    return Genome.decode(np.asarray(vals, dtype=np.float64))


def random_population(rng: np.random.Generator, size: int) -> np.ndarray:
    """Draw ``size`` random genomes as a ``(size, NUM_GENES)`` float64 array."""
    return np.stack([random_genome(rng).encode() for _ in range(size)])


def clamp_genomes(arr: Any) -> np.ndarray:
    """Clamp a ``(P, NUM_GENES)`` genome batch back into valid ranges.

    Continuous genes are clipped to ``[low, high]``; discrete and integer genes are also
    rounded to integers (so indicator periods are reproducible settings, not fractions).
    Returns a new numpy float64 array (never mutates the input in place).
    """
    a = np.asarray(arr, dtype=np.float64)
    out = a.copy()
    for i, g in enumerate(GENES):
        out[:, i] = np.clip(out[:, i], g.low, g.high)
        if g.discrete or g.integer:
            out[:, i] = np.rint(out[:, i])
    return out


def round_integer_genes(genome_batch: Any) -> Any:
    """Round integer-flagged genes at use-site (defensive; the GA already keeps them
    integral). Period genes must be whole bars so an optimized genome is reproducible."""
    a = xp.asarray(genome_batch, dtype=xp.float64)
    out = a
    for i, g in enumerate(GENES):
        if g.integer or g.discrete:
            col = xp.rint(a[:, i])
            if out is a:
                out = xp.array(a, dtype=xp.float64)
            out[:, i] = col
    return out


@dataclass(frozen=True)
class SignalFeatures:
    """Closed-candle 1h feature bundle consumed by ``compute_signals``."""

    open: Any  # (N,)
    high: Any  # (N,)
    low: Any  # (N,)
    close: Any  # (N,)
    volume_zscore: Any = None  # (N,) causal volume z-score
    funding_zscore: Any = None  # (N,) causal funding rate z-score


@dataclass
class SignalIntents:
    """Per-genome, per-bar signal output (all arrays are ``(P, N)``).

    ``entry_price`` is NaN where there is no entry intent; ``long_entry`` /
    ``short_entry`` are boolean masks over the ``P`` axis.
    """

    regime: Any  # float64: +1 long-regime, -1 short-regime, 0 flat
    atr: Any  # float64: ATR used to size entry/TP/SL distances
    long_entry: Any  # bool
    short_entry: Any  # bool
    entry_price: Any  # float64 limit price (NaN where no entry)


def compute_signals(features: SignalFeatures, genome_batch: Any,
                    progress_cb: Any = None) -> SignalIntents:
    """Compute entry/regime intents for a whole population, vectorized over ``P``.

    Args:
        features: ``SignalFeatures`` of closed 1h candles (shape ``(N,)`` each).
        genome_batch: ``(P, NUM_GENES)`` float64 array (numpy or xp).
        progress_cb: optional ``cb(fraction)`` for live progress (fired inside the
            indicator time loops, so the UI animates smoothly during this phase).

    Returns:
        ``SignalIntents`` with ``(P, N)`` arrays. No value here depends on data after
        its own bar index (no lookahead); the ``T -> T+1`` action shift is applied by
        the backtester.
    """
    close = xp.asarray(features.close, dtype=xp.float64)
    high = xp.asarray(features.high, dtype=xp.float64)
    low = xp.asarray(features.low, dtype=xp.float64)
    # Period genes are whole bars: round at use-site so a genome is reproducible.
    g = round_integer_genes(genome_batch)

    ema_fast = g[:, GENE_INDEX["ema_fast"]]
    ema_slow = g[:, GENE_INDEX["ema_slow"]]
    st_period = g[:, GENE_INDEX["st_period"]]
    st_mult = g[:, GENE_INDEX["st_mult"]]
    atr_period = g[:, GENE_INDEX["atr_period"]]
    entry_offset = g[:, GENE_INDEX["entry_offset_atr_mult"]]
    mode = g[:, GENE_INDEX["regime_mode"]]

    n = close.shape[0]
    p = g.shape[0]

    # Progress: split [0,1] across the four passes (ema_fast, ema_slow, atr, supertrend).
    pe = max(1, n // 60)
    def _sub(base: float, span: float):
        if progress_cb is None:
            return None
        return lambda t, total: progress_cb(base + span * (t / total if total else 1.0))

    atr = indicators.atr_batch(high, low, close, atr_period, _sub(0.50, 0.20), pe)  # (P, N)

    # --- EMA regime (flat allowed) -------------------------------------------
    ema_f = indicators.ema_batch(close, ema_fast, _sub(0.0, 0.25), pe)
    ema_s = indicators.ema_batch(close, ema_slow, _sub(0.25, 0.25), pe)
    rising = xp.zeros((p, n), dtype=xp.bool_)
    falling = xp.zeros((p, n), dtype=xp.bool_)
    if n > 1:
        rising[:, 1:] = ema_f[:, 1:] > ema_f[:, :-1]
        falling[:, 1:] = ema_f[:, 1:] < ema_f[:, :-1]

    long_ema = (ema_f > ema_s) & rising
    short_ema = (ema_f < ema_s) & falling
    ema_regime = xp.where(long_ema, 1.0, xp.where(short_ema, -1.0, 0.0))

    # --- Supertrend regime (always directional) -------------------------------
    st_regime = indicators.supertrend_batch(
        high, low, close, st_period, st_mult, _sub(0.70, 0.30), pe).astype(xp.float64)

    # Select regime per genome's discrete mode toggle.
    is_ema = (mode == REGIME_EMA)[:, None]  # (P, 1)
    regime = xp.where(is_ema, ema_regime, st_regime)  # (P, N) float

    # --- Volume & Funding gates -----------------------------------------------
    vol_z = xp.zeros(n, dtype=xp.float64) if features.volume_zscore is None else xp.nan_to_num(xp.asarray(features.volume_zscore, dtype=xp.float64), nan=0.0)
    fund_z = xp.zeros(n, dtype=xp.float64) if features.funding_zscore is None else xp.nan_to_num(xp.asarray(features.funding_zscore, dtype=xp.float64), nan=0.0)

    vol_gate = g[:, GENE_INDEX["volume_filter_z"]]
    fund_gate = g[:, GENE_INDEX["funding_filter_z"]]

    vol_pass = vol_z[None, :] >= vol_gate[:, None]
    long_fund_pass = fund_z[None, :] <= fund_gate[:, None]
    short_fund_pass = fund_z[None, :] >= -fund_gate[:, None]

    long_entry = (regime > 0.5) & vol_pass & long_fund_pass
    short_entry = (regime < -0.5) & vol_pass & short_fund_pass

    # --- Entry limits ----------------------------------------------------------
    # long: buy at close - offset*ATR ; short: sell at close + offset*ATR.
    # Where filters reject or regime is flat: NaN
    entry_price = xp.full((p, n), xp.nan, dtype=xp.float64)
    entry_price = xp.where(long_entry, close[None, :] - entry_offset[:, None] * atr, entry_price)
    entry_price = xp.where(short_entry, close[None, :] + entry_offset[:, None] * atr, entry_price)

    return SignalIntents(
        regime=regime,
        atr=atr,
        long_entry=long_entry,
        short_entry=short_entry,
        entry_price=entry_price,
    )
