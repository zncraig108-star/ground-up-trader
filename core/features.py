"""core/features.py — causal 1h feature extraction from closed candles + funding.

Role
----
Turn a validated 1h OHLCV frame (see ``core.data``) plus an hourly funding series
into a ``FeatureSet`` of plain ``numpy.float64`` arrays aligned 1:1 to the 1h index.
These features are an additional, regime-independent signal source that the strategy
layer may consume alongside the indicator kernels in ``core.indicators``.

Causality contract
------------------
Every value at bar ``t`` is computed from data at bars ``<= t`` only. The current bar
is treated as a fully closed observation: its own close/high/low/volume may take part
in a statistic ("the current observation"), but no future bar ever does. All rolling
statistics use pandas ``.rolling`` (backward-looking by construction) and, where the
current observation must not inflate its own baseline, a one-bar ``.shift(1)`` — that
choice is documented per feature in ``FeatureSet``. No centered windows, no negative
shifts, no ``bfill``.

Warm-up
-------
Each feature has a documented leading NaN warm-up (see ``compute_features``). Beyond
that warm-up all values are finite; the strategy layer is responsible for masking the
warm-up region before use. Inputs are validated finite, which preserves this contract.

Backend
-------
The 1h series are short (a few thousand bars), so this module uses NumPy/pandas
directly and never touches the ``xp`` GPU backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "FeatureConfig",
    "FeatureSet",
    "FEATURE_NAMES",
    "compute_features",
    "feature_matrix",
]

_EPS = 1e-12

# Stable, documented column order for ``feature_matrix``. Must match FeatureSet fields.
FEATURE_NAMES: tuple[str, ...] = (
    "realized_vol",
    "volume_zscore",
    "range_position",
    "trend_strength",
    "funding_rate",
    "funding_zscore",
    "tod_sin",
    "tod_cos",
)


@dataclass(frozen=True)
class FeatureConfig:
    """Rolling-window lengths in 1h bars for ``compute_features``.

    Attributes:
        vol_window: Lookback for realized volatility (log-return std).
        trend_window: EMA span and slope lookback for trend strength.
        range_window: Lookback for the trailing high/low range.
        zscore_window: Baseline lookback for the volume z-score.
        funding_window: Baseline lookback for the funding z-score.
    """

    vol_window: int = 24
    trend_window: int = 72
    range_window: int = 48
    zscore_window: int = 168
    funding_window: int = 72


@dataclass(frozen=True)
class FeatureSet:
    """Causal 1h feature bundle aligned 1:1 to the input 1h index.

    All fields are plain ``(N,)`` numpy float64 arrays. Leading NaN regions are the
    documented warm-ups; the tails are finite. Field order matches ``FEATURE_NAMES``.

    Attributes:
        realized_vol: Rolling std (ddof=1) of 1h log returns over ``vol_window`` bars,
            including the current closed bar's return. Units are per-hour (not
            annualized). Warm-up: ``vol_window`` bars.
        volume_zscore: Z-score of volume against the previous ``zscore_window`` bars
            (baseline shifted by one bar, so the observation does not deflate its own
            denominator); 0.0 when that baseline std is 0. Warm-up:
            ``zscore_window`` bars.
        range_position: Close position inside the trailing ``range_window`` high/low
            range, mapped to ``[-1, 1]`` (+1 at the range high, -1 at the low); the
            current bar's high/low are included; 0.0 for a zero-width range. Warm-up:
            ``range_window - 1`` bars.
        trend_strength: Normalized EMA slope
            ``(ema[t] / ema[t - trend_window] - 1) / trend_window`` where ``ema`` is
            the ``trend_window``-span EMA of close. Units: fractional price change per
            bar (0.001 => +0.1% per bar). Warm-up: ``trend_window`` bars.
        funding_rate: Funding rate aligned to the 1h index by forward-fill; a missing
            rate (including before the first record) is 0.0. No warm-up.
        funding_zscore: Z-score of ``funding_rate`` against the previous
            ``funding_window`` bars (baseline shifted by one bar); 0.0 when that
            baseline std is 0. Warm-up: ``funding_window`` bars.
        tod_sin: ``sin(2*pi*hour/24)`` of the bar's UTC timestamp hour. No warm-up.
        tod_cos: ``cos(2*pi*hour/24)`` of the bar's UTC timestamp hour. No warm-up.
    """

    realized_vol: np.ndarray
    volume_zscore: np.ndarray
    range_position: np.ndarray
    trend_strength: np.ndarray
    funding_rate: np.ndarray
    funding_zscore: np.ndarray
    tod_sin: np.ndarray
    tod_cos: np.ndarray


def _as_float_series(ohlcv_1h: pd.DataFrame, column: str) -> pd.Series:
    """Extract one OHLCV column as a finite float64 Series (raise on bad values)."""
    if column not in ohlcv_1h.columns:
        raise ValueError(f"ohlcv_1h: missing required column {column!r}")
    values = pd.to_numeric(ohlcv_1h[column], errors="coerce").astype(np.float64)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"ohlcv_1h[{column!r}] contains non-finite values")
    return values


def _backward_zscore(values: pd.Series, window: int) -> pd.Series:
    """Z-score of each observation against the trailing ``window`` bars before it.

    The baseline excludes the current bar (``.shift(1)``) so the observation cannot
    deflate its own denominator. A zero baseline std yields 0.0 after warm-up; the
    warm-up region itself stays NaN.
    """
    baseline = values.rolling(window, min_periods=window)
    mean = baseline.mean().shift(1)
    std = baseline.std(ddof=1).shift(1)
    denom = std.where(std > 0.0)
    z = ((values - mean) / denom).fillna(0.0)
    z[std.isna()] = np.nan
    return z.astype(np.float64)


def _align_funding(funding: pd.DataFrame | pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Align a funding series to the 1h index (forward-fill, missing -> 0.0)."""
    if isinstance(funding, pd.DataFrame):
        if "funding_rate" not in funding.columns:
            raise ValueError("funding: missing required column 'funding_rate'")
        series = pd.to_numeric(funding["funding_rate"], errors="coerce").astype(np.float64)
    else:
        series = pd.to_numeric(pd.Series(funding), errors="coerce").astype(np.float64)

    if series.index.tz is not None:
        series.index = series.index.tz_convert("UTC").tz_localize(None)
    if series.index.has_duplicates:
        series = series[~series.index.duplicated(keep="last")]
    series = series.sort_index()

    # ASSUMPTION: a bar with no published funding rate (or with no funding data at all
    # before the first record) is treated as 0.0 rather than dropped, so the feature is
    # defined from the first 1h bar; forward-fill propagates the last settled rate.
    return series.reindex(index).ffill().fillna(0.0).astype(np.float64)


def compute_features(ohlcv_1h: pd.DataFrame, funding: pd.DataFrame | pd.Series,
                     config: FeatureConfig | None = None) -> FeatureSet:
    """Compute the causal 1h feature bundle from closed candles and funding.

    Args:
        ohlcv_1h: Validated 1h OHLCV frame with a naive-UTC ``DatetimeIndex`` and
            columns ``open, high, low, close, volume`` (as returned by ``core.data``).
        funding: Hourly funding series: a DataFrame with a ``funding_rate`` column or
            a Series of per-interval rates. Reindexed to the 1h index, forward-filled,
            missing -> 0.0.
        config: Window lengths; defaults to ``FeatureConfig()``.

    Returns:
        ``FeatureSet`` of ``(N,)`` float64 arrays aligned to ``ohlcv_1h.index``.

    Leading NaN warm-ups (bars) per feature:
        realized_vol ``vol_window``; volume_zscore ``zscore_window``; range_position
        ``range_window - 1``; trend_strength ``trend_window``; funding_zscore
        ``funding_window``; funding_rate, tod_sin, tod_cos none.
    """
    cfg = config if config is not None else FeatureConfig()
    for name in ("vol_window", "trend_window", "range_window", "zscore_window",
                 "funding_window"):
        if int(getattr(cfg, name)) < 1:
            raise ValueError(f"FeatureConfig.{name} must be >= 1")

    if not isinstance(ohlcv_1h.index, pd.DatetimeIndex):
        raise ValueError("ohlcv_1h must have a DatetimeIndex (naive UTC)")

    index = ohlcv_1h.index
    close = _as_float_series(ohlcv_1h, "close")
    high = _as_float_series(ohlcv_1h, "high")
    low = _as_float_series(ohlcv_1h, "low")
    volume = _as_float_series(ohlcv_1h, "volume")
    funding_aligned = _align_funding(funding, index)

    # Realized vol: uses returns up to and including the current closed bar (allowed;
    # it is the current observation). First valid at t == vol_window.
    log_ret = np.log(close).diff()
    realized_vol = log_ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=1)

    # Volume z-score: backward baseline excludes the current observation.
    volume_zscore = _backward_zscore(volume, cfg.zscore_window)

    # Range position: trailing high/low includes the current closed bar.
    high_max = high.rolling(cfg.range_window, min_periods=cfg.range_window).max()
    low_min = low.rolling(cfg.range_window, min_periods=cfg.range_window).min()
    span = high_max - low_min
    denom = span.where(span > 0.0)
    range_position = ((close - low_min) / denom * 2.0 - 1.0).fillna(0.0).clip(-1.0, 1.0)
    range_position[span.isna()] = np.nan

    # Trend strength: EMA slope in fractional price change per bar. ASSUMPTION: the EMA
    # is seeded at the first close (pandas ``adjust=False``), matching the warm-up
    # convention of ``core.indicators``; slope is valid from t == trend_window.
    ema = close.ewm(span=cfg.trend_window, adjust=False).mean()
    trend_strength = (ema / ema.shift(cfg.trend_window) - 1.0) / float(cfg.trend_window)

    funding_zscore = _backward_zscore(funding_aligned, cfg.funding_window)

    # Time-of-day session encoding from the bar's UTC hour (known at bar close).
    hours = index.hour.to_numpy(dtype=np.float64)
    angle = 2.0 * np.pi * hours / 24.0
    tod_sin = np.sin(angle)
    tod_cos = np.cos(angle)

    return FeatureSet(
        realized_vol=realized_vol.to_numpy(dtype=np.float64),
        volume_zscore=volume_zscore.to_numpy(dtype=np.float64),
        range_position=range_position.to_numpy(dtype=np.float64),
        trend_strength=trend_strength.to_numpy(dtype=np.float64),
        funding_rate=funding_aligned.to_numpy(dtype=np.float64),
        funding_zscore=funding_zscore.to_numpy(dtype=np.float64),
        tod_sin=tod_sin,
        tod_cos=tod_cos,
    )


def feature_matrix(fs: FeatureSet) -> np.ndarray:
    """Stack a ``FeatureSet`` into a ``(F, N)`` float64 matrix.

    ``F == len(FEATURE_NAMES)`` and rows follow ``FEATURE_NAMES`` exactly.
    """
    return np.stack(
        [np.asarray(getattr(fs, name), dtype=np.float64) for name in FEATURE_NAMES],
        axis=0,
    )
