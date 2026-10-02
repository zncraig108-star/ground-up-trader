"""core/data.py — data loading + deterministic synthetic generation.

Responsibilities
----------------
1. ``load_ohlcv`` / ``load_funding`` — read Parquet (via pyarrow) or CSV with a strict,
   documented schema and validation (monotonic timestamps, no duplicates, gap handling
   with an explicit flag + warning).
2. ``load_market_data`` — the real-data entry point: load a 1m OHLCV file, resample the
   1h signal series from it (so the two timeframes always align), and align a real
   funding series onto the same index. ``write_parquet_bundle`` / ``data_manifest`` cache
   and fingerprint a bundle for reproducible runs.
3. ``fetch_coinbase`` — optional REST fetcher for real Coinbase 1h/1m candles + funding,
   cached to Parquet on disk. Network/credentials are OPTIONAL; the offline demo never
   calls this.
4. ``make_synthetic`` — deterministic generator of correlated 1m + 1h OHLCV and an
   hourly funding series, so the whole system runs with zero external dependencies.

Schema (enforced + documented)
------------------------------
    ohlcv :  timestamp (UTC), open, high, low, close, volume
    funding: timestamp (UTC), funding_rate  (per-interval rate)

DataFrames returned by the loader/generator use a ``DatetimeIndex`` (naive UTC, name
"timestamp") and, for OHLCV, columns ``[open, high, low, close, volume]``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from config import DataConfig

__all__ = [
    "OHLCV_COLUMNS",
    "FUNDING_COLUMNS",
    "SyntheticData",
    "load_ohlcv",
    "load_funding",
    "load_market_data",
    "write_parquet_bundle",
    "data_manifest",
    "fetch_coinbase",
    "make_synthetic",
    "parse_year_spec",
    "slice_year",
    "available_years_for_data",
    "get_or_create_year_data",
    "get_or_create_all_time_data",
]

logger = logging.getLogger("gt.core.data")

OHLCV_COLUMNS: Sequence[str] = ("open", "high", "low", "close", "volume")
FUNDING_COLUMNS: Sequence[str] = ("funding_rate",)
_RAW_OHLCV_COLUMNS: Sequence[str] = ("timestamp", "open", "high", "low", "close", "volume")
_RAW_FUNDING_COLUMNS: Sequence[str] = ("timestamp", "funding_rate")

# Timeframe strings use the compact "1m"/"1h"/"1d" form, but pandas >= 2.2 treats
# bare "m" as *month* (not minute). Map explicitly to avoid silently resampling by
# months (which also overflows the default datetime64 bounds).
_FREQ_MAP = {"1m": "1min", "1h": "1h", "1d": "1D", "1w": "1W"}


def _pandas_freq(timeframe: str) -> str:
    """Translate a compact timeframe token to a pandas frequency alias."""
    return _FREQ_MAP.get(timeframe, timeframe)


@dataclass(frozen=True)
class SyntheticData:
    """Deterministic synthetic market bundle.

    ``ohlcv_1h`` is resampled from ``ohlcv_1m`` so the two timeframes are guaranteed to
    align: each 1h candle (timestamped at the hour end) aggregates exactly the 60
    preceding 1m candles. ``funding`` is indexed at the same hourly boundaries.
    """

    ohlcv_1m: pd.DataFrame
    ohlcv_1h: pd.DataFrame
    funding: pd.DataFrame


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------

def _read_frame(path: str) -> pd.DataFrame:
    """Read a Parquet or CSV file based on its extension."""
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    # Fall back to pandas' inference (assume CSV) with a warning.
    logger.warning("Unrecognized extension for %s; assuming CSV.", path)
    return pd.read_csv(path)


def _normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse/normalize the 'timestamp' column to a naive-UTC DatetimeIndex.

    Accepts ISO-8601 strings or integer epoch milliseconds (spec §3.3). Naive UTC is
    used internally to avoid timezone-object overhead across the vectorized pipeline.
    """
    ts = df["timestamp"]
    if np.issubdtype(ts.dtype, np.integer) or np.issubdtype(ts.dtype, np.floating):
        # Assume epoch milliseconds.
        ts = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        ts = pd.to_datetime(ts, utc=True)
    # Drop the timezone (values are already UTC); keep the datetime as naive UTC.
    ts = ts.dt.tz_localize(None)
    df = df.drop(columns=["timestamp"])
    df.index = pd.DatetimeIndex(ts, name="timestamp")
    return df


def _validate_ohlcv(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Enforce the OHLCV schema and column set."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing OHLCV column(s) {missing}")
    # Basic sanity: high >= low, prices > 0.
    bad = (df["high"] < df["low"]) | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        raise ValueError(f"{source}: {int(bad.sum())} bar(s) with high<low or non-positive price")
    return df[list(OHLCV_COLUMNS)]


def _enforce_regularity(df: pd.DataFrame, timeframe: str, allow_ffill: bool,
                        source: str) -> pd.DataFrame:
    """Enforce monotonic increasing timestamps, no duplicates, and detect gaps.

    On gaps: if ``allow_ffill`` is True, reindex to the expected frequency and
    forward-fill prices (volume zero-filled) with a warning; otherwise raise.
    """
    idx = df.index
    if not idx.is_monotonic_increasing:
        df = df.sort_index()
        logger.warning("%s: timestamps were not monotonic; sorted.", source)
    if idx.has_duplicates:
        raise ValueError(f"{source}: duplicate timestamps detected")

    expected = pd.date_range(idx[0], idx[-1], freq=_pandas_freq(timeframe))
    missing = expected.difference(idx)
    if len(missing) == 0:
        return df

    if not allow_ffill:
        raise ValueError(f"{source}: {len(missing)} gaps detected and forward-fill disabled")

    logger.warning("%s: %d gap(s) detected; forward-filling prices (volume=0).",
                   source, len(missing))
    df = df.reindex(expected)
    for col in OHLCV_COLUMNS:
        if col == "volume":
            df[col] = df[col].fillna(0.0)
        else:
            df[col] = df[col].ffill()  # ASSUMPTION: gap bars carry last known price
    df.index.name = "timestamp"
    return df


def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Accept files that store the timestamps as a DatetimeIndex (Parquet round-trip)."""
    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.rename_axis("timestamp").reset_index()
        else:
            raise ValueError("missing required column 'timestamp'")
    return df


def load_ohlcv(path: str, timeframe: str = "1h",
               allow_ffill: Optional[bool] = None) -> pd.DataFrame:
    """Load an OHLCV file (Parquet or CSV) into a validated DataFrame.

    Returns a DataFrame with a naive-UTC ``DatetimeIndex`` (name "timestamp") and
    columns ``[open, high, low, close, volume]``.
    """
    df = _ensure_timestamp_column(_read_frame(path))
    missing = [c for c in _RAW_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")
    df = _normalize_timestamps(df)
    df = _validate_ohlcv(df, path)
    allow = allow_ffill if allow_ffill is not None else DataConfig.allow_gap_ffill
    return _enforce_regularity(df, timeframe, allow, path)


def load_funding(path: str, timeframe: str = "1h") -> pd.DataFrame:
    """Load a funding-rate file (Parquet or CSV) into a validated DataFrame.

    Returns a DataFrame with a naive-UTC ``DatetimeIndex`` (name "timestamp") and a
    single column ``funding_rate`` (per-interval rate).
    """
    df = _ensure_timestamp_column(_read_frame(path))
    missing = [c for c in _RAW_FUNDING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")
    df = _normalize_timestamps(df)
    return df[list(FUNDING_COLUMNS)]


# ---------------------------------------------------------------------------
# Real-data bundle entry points
# ---------------------------------------------------------------------------

def _resample_signals(ohlcv_1m: pd.DataFrame, signal_timeframe: str) -> pd.DataFrame:
    """Resample 1m OHLCV into the signal timeframe (right-closed, right-labeled).

    ``closed='right', label='right'`` means the signal candle timestamped at ``T``
    aggregates the ``bars_per_hour`` 1m candles ending at ``T``; the backtester's
    ``T -> T+1`` shift is what keeps the strategy causal.
    """
    return ohlcv_1m.resample(_pandas_freq(signal_timeframe), closed="right",
                             label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )


def load_market_data(ohlcv_1m_path: str, funding_path: Optional[str] = None,
                     signal_timeframe: str = "1h", fill_timeframe: str = "1m",
                     allow_ffill: Optional[bool] = None) -> "SyntheticData":
    """Load a real market bundle from files (the real-data entry point).

    The 1h signal series is **resampled from the 1m file**, so the two timeframes are
    always aligned (``H == M // bars_per_hour``). Funding is loaded separately, aligned
    onto the 1h index (forward-fill) and zero-filled only for uncovered leading/trailing
    bars with a warning — a funding file that shares no timestamps with the 1h series is
    rejected (never silently zero).

    Raises:
        ValueError: 1m row count is not a whole number of 1h bars, or the funding series
            does not overlap the 1h index / has no parsable timestamps.
    """
    ohlcv_1m = load_ohlcv(ohlcv_1m_path, fill_timeframe, allow_ffill)
    minutes = int(pd.Timedelta(_pandas_freq(signal_timeframe)) / pd.Timedelta(minutes=1))
    if len(ohlcv_1m) % minutes != 0:
        trimmed = (len(ohlcv_1m) // minutes) * minutes
        logger.warning("%s: trimming %d trailing bar(s) to align to %d-minute signal bars",
                       ohlcv_1m_path, len(ohlcv_1m) - trimmed, minutes)
        ohlcv_1m = ohlcv_1m.iloc[:trimmed]
    ohlcv_1h = _resample_signals(ohlcv_1m, signal_timeframe)

    expected_h = len(ohlcv_1m) // minutes
    if len(ohlcv_1h) != expected_h:
        raise ValueError(
            f"{ohlcv_1m_path}: resample produced {len(ohlcv_1h)} signal bars, "
            f"expected {expected_h} (misaligned timestamps)")

    if funding_path:
        raw = load_funding(funding_path)
        if raw.index.intersection(ohlcv_1h.index).empty:
            raise ValueError(f"{funding_path}: funding has no timestamps overlapping the "
                             "1h price series")
        aligned = raw["funding_rate"].reindex(ohlcv_1h.index).ffill()
        if bool(aligned.isna().any()):
            logger.warning("%s: funding series does not cover the whole 1h price index; "
                           "zero-filled uncovered bars.", funding_path)
        aligned = aligned.fillna(0.0)
        funding = pd.DataFrame({"funding_rate": aligned.to_numpy()}, index=ohlcv_1h.index)
    else:
        logger.warning("load_market_data: no funding_path provided; using zero funding. "
                       "Run decisions on a zero-funding series are not decision-grade.")
        funding = pd.DataFrame({"funding_rate": np.zeros(len(ohlcv_1h))}, index=ohlcv_1h.index)

    return SyntheticData(ohlcv_1m=ohlcv_1m, ohlcv_1h=ohlcv_1h, funding=funding)


def write_parquet_bundle(bundle: "SyntheticData", out_dir: str) -> Dict[str, str]:
    """Persist a bundle as three Parquet files; returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "ohlcv_1m": os.path.join(out_dir, "ohlcv_1m.parquet"),
        "ohlcv_1h": os.path.join(out_dir, "ohlcv_1h.parquet"),
        "funding": os.path.join(out_dir, "funding.parquet"),
    }
    bundle.ohlcv_1m.to_parquet(paths["ohlcv_1m"])
    bundle.ohlcv_1h.to_parquet(paths["ohlcv_1h"])
    bundle.funding.to_parquet(paths["funding"])
    return paths


def _hash_frame(df: pd.DataFrame) -> str:
    """Deterministic sha256 over a frame's timestamps + float64 OHLCV columns."""
    h = hashlib.sha256()
    ns = df.index.values.astype("datetime64[ns]").view("int64")
    h.update(np.ascontiguousarray(ns).tobytes())
    for col in OHLCV_COLUMNS:
        h.update(np.ascontiguousarray(df[col].to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()


def data_manifest(bundle: "SyntheticData") -> Dict[str, Any]:
    """Return a deterministic, JSON-serializable fingerprint of a bundle.

    Used to pin a training run to an exact dataset (coverage, gaps, content hash) so
    results are reproducible and comparable across sessions.
    """
    idx = bundle.ohlcv_1m.index
    step_ns = int(pd.Timedelta(_pandas_freq("1m")).value)  # ASSUMPTION: 1m fill data
    if len(idx) > 1:
        expected_1m = int((idx[-1].value - idx[0].value) // step_ns) + 1
    else:
        expected_1m = len(idx)
    fund = bundle.funding["funding_rate"].to_numpy(dtype=np.float64)
    return {
        "rows_1m": int(len(bundle.ohlcv_1m)),
        "rows_1h": int(len(bundle.ohlcv_1h)),
        "coverage_start": str(bundle.ohlcv_1m.index[0]),
        "coverage_end": str(bundle.ohlcv_1m.index[-1]),
        "expected_1m_bars": expected_1m,
        "missing_1m_bars": int(expected_1m - len(idx)),
        "gap_count": int(expected_1m - len(idx)),
        "sha256_1m": _hash_frame(bundle.ohlcv_1m),
        "sha256_1h": _hash_frame(bundle.ohlcv_1h),
        "funding_rows": int(len(bundle.funding)),
        "funding_mean": float(np.nanmean(fund)) if len(fund) else 0.0,
        "funding_std": float(np.nanstd(fund)) if len(fund) else 0.0,
    }


# ---------------------------------------------------------------------------
# Optional Coinbase fetcher (network + credentials only; never required)
# ---------------------------------------------------------------------------

async def fetch_coinbase(product_id: str = "SOL-PERP", start: Optional[str] = None,
                         end: Optional[str] = None, cache_dir: str = "data",
                         timeframe_1m: str = "1m", timeframe_1h: str = "1h") -> SyntheticData:
    """Fetch real Coinbase 1h + 1m candles and funding, cache to Parquet, and return a
    ``SyntheticData``-shaped bundle.

    This is an OPTIONAL network path. It requires credentials (env keys) and an
    internet connection; the offline demo never calls it. The exact REST endpoints and
    response schemas are unverified and should be confirmed against the live API.
    """
    # ASSUMPTION: Coinbase Advanced Trade exposes public candlestick endpoints under
    # /api/v3/brokerage/products/{product}/candles. VERIFY vs live Coinbase/CDP spec.
    try:
        import aiohttp
    except ImportError as exc:  # pragma: no cover - aiohttp is a required dep
        raise RuntimeError("aiohttp is required for fetch_coinbase") from exc

    os.makedirs(cache_dir, exist_ok=True)
    base = f"https://api.coinbase.com/api/v3/brokerage/products/{product_id}"

    # VERIFY vs live Coinbase/CDP spec: Advanced Trade granularity uses named tokens.
    _granularity = {"1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE",
                    "1h": "ONE_HOUR", "6h": "SIX_HOUR", "1d": "ONE_DAY"}

    async def _page(session: Any, tf: str, page_start: Optional[str],
                    page_end: Optional[str], limit: int = 300) -> list:
        params = {"granularity": _granularity.get(tf, tf), "limit": str(limit)}
        if page_start:
            params["start"] = page_start
        if page_end:
            params["end"] = page_end
        async with session.get(f"{base}/candles", params=params) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        # VERIFY: the candles endpoint returns {"candles": [...]} with newest-first
        # ordering and string OHLCV fields.
        return payload.get("candles", payload if isinstance(payload, list) else [])

    async def _get_candles(tf: str) -> list:
        """Fetch candles, paging backward until the requested range is covered."""
        collected: dict = {}
        cursor_end = end
        max_pages = 200  # hard cap so a bad cursor can never loop forever
        async with aiohttp.ClientSession() as session:
            for _ in range(max_pages):
                rows = await _page(session, tf, start, cursor_end)
                if not rows:
                    break
                fresh = 0
                for r in rows:
                    key = r.get("start")
                    if key is not None and key not in collected:
                        collected[key] = r
                        fresh += 1
                if fresh == 0:
                    break
                earliest = min(r.get("start", "") for r in rows)
                if start is not None and earliest <= start:
                    break
                cursor_end = earliest
        return [collected[k] for k in sorted(collected)]

    candles_1m = await _get_candles(timeframe_1m)
    candles_1h = await _get_candles(timeframe_1h)

    # VERIFY: candle element schema is assumed to be
    # {"start": str, "open": str, "high": str, "low": str, "close": str, "volume": str}.
    def _to_ohlcv(rows: list) -> pd.DataFrame:
        recs = [
            {
                "timestamp": r["start"],
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            }
            for r in rows
        ]
        if not recs:
            raise RuntimeError("fetch_coinbase: no candles returned for the requested range")
        df = pd.DataFrame.from_records(recs)
        return _normalize_timestamps(df)[list(OHLCV_COLUMNS)].sort_index()

    df_1m = _to_ohlcv(candles_1m)
    df_1h = _to_ohlcv(candles_1h)

    async def _get_funding() -> pd.DataFrame:
        """Fetch the funding-rate history and align it to the 1h index.

        VERIFY vs live Coinbase/CDP spec: the funding endpoint and item schema are
        unverified. A missing/malformed response is a hard error — a silent NaN/zero
        funding series would make backtests non-decision-grade.
        """
        params = {"limit": "100"}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base}/funding_rate", params=params) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        items = payload.get("funding_rate") or payload.get("funding_rates") or payload
        if not isinstance(items, list) or not items:
            raise RuntimeError("fetch_coinbase: funding_rate endpoint returned no items "
                               "(VERIFY the endpoint/schema)")
        recs = []
        for it in items:
            ts = it.get("funding_time") or it.get("timestamp") or it.get("time")
            rate = it.get("funding_rate") or it.get("rate")
            if ts is None or rate is None:
                raise RuntimeError("fetch_coinbase: unparsable funding item (VERIFY schema)")
            recs.append({"timestamp": ts, "funding_rate": float(rate)})
        raw = _normalize_timestamps(pd.DataFrame.from_records(recs))
        return pd.DataFrame({"funding_rate": raw["funding_rate"]}, index=raw.index).sort_index()

    raw_funding = await _get_funding()
    aligned = raw_funding["funding_rate"].reindex(df_1h.index).ffill()
    funding = pd.DataFrame({"funding_rate": aligned.to_numpy()}, index=df_1h.index)

    out = SyntheticData(ohlcv_1m=df_1m, ohlcv_1h=df_1h, funding=funding)
    df_1m.to_parquet(os.path.join(cache_dir, "coinbase_1m.parquet"))
    df_1h.to_parquet(os.path.join(cache_dir, "coinbase_1h.parquet"))
    funding.to_parquet(os.path.join(cache_dir, "coinbase_funding.parquet"))
    return out


# ---------------------------------------------------------------------------
# Deterministic synthetic generation
# ---------------------------------------------------------------------------

def make_synthetic(seed: int, days: Optional[int] = None,
                   start: Optional[str] = None,
                   start_price: Optional[float] = None,
                   signal_timeframe: str = "1h",
                   fill_timeframe: str = "1m",
                   cfg: Optional[DataConfig] = None) -> SyntheticData:
    """Generate deterministic, correlated 1m + 1h OHLCV and an hourly funding series.

    The price path is a regime-switching geometric Brownian motion with intrabar
    (high/low) noise and activity-correlated volume. Regimes (-1 bear / 0 sideways /
    +1 bull) shift daily with persistence and scale drift/vol. Everything draws from a
    single ``np.random.default_rng(seed)`` so the same seed + config reproduces the
    exact same bundle.

    ``ohlcv_1h`` is resampled from ``ohlcv_1m`` with ``closed='right', label='right'``:
    the 1h candle timestamped at ``T`` aggregates the 60 1m candles ending at ``T``.
    A signal from that candle may therefore only act on 1m candles with timestamp
    ``> T`` (enforced structurally later, in the backtester).
    """
    cfg = cfg if cfg is not None else DataConfig()
    days = days if days is not None else cfg.synth_days
    start = start if start is not None else cfg.synth_start
    start_price = start_price if start_price is not None else cfg.synth_start_price

    rng = np.random.default_rng(seed)

    minutes_per_day = 24 * 60
    n = days * minutes_per_day
    dt_annual = 1.0 / (365.0 * minutes_per_day)  # fraction of a year per 1m bar

    # --- daily regimes (persistent regime-switching) --------------------------
    day_regimes = []
    regime = 0
    for _ in range(days):
        if rng.random() < 0.35:
            regime = int(rng.integers(-1, 2))  # -1, 0, +1
        day_regimes.append(regime)
    regime_min = np.repeat(np.asarray(day_regimes, dtype=np.float64), minutes_per_day)

    # Per-minute annualized drift/vol from regime state.
    drift = cfg.synth_annual_drift + regime_min * cfg.synth_regime_drift
    vol = cfg.synth_annual_vol * (1.0 + cfg.synth_regime_vol_mult * np.abs(regime_min))

    # --- 1m close path via GBM ------------------------------------------------
    z = rng.normal(size=n)
    log_ret = drift * dt_annual + vol * np.sqrt(dt_annual) * z
    log_start = np.log(start_price)
    log_price = np.concatenate(([log_start], log_start + np.cumsum(log_ret)))
    close = np.exp(log_price[1:])
    close_prev = np.exp(log_price[:-1])  # previous close == this bar's open

    # --- intrabar high/low noise ----------------------------------------------
    sigma_bar = vol * np.sqrt(dt_annual)
    up = np.abs(rng.normal(size=n)) * sigma_bar
    dn = np.abs(rng.normal(size=n)) * sigma_bar
    high = np.maximum(close_prev, close) * (1.0 + up)
    low = np.minimum(close_prev, close) * (1.0 - dn)
    low = np.maximum(low, 1e-6)  # keep prices strictly positive

    # --- activity-correlated volume --------------------------------------------
    vol_shock = np.exp(rng.normal(size=n))
    rel = np.abs(log_ret) / (sigma_bar + 1e-12)
    volume = 1000.0 * vol_shock * (1.0 + 2.0 * rel)

    # --- timestamps -------------------------------------------------------------
    # Naive UTC internally (matches loader normalization); strip any tz/Z suffix.
    # Bar ``j`` is timestamped ``start + (j+1) min`` so each 1m bar covers a complete
    # minute ``(start+j min, start+(j+1) min]``. This keeps the 1h resample clean:
    # exactly 60 1m bars per 1h bar, with the first full 1h candle at ``start+1h``.
    start_ts = pd.Timestamp(start).tz_localize(None)
    timestamps = pd.date_range(start_ts + pd.Timedelta(minutes=1), periods=n,
                               freq=_pandas_freq(fill_timeframe))

    ohlcv_1m = pd.DataFrame(
        {"open": close_prev, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )

    # --- resample to signal timeframe -------------------------------------------
    ohlcv_1h = ohlcv_1m.resample(_pandas_freq(signal_timeframe), closed="right", label="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )

    # --- hourly funding series (mean-reverting AR(1) + return sensitivity) --------
    n_hours = len(ohlcv_1h)
    hourly_ret = ohlcv_1h["close"].pct_change().fillna(0.0).to_numpy()
    eps = rng.normal(size=n_hours)
    fund = np.empty(n_hours)
    fund[0] = cfg.synth_funding_mean + cfg.synth_funding_std * eps[0]
    for i in range(1, n_hours):
        ar = 0.7 * (fund[i - 1] - cfg.synth_funding_mean)
        fund[i] = cfg.synth_funding_mean + ar + cfg.synth_funding_std * eps[i] \
            + cfg.synth_funding_beta * hourly_ret[i]

    funding = pd.DataFrame(
        {"funding_rate": fund},
        index=pd.DatetimeIndex(ohlcv_1h.index, name="timestamp"),
    )

    return SyntheticData(ohlcv_1m=ohlcv_1m, ohlcv_1h=ohlcv_1h, funding=funding)


def parse_year_spec(year: Any) -> tuple[Optional[int], Optional[int]]:
    """Parse a year specification into (start_year, end_year).

    Returns (None, None) for all-time.
    Supports:
        - int: 2024 -> (2024, 2024)
        - str: "2024" -> (2024, 2024)
        - str range: "2022-2024", "2021:2025", "2022..2024" -> (2022, 2024)
        - all-time: "all", "all-time", "all_time", None -> (None, None)
        - tuple: (2022, 2024) -> (2022, 2024)
    """
    if year is None:
        return None, None
    if isinstance(year, (tuple, list)):
        return (int(year[0]), int(year[1]))
    s = str(year).strip().lower()
    if s in ("all", "all-time", "all_time", "alltime", "none", "*"):
        return None, None
    for sep in ("-", ":", ".."):
        if sep in s:
            parts = s.split(sep, 1)
            y1 = int(parts[0].strip())
            y2 = int(parts[1].strip())
            return (min(y1, y2), max(y1, y2))
    y = int(s)
    return (y, y)


def slice_year(data: SyntheticData, year: Any) -> SyntheticData:
    """Slice a SyntheticData bundle to a specific calendar year, year range, or all-time.

    Guarantees that the returned 1m bars cleanly align to completed 1h signal bars
    so that len(ohlcv_1m) % 60 == 0 always holds.

    Raises:
        ValueError: if no 1m bars fall into the requested year/range.
    """
    y1, y2 = parse_year_spec(year)
    if y1 is None:
        m_len = (len(data.ohlcv_1m) // 60) * 60
        if m_len == len(data.ohlcv_1m):
            return data
        m1 = data.ohlcv_1m.iloc[:m_len]
        h_len = m_len // 60
        h1 = data.ohlcv_1h.iloc[:h_len] if len(data.ohlcv_1h) >= h_len else _resample_signals(m1, "1h")
        fund = data.funding.reindex(h1.index).ffill().fillna(0.0)
        return SyntheticData(ohlcv_1m=m1, ohlcv_1h=h1, funding=fund)

    start_str = f"{y1}-01-01 00:00:00"
    end_str = f"{y2 + 1}-01-01 00:00:00"
    m1 = data.ohlcv_1m.loc[start_str:end_str]
    if len(m1) == 0:
        years = sorted(list(set(data.ohlcv_1m.index.year)))
        req = f"{y1}" if y1 == y2 else f"{y1}-{y2}"
        raise ValueError(
            f"Year specification '{req}' has no data in the dataset. Available years: {years}")

    m_len = (len(m1) // 60) * 60
    if m_len == 0:
        req = f"{y1}" if y1 == y2 else f"{y1}-{y2}"
        raise ValueError(f"Year specification '{req}' has fewer than 60 1m bars ({len(m1)} bars)")
    m1 = m1.iloc[:m_len]
    h_len = m_len // 60

    h1 = data.ohlcv_1h.loc[start_str:end_str]
    if len(h1) >= h_len:
        h1 = h1.iloc[:h_len]
    else:
        h1 = _resample_signals(m1, "1h")

    fund = data.funding.reindex(h1.index).ffill().fillna(0.0)
    return SyntheticData(ohlcv_1m=m1, ohlcv_1h=h1, funding=fund)


def available_years_for_data(data: Optional[SyntheticData] = None) -> list[int]:
    """Return all benchmark years [2021..2026], plus any extra years present in data."""
    years = {2021, 2022, 2023, 2024, 2025, 2026}
    if data is not None and len(data.ohlcv_1m) > 0:
        years.update(data.ohlcv_1m.index.year)
    return sorted(years)


def get_or_create_year_data(
    year: Any,
    source_data: Optional[SyntheticData] = None,
    seed: int = 42,
    cfg: Optional[DataConfig] = None,
) -> SyntheticData:
    """Return full market data for a calendar year, year range, or all-time for backtesting.

    If source_data has data spanning this year/range, returns that slice.
    Checks data/sol_1m.parquet on disk if source_data is not provided or insufficient.
    Otherwise generates a deterministic synthetic bundle covering the requested timeframe.
    """
    import datetime

    y1, y2 = parse_year_spec(year)
    if y1 is None:
        return get_or_create_all_time_data(source_data=source_data, seed=seed, cfg=cfg)

    # 1. Check in-memory source_data
    if source_data is not None and len(source_data.ohlcv_1m) > 0:
        try:
            sliced = slice_year(source_data, (y1, y2) if y1 != y2 else y1)
            min_days = 30 if y1 == 2026 else (180 if y1 == y2 else 180 * (y2 - y1 + 1))
            if len(sliced.ohlcv_1m) >= min_days * 1440:
                return sliced
        except ValueError:
            pass

    # 2. Check local Parquet cache
    parquet_path = os.path.join("data", "sol_1m.parquet")
    if os.path.exists(parquet_path):
        try:
            funding_path = os.path.join("data", "sol_funding.parquet")
            disk_data = load_market_data(parquet_path, funding_path if os.path.exists(funding_path) else None)
            sliced = slice_year(disk_data, (y1, y2) if y1 != y2 else y1)
            min_days = 30 if y1 == 2026 else (180 if y1 == y2 else 180 * (y2 - y1 + 1))
            if len(sliced.ohlcv_1m) >= min_days * 1440:
                return sliced
        except Exception:
            pass

    # 3. Generate deterministic synthetic data
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    end_year = min(y2, now_dt.year)
    start_dt = datetime.datetime(y1, 1, 1, tzinfo=datetime.timezone.utc)
    if end_year == now_dt.year:
        end_dt = now_dt
    else:
        end_dt = datetime.datetime(end_year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    days = max(30, (end_dt - start_dt).days)
    start = f"{y1}-01-01T00:00:00Z"
    return make_synthetic(seed=seed + y1, days=days, start=start, cfg=cfg)


def get_or_create_all_time_data(
    source_data: Optional[SyntheticData] = None,
    seed: int = 42,
    cfg: Optional[DataConfig] = None,
) -> SyntheticData:
    """Return full multi-year market data spanning from earliest available date to present.

    Prioritizes the dataset with the widest date range (earliest start to latest end)
    between local Parquet files and in-memory source_data, falling back to multi-year
    synthetic data if neither is available.
    """
    import datetime

    candidate: Optional[SyntheticData] = None

    # 1. Check local Parquet cache (real data spanning all available years)
    parquet_path = os.path.join("data", "sol_1m.parquet")
    if os.path.exists(parquet_path):
        try:
            funding_path = os.path.join("data", "sol_funding.parquet")
            candidate = load_market_data(parquet_path, funding_path if os.path.exists(funding_path) else None)
        except Exception as e:
            logger.warning("Failed to load all-time data from %s: %s", parquet_path, e)

    # 2. Check source_data: if it has more bars or broader range than candidate, prefer it
    if source_data is not None:
        if candidate is None or len(source_data.ohlcv_1m) > len(candidate.ohlcv_1m):
            candidate = source_data

    if candidate is not None and len(candidate.ohlcv_1m) > 0:
        m_len = (len(candidate.ohlcv_1m) // 60) * 60
        m1 = candidate.ohlcv_1m.iloc[:m_len]
        h_len = m_len // 60
        h1 = candidate.ohlcv_1h.iloc[:h_len] if len(candidate.ohlcv_1h) >= h_len else _resample_signals(m1, "1h")
        fund = candidate.funding.reindex(h1.index).ffill().fillna(0.0)
        return SyntheticData(ohlcv_1m=m1, ohlcv_1h=h1, funding=fund)

    # 3. Deterministic multi-year synthetic continuous history from 2021 to present
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    start_dt = datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc)
    total_days = max(365, (now_dt - start_dt).days)
    return make_synthetic(seed=seed, days=total_days, start="2021-01-01T00:00:00Z", cfg=cfg)
