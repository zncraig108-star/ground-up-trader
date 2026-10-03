"""core/micro_futures_strategy.py — Signal Engine for NYS Micro Futures.

Supports both 4h macro and 1h intraday signal timeframes across regulated crypto futures:
- BTC (Nano BTI / Micro MBT)
- ETH (Micro MET)
- SOL (Micro SOL)

Generates synchronized dual-directional (Long + Short) breakout and trend signals,
incorporating Kaufman Efficiency Ratio (ER) filters for normal vs defensive regimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from config import MicroFuturesConfig


@dataclass(frozen=True)
class MicroAssetSignals:
    """Precomputed signals for a single micro futures asset, aligned to 1h execution timestamps."""
    product_id: str
    timestamps: pd.DatetimeIndex
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    upper_breakout: np.ndarray     # 20d High channel (Long entry)
    lower_breakout: np.ndarray     # 20d Low channel (Short entry)
    trailing_exit_low: np.ndarray  # Exit stop for Longs
    trailing_exit_high: np.ndarray # Exit stop for Shorts
    ema_50d: np.ndarray            # 50-day trend EMA
    ema_slope: np.ndarray          # Slope of 50d EMA
    atr: np.ndarray                # ATR for stop distance calculation
    er: np.ndarray                 # Kaufman Efficiency Ratio
    sma_20: np.ndarray             # 20-period SMA for parabolic climax detection
    is_defensive_clean: np.ndarray # True if ER >= defensive threshold (A+ setup)


@dataclass(frozen=True)
class MicroFuturesSignals:
    """Synchronized signals across the micro futures universe."""
    universe: Tuple[str, ...]
    timestamps: pd.DatetimeIndex
    asset_signals: Mapping[str, MicroAssetSignals]


def compute_micro_futures_signals(
    market_data_1h: Mapping[str, pd.DataFrame],
    cfg: MicroFuturesConfig,
) -> MicroFuturesSignals:
    """Compute synchronized signals for micro futures across the specified timeframe (4h or 1h)."""
    universe = cfg.universe
    common_idx = market_data_1h[universe[0]].index
    for p in universe[1:]:
        common_idx = common_idx.intersection(market_data_1h[p].index)

    tf_hours = cfg.signal_timeframe_hours
    asset_signals = {}

    for p in universe:
        df_raw = market_data_1h[p].reindex(common_idx).ffill().bfill()

        # Resample if 4h, else use 1h directly
        if tf_hours > 1:
            rule = f"{tf_hours}h"
            df_tf = df_raw.resample(rule, closed="left", label="left").agg({
                "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
            }).dropna()
            bars_per_day = 24 // tf_hours
        else:
            df_tf = df_raw.copy()
            bars_per_day = 24

        c = df_tf["close"]
        h = df_tf["high"]
        l = df_tf["low"]

        # 1. 20-day High / Low Donchian Breakout Channels
        trend_bars = cfg.trend_lookback_days * bars_per_day
        up_ch = h.rolling(trend_bars, min_periods=trend_bars // 2).max().shift(1)
        dn_ch = l.rolling(trend_bars, min_periods=trend_bars // 2).min().shift(1)

        # 2. 4-day Trailing Stop Channels
        exit_bars = cfg.exit_lookback_days * bars_per_day
        ex_l = l.rolling(exit_bars, min_periods=exit_bars // 2).min().shift(1)
        ex_h = h.rolling(exit_bars, min_periods=exit_bars // 2).max().shift(1)

        # 3. 50-day EMA and Slope
        ema_span = 50 * bars_per_day
        ema_50d = c.ewm(span=ema_span, adjust=False).mean().shift(1)
        slope = ema_50d.diff(bars_per_day).shift(1)

        # 4. ATR 14
        atr_period = cfg.atr_period_days * bars_per_day // 2
        tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
        atr = tr.rolling(max(14, atr_period), min_periods=7).mean().shift(1)

        # 5. 20-period SMA for Climax Detection
        sma_20 = c.rolling(20, min_periods=10).mean().shift(1)

        # 6. Kaufman Efficiency Ratio (ER) over 30 bars
        net_change = (c - c.shift(30)).abs()
        gross_noise = (c - c.shift(1)).abs().rolling(30).sum()
        er = (net_change / np.maximum(gross_noise, 1e-8)).shift(1).fillna(0.0)

        # Forward map to 1h execution timestamps
        df_sig_tf = pd.DataFrame({
            "up_ch": up_ch,
            "dn_ch": dn_ch,
            "ex_l": ex_l,
            "ex_h": ex_h,
            "ema_50d": ema_50d,
            "slope": slope,
            "atr": atr,
            "sma_20": sma_20,
            "er": er,
        }, index=df_tf.index)

        df_sig_1h = df_sig_tf.reindex(common_idx).ffill()

        er_arr = df_sig_1h["er"].to_numpy(dtype=np.float64)
        is_def_clean = er_arr >= cfg.er_min_defensive

        asset_signals[p] = MicroAssetSignals(
            product_id=p,
            timestamps=common_idx,
            close=df_raw["close"].to_numpy(dtype=np.float64),
            high=df_raw["high"].to_numpy(dtype=np.float64),
            low=df_raw["low"].to_numpy(dtype=np.float64),
            upper_breakout=df_sig_1h["up_ch"].to_numpy(dtype=np.float64),
            lower_breakout=df_sig_1h["dn_ch"].to_numpy(dtype=np.float64),
            trailing_exit_low=df_sig_1h["ex_l"].to_numpy(dtype=np.float64),
            trailing_exit_high=df_sig_1h["ex_h"].to_numpy(dtype=np.float64),
            ema_50d=df_sig_1h["ema_50d"].to_numpy(dtype=np.float64),
            ema_slope=df_sig_1h["slope"].to_numpy(dtype=np.float64),
            atr=df_sig_1h["atr"].to_numpy(dtype=np.float64),
            er=er_arr,
            sma_20=df_sig_1h["sma_20"].to_numpy(dtype=np.float64),
            is_defensive_clean=is_def_clean,
        )

    return MicroFuturesSignals(
        universe=universe,
        timestamps=common_idx,
        asset_signals=asset_signals,
    )
