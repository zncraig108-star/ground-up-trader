"""core/portfolio_strategy.py — Multi-Asset Macro Trend & Carry CTA Strategy.

Computes parameter-light, robust signals across a multi-asset crypto universe
using 4h macro candles derived from 1h historical data:
  1. Donchian Channel Breakout (20-day entry, 10-day exit).
  2. Macro Trend Filter (50-day EMA slope).
  3. Funding Carry Filter (suppress entries into extreme adverse carry).
  4. Volatility-Parity (Risk-Parity) Target Allocation:
     Assets with lower volatility (e.g. BTC) get larger allocations;
     high-beta altcoins get smaller allocations to equalize risk contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from config import InstrumentSpec, PortfolioConfig


@dataclass(frozen=True)
class AssetSignals:
    """Precomputed 4h macro signals for a single asset, aligned to 1h timestamps."""
    product_id: str
    timestamps: pd.DatetimeIndex
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    regime: np.ndarray       # +1 (long), -1 (short), 0 (flat)
    target_weight: np.ndarray  # Fraction of portfolio equity (signed)
    stop_loss_px: np.ndarray   # ATR trailing stop level
    atr: np.ndarray            # 4h ATR mapped forward to 1h bars
    funding_rate: np.ndarray   # Hourly funding rate


@dataclass(frozen=True)
class PortfolioSignals:
    """Synchronized multi-asset signals aligned across the entire universe."""
    universe: Tuple[str, ...]
    timestamps: pd.DatetimeIndex
    asset_signals: Mapping[str, AssetSignals]


def compute_asset_signals_4h(
    ohlcv_1h: pd.DataFrame,
    funding_1h: pd.Series,
    product_id: str,
    cfg: PortfolioConfig,
) -> AssetSignals:
    """Compute 4h macro trend and volatility signals for one asset, mapped back to 1h."""
    ts_1h = ohlcv_1h.index
    n_1h = len(ohlcv_1h)

    # Resample 1h to 4h
    ohlcv_4h = ohlcv_1h.resample("4h", closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    c4 = ohlcv_4h["close"].to_numpy(dtype=np.float64)
    h4 = ohlcv_4h["high"].to_numpy(dtype=np.float64)
    l4 = ohlcv_4h["low"].to_numpy(dtype=np.float64)
    n4 = len(c4)

    # 1. Donchian Breakout Channels
    lookback_in = cfg.trend_lookback_bars  # e.g. 120 bars = 20 days
    lookback_out = cfg.exit_lookback_bars   # e.g. 60 bars = 10 days

    upper_channel = pd.Series(h4).rolling(lookback_in, min_periods=lookback_in).max().shift(1).to_numpy()
    lower_channel = pd.Series(l4).rolling(lookback_in, min_periods=lookback_in).min().shift(1).to_numpy()
    exit_lower = pd.Series(l4).rolling(lookback_out, min_periods=lookback_out).min().shift(1).to_numpy()
    exit_upper = pd.Series(h4).rolling(lookback_out, min_periods=lookback_out).max().shift(1).to_numpy()

    # 2. 7-day Wilder ATR on 4h bars
    atr_period = cfg.atr_period_bars  # e.g. 42 bars = 7 days
    tr = np.zeros(n4, dtype=np.float64)
    for i in range(1, n4):
        tr[i] = max(h4[i] - l4[i], abs(h4[i] - c4[i - 1]), abs(l4[i] - c4[i - 1]))
    atr_4h = pd.Series(tr).ewm(alpha=1.0 / max(atr_period, 1), adjust=False).mean().to_numpy()

    # 3. 50-day EMA Trend Filter
    ema_span = 300  # 300 * 4h = 50 days
    ema_50d = pd.Series(c4).ewm(span=ema_span, adjust=False).mean().to_numpy()
    ema_slope = pd.Series(ema_50d).diff(6).fillna(0.0).to_numpy()  # 24h slope

    # 4. State-machine for Regime (Long / Flat)
    regime_4h = np.zeros(n4, dtype=np.float64)
    stop_px_4h = np.zeros(n4, dtype=np.float64)
    current_regime = 0.0
    entry_px = 0.0
    trailing_peak = 0.0
    trailing_trough = 1e9

    # Resample funding to 4h
    fund_4h = funding_1h.resample("4h", closed="left", label="left").mean().reindex(ohlcv_4h.index).ffill().fillna(0.0).to_numpy()
    fund_z = (fund_4h - pd.Series(fund_4h).rolling(180, min_periods=30).mean().to_numpy()) / np.maximum(pd.Series(fund_4h).rolling(180, min_periods=30).std().to_numpy(), 1e-6)
    fund_z = np.nan_to_num(fund_z, nan=0.0)

    allow_short = cfg.allow_alt_shorts

    for i in range(max(lookback_in, atr_period), n4):
        # Entry triggers
        if current_regime == 0.0:
            # Long entry: price breaks breakout high + 50d EMA rising + funding not over-crowded (<25% APR)
            if (c4[i - 1] > upper_channel[i - 1] and ema_slope[i - 1] > 0 
                and fund_z[i - 1] < cfg.funding_filter_z and fund_4h[i - 1] < 0.00003):
                current_regime = 1.0
                entry_px = c4[i - 1]
                trailing_peak = c4[i - 1]
                stop_px_4h[i] = entry_px - cfg.atr_stop_mult * atr_4h[i - 1]
            elif allow_short and c4[i - 1] < lower_channel[i - 1] and ema_slope[i - 1] < 0 and fund_z[i - 1] > -cfg.funding_filter_z:
                current_regime = -1.0
                entry_px = c4[i - 1]
                trailing_trough = c4[i - 1]
                stop_px_4h[i] = entry_px + cfg.atr_stop_mult * atr_4h[i - 1]
        elif current_regime == 1.0:
            trailing_peak = max(trailing_peak, h4[i - 1])
            init_sl = entry_px - cfg.atr_stop_mult * atr_4h[i - 1]
            # Trail only after +1.5 ATR excursion
            if trailing_peak >= entry_px + 1.5 * atr_4h[i - 1]:
                sl = max(init_sl, trailing_peak - cfg.atr_stop_mult * atr_4h[i - 1])
            else:
                sl = init_sl
            # Exit on ATR trailing stop or 50-day EMA macro trend breakdown
            if c4[i - 1] < sl or (c4[i - 1] < ema_50d[i - 1] and ema_slope[i - 1] < 0):
                current_regime = 0.0
            stop_px_4h[i] = sl
        elif current_regime == -1.0:
            trailing_trough = min(trailing_trough, l4[i - 1])
            init_sl = entry_px + cfg.atr_stop_mult * atr_4h[i - 1]
            # Trail only after +1.5 ATR excursion
            if trailing_trough <= entry_px - 1.5 * atr_4h[i - 1]:
                sl = min(init_sl, trailing_trough + cfg.atr_stop_mult * atr_4h[i - 1])
            else:
                sl = init_sl
            # Exit on ATR trailing stop or 50-day EMA macro trend reversal
            if c4[i - 1] > sl or (c4[i - 1] > ema_50d[i - 1] and ema_slope[i - 1] > 0):
                current_regime = 0.0
            stop_px_4h[i] = sl

        regime_4h[i] = current_regime

    # Map 4h signals forward to the 1h index cleanly via forward-fill
    df_4h = pd.DataFrame({
        "regime": regime_4h,
        "stop_loss_px": stop_px_4h,
        "atr": atr_4h,
    }, index=ohlcv_4h.index)

    df_1h = df_4h.reindex(ts_1h).ffill().fillna(0.0)

    # Normalized asset annual volatility: (ATR / Close) * sqrt(2190 4h bars per year)
    ann_vol_4h = (atr_4h / np.maximum(c4, 1e-6)) * np.sqrt(2190.0)
    vol_series = pd.Series(ann_vol_4h, index=ohlcv_4h.index).reindex(ts_1h).ffill().fillna(0.60).to_numpy()

    # Raw un-normalized risk parity weight (1 / vol)
    safe_vol = np.maximum(vol_series, 0.05)
    raw_weight = np.where(safe_vol > 0.05, 1.0 / safe_vol, 1.0)

    # Align funding rate
    fund_aligned = funding_1h.reindex(ts_1h).ffill().fillna(0.0).to_numpy(dtype=np.float64)

    return AssetSignals(
        product_id=product_id,
        timestamps=ts_1h,
        close=ohlcv_1h["close"].to_numpy(dtype=np.float64),
        high=ohlcv_1h["high"].to_numpy(dtype=np.float64),
        low=ohlcv_1h["low"].to_numpy(dtype=np.float64),
        regime=df_1h["regime"].to_numpy(dtype=np.float64),
        target_weight=raw_weight,  # normalized across portfolio in compute_portfolio_signals
        stop_loss_px=df_1h["stop_loss_px"].to_numpy(dtype=np.float64),
        atr=df_1h["atr"].to_numpy(dtype=np.float64),
        funding_rate=fund_aligned,
    )


def compute_portfolio_signals(
    market_data_1h: Mapping[str, pd.DataFrame],
    funding_data_1h: Mapping[str, pd.Series],
    cfg: PortfolioConfig,
) -> PortfolioSignals:
    """Compute synchronized, risk-parity normalized signals across the universe."""
    universe = cfg.universe
    asset_sigs: Dict[str, AssetSignals] = {}

    # Common time index
    common_index: Optional[pd.DatetimeIndex] = None
    for p in universe:
        if p in market_data_1h:
            idx = market_data_1h[p].index
            common_index = idx if common_index is None else common_index.intersection(idx)

    if common_index is None or len(common_index) == 0:
        raise ValueError("No common timestamps found across portfolio universe")

    # Step 1: Compute per-asset signals
    for p in universe:
        ohlcv = market_data_1h[p].reindex(common_index).ffill()
        fund = funding_data_1h.get(p, pd.Series(0.0, index=common_index)).reindex(common_index).ffill()
        asset_sigs[p] = compute_asset_signals_4h(ohlcv, fund, p, cfg)

    # Step 2: Normalize weights across universe for Volatility-Parity (Risk-Parity)
    n_bars = len(common_index)
    sum_weights = np.zeros(n_bars, dtype=np.float64)
    for p in universe:
        sum_weights += asset_sigs[p].target_weight

    # Scale so sum of active weights == cfg.max_portfolio_leverage
    normalized_signals: Dict[str, AssetSignals] = {}
    for p in universe:
        sig = asset_sigs[p]
        norm_w = (sig.target_weight / np.maximum(sum_weights, 1e-6)) * cfg.max_portfolio_leverage
        # Clamp single asset weight
        norm_w = np.clip(norm_w, 0.0, cfg.max_single_asset_leverage)
        # Apply regime direction (+1 long, -1 short, 0 flat)
        final_weight = norm_w * sig.regime

        normalized_signals[p] = AssetSignals(
            product_id=p,
            timestamps=common_index,
            close=sig.close,
            high=sig.high,
            low=sig.low,
            regime=sig.regime,
            target_weight=final_weight,
            stop_loss_px=sig.stop_loss_px,
            atr=sig.atr,
            funding_rate=sig.funding_rate,
        )

    return PortfolioSignals(
        universe=universe,
        timestamps=common_index,
        asset_signals=normalized_signals,
    )
