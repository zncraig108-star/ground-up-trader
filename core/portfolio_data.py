"""core/portfolio_data.py — Multi-Asset Portfolio Data Loader and Synthetic Generator.

Handles loading, aligning, and synthetic generation for a 9-asset crypto universe:
  BTC-PERP, ETH-PERP, SOL-PERP, AVAX-PERP, DOGE-PERP, LINK-PERP, SUI-PERP, NEAR-PERP, XRP-PERP.

Ensures zero-lookahead alignment and graceful degradation with offline synthetic data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from config import PortfolioConfig


# Real-world crypto volatility hierarchy (annualized)
ASSET_VOL_MAP: Mapping[str, float] = {
    "BTC-PERP": 0.45,
    "ETH-PERP": 0.55,
    "SOL-PERP": 0.70,
    "AVAX-PERP": 0.75,
    "DOGE-PERP": 0.85,
    "LINK-PERP": 0.65,
    "SUI-PERP": 0.80,
    "NEAR-PERP": 0.75,
    "XRP-PERP": 0.60,
}

ASSET_BASE_PRICE_MAP: Mapping[str, float] = {
    "BTC-PERP": 45000.0,
    "ETH-PERP": 2500.0,
    "SOL-PERP": 100.0,
    "AVAX-PERP": 35.0,
    "DOGE-PERP": 0.12,
    "LINK-PERP": 15.0,
    "SUI-PERP": 1.5,
    "NEAR-PERP": 5.0,
    "XRP-PERP": 0.55,
}


@dataclass(frozen=True)
class PortfolioDataBundle:
    """Synchronized market data for the portfolio universe."""
    universe: Tuple[str, ...]
    market_data_1h: Mapping[str, pd.DataFrame]
    funding_data_1h: Mapping[str, pd.Series]


def make_synthetic_portfolio(
    cfg: PortfolioConfig,
    n_days: int = 180,
    seed: int = 42,
    start_date: str = "2024-01-01T00:00:00Z",
) -> PortfolioDataBundle:
    """Generate realistic correlated synthetic market data for the entire universe.

    Guarantees offline execution without network or disk files.
    """
    rng = np.random.default_rng(seed)
    n_hours = n_days * 24
    dt_index = pd.date_range(start_date, periods=n_hours, freq="1h", tz="UTC")

    # Common macro crypto factor (BTC/Macro beta)
    macro_drift = 0.15 / 8760.0  # slight bull market bias
    macro_vol = 0.40 / np.sqrt(8760.0)
    macro_returns = rng.normal(macro_drift, macro_vol, size=n_hours)

    market_data: Dict[str, pd.DataFrame] = {}
    funding_data: Dict[str, pd.Series] = {}

    for product_id in cfg.universe:
        ann_vol = ASSET_VOL_MAP.get(product_id, 0.70)
        base_px = ASSET_BASE_PRICE_MAP.get(product_id, 10.0)
        idio_vol = (ann_vol * 0.7) / np.sqrt(8760.0)

        # Correlated returns = 0.7 * macro + 0.3 * idiosyncratic
        idio_returns = rng.normal(0.0, idio_vol, size=n_hours)
        total_returns = 0.75 * macro_returns + 0.55 * idio_returns

        # Generate price series
        log_px = np.log(base_px) + np.cumsum(total_returns)
        close = np.exp(log_px)

        # High, Low, Open from intra-hour volatility
        bar_vol = idio_vol * close
        open_px = np.roll(close, 1)
        open_px[0] = base_px
        high = np.maximum(open_px, close) + rng.uniform(0.1, 0.8, size=n_hours) * bar_vol
        low = np.minimum(open_px, close) - rng.uniform(0.1, 0.8, size=n_hours) * bar_vol
        volume = rng.lognormal(mean=12.0, sigma=0.8, size=n_hours)

        df_1h = pd.DataFrame({
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=dt_index)

        # Funding rate: correlated with recent returns
        hourly_funding = np.clip(rng.normal(1.25e-5, 2e-5, size=n_hours) + 0.05 * total_returns, -0.001, 0.001)
        s_funding = pd.Series(hourly_funding, index=dt_index)

        market_data[product_id] = df_1h
        funding_data[product_id] = s_funding

    return PortfolioDataBundle(
        universe=cfg.universe,
        market_data_1h=market_data,
        funding_data_1h=funding_data,
    )


def load_portfolio_data(
    cfg: PortfolioConfig,
    data_dir: str = "data",
) -> PortfolioDataBundle:
    """Load existing multi-asset parquet files from disk if available, else fallback."""
    market_data: Dict[str, pd.DataFrame] = {}
    funding_data: Dict[str, pd.Series] = {}

    missing: List[str] = []
    for p in cfg.universe:
        sym = p.split("-")[0].lower()
        ohlcv_path = os.path.join(data_dir, f"{sym}_1h.parquet")
        fund_path = os.path.join(data_dir, f"{sym}_funding.parquet")

        if os.path.exists(ohlcv_path):
            df = pd.read_parquet(ohlcv_path)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
            market_data[p] = df

            if os.path.exists(fund_path):
                f_df = pd.read_parquet(fund_path)
                if f_df.index.tz is not None:
                    f_df.index = f_df.index.tz_convert("UTC").tz_localize(None)
                funding_data[p] = f_df["funding_rate"]
            else:
                funding_data[p] = pd.Series(0.0, index=df.index)
        else:
            missing.append(p)

    if missing:
        # Graceful fallback: if real data isn't fully downloaded, return synthetic bundle
        return make_synthetic_portfolio(cfg)

    # Align common timestamps
    common_idx = None
    for p in cfg.universe:
        idx = market_data[p].index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

    for p in cfg.universe:
        market_data[p] = market_data[p].reindex(common_idx).ffill()
        funding_data[p] = funding_data[p].reindex(common_idx).ffill().fillna(0.0)

    return PortfolioDataBundle(
        universe=cfg.universe,
        market_data_1h=market_data,
        funding_data_1h=funding_data,
    )
