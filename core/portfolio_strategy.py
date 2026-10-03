"""core/portfolio_strategy.py — Asymmetric 100x Compounding Macro Portfolio Strategy.

Implements the 4-lever compounding macro CTA architecture:
  1. Top-2 Relative Strength Momentum Leader Selection (concentrated winners).
  2. Sticky Trend: Let leaders ride full multi-week trends without rank churn.
  3. 3-Tranche Pyramiding: Scale from 0.70x to 1.75x leverage per leader using house money:
     - Tranche 1: Base entry (0.70x leverage).
     - Tranche 2: At +2.0 ATR profit, snap stop to breakeven + add 0.50x leverage (1.20x total).
     - Tranche 3: At +4.0 ATR profit, lock stop to +2.0 ATR + add 0.55x leverage (1.75x total).
  4. BTC 200-day SMA Macro Regime Filter:
     - BTC > 200d SMA: Bull mode, long top momentum leaders.
     - BTC < 200d SMA: Bear mode, short only BTC & ETH, keep altcoins in 100% Cash.
  5. Parabolic Climax Exits: Tightens trailing stop to 1.0 ATR on vertical blow-offs.
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
    regime: np.ndarray         # +1 (long), -1 (short), 0 (flat)
    target_weight: np.ndarray  # Fraction of portfolio equity (signed)
    stop_loss_px: np.ndarray   # Dynamic trailing stop level
    atr: np.ndarray            # 4h ATR mapped forward to 1h bars
    funding_rate: np.ndarray   # Hourly funding rate
    tranche: np.ndarray        # Active tranche level (1, 2, 3, or -1)
    sma_20: Optional[np.ndarray] = None  # 4h SMA20 mapped forward to 1h bars for climax trimming


@dataclass(frozen=True)
class PortfolioSignals:
    """Synchronized multi-asset signals aligned across the entire universe."""
    universe: Tuple[str, ...]
    timestamps: pd.DatetimeIndex
    asset_signals: Mapping[str, AssetSignals]


def compute_portfolio_signals(
    market_data_1h: Mapping[str, pd.DataFrame],
    funding_data_1h: Mapping[str, pd.Series],
    cfg: PortfolioConfig,
) -> PortfolioSignals:
    """Compute synchronized, momentum-ranked pyramided signals across the universe."""
    universe = cfg.universe

    # Common time index
    common_index: Optional[pd.DatetimeIndex] = None
    for p in universe:
        if p in market_data_1h:
            idx = market_data_1h[p].index
            common_index = idx if common_index is None else common_index.intersection(idx)

    if common_index is None or len(common_index) == 0:
        raise ValueError("No common timestamps found across portfolio universe")

    # Step 1: Precompute 4h Resampled OHLCV and Indicators for every asset
    data_4h: Dict[str, pd.DataFrame] = {}
    atr_4h_map: Dict[str, np.ndarray] = {}
    ema_50d_map: Dict[str, np.ndarray] = {}
    ema_slope_map: Dict[str, np.ndarray] = {}
    upper_chan_map: Dict[str, np.ndarray] = {}
    lower_chan_map: Dict[str, np.ndarray] = {}
    fund_4h_map: Dict[str, np.ndarray] = {}
    fund_z_map: Dict[str, np.ndarray] = {}
    roc_30d_map: Dict[str, np.ndarray] = {}

    lookback_in = cfg.trend_lookback_bars   # 240 * 4h = 40 days
    atr_period = cfg.atr_period_bars       # 42 * 4h = 7 days
    ema_span = 300                         # 300 * 4h = 50 days
    roc_period = cfg.momentum_lookback_bars  # 180 * 4h = 30 days

    # Step 1: Precompute unified 4h master index aligned on exact 4h candle boundaries
    ref_4h_idx = market_data_1h[universe[0]].resample("4h", closed="left", label="left").agg({"close": "last"}).index
    n4 = len(ref_4h_idx)

    er_map: Dict[str, np.ndarray] = {}
    sma_20_map: Dict[str, np.ndarray] = {}
    exit_chan_low: Dict[str, np.ndarray] = {}
    exit_chan_high: Dict[str, np.ndarray] = {}

    for p in universe:
        ohlcv = market_data_1h[p].reindex(common_index)
        df_4h = ohlcv.resample("4h", closed="left", label="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).reindex(ref_4h_idx)
        data_4h[p] = df_4h

        c = df_4h["close"].to_numpy(dtype=np.float64)
        h = df_4h["high"].to_numpy(dtype=np.float64)
        l = df_4h["low"].to_numpy(dtype=np.float64)
        n = len(c)


        # 20-day Entry Breakout Channels (120 4h bars)
        upper_chan_map[p] = pd.Series(h).rolling(lookback_in, min_periods=lookback_in // 2).max().shift(1).to_numpy()
        lower_chan_map[p] = pd.Series(l).rolling(lookback_in, min_periods=lookback_in // 2).min().shift(1).to_numpy()

        # 12-day Exit Channels (72 4h bars)
        exit_chan_low[p] = pd.Series(l).rolling(cfg.exit_lookback_bars, min_periods=cfg.exit_lookback_bars // 2).min().shift(1).to_numpy()
        exit_chan_high[p] = pd.Series(h).rolling(cfg.exit_lookback_bars, min_periods=cfg.exit_lookback_bars // 2).max().shift(1).to_numpy()

        # ATR 4h
        tr = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if np.isnan(c[i]) or np.isnan(c[i - 1]):
                tr[i] = np.nan
            else:
                tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        atr_4h_map[p] = pd.Series(tr).ewm(alpha=1.0 / max(atr_period, 1), adjust=False).mean().to_numpy()

        # 50-day EMA & Slope
        ema = pd.Series(c).ewm(span=ema_span, adjust=False).mean().to_numpy()
        ema_50d_map[p] = ema
        ema_slope_map[p] = pd.Series(ema).diff(6).fillna(0.0).to_numpy()

        # 30-day Momentum (Rate of Change), NaNs become -999.0
        roc = (pd.Series(c) / pd.Series(c).shift(roc_period) - 1.0).fillna(-999.0).to_numpy()
        roc_30d_map[p] = roc

        # Funding rate & z-score
        fund = funding_data_1h.get(p, pd.Series(0.0, index=common_index)).reindex(common_index).fillna(0.0)
        f4 = fund.resample("4h", closed="left", label="left").mean().reindex(ref_4h_idx).fillna(0.0).to_numpy()
        fund_4h_map[p] = f4
        f_roll = pd.Series(f4).rolling(180, min_periods=30)
        fz = (f4 - f_roll.mean().to_numpy()) / np.maximum(f_roll.std().to_numpy(), 1e-6)
        fund_z_map[p] = np.nan_to_num(fz, nan=0.0)

        # Kaufman Efficiency Ratio (ER) on 4h (30 periods)
        net_c = pd.Series(c).diff(30).abs()
        gross_c = pd.Series(c).diff(1).abs().rolling(30).sum()
        er_map[p] = (net_c / np.maximum(gross_c, 1e-8)).fillna(0.0).to_numpy()

        # SMA 20 on 4h for Parabolic Climax detection
        sma_20_map[p] = pd.Series(c).rolling(20, min_periods=10).mean().to_numpy()


    # Step 2: Compute Macro BTC 200-day SMA Regime Filter
    btc_prod = "BTC-PERP" if "BTC-PERP" in data_4h else universe[0]
    btc_c = data_4h[btc_prod]["close"].ffill().bfill().to_numpy()
    btc_sma_200 = pd.Series(btc_c).rolling(cfg.macro_btc_sma_bars, min_periods=120).mean().to_numpy()
    is_macro_bull = btc_c >= btc_sma_200
    is_macro_bear = ~is_macro_bull

    # Step 3: Compute Cross-Sectional Momentum Rankings across universe per 4h bar
    all_rocs = np.stack([roc_30d_map[p] for p in universe], axis=0)  # (M, n4)
    # Order: highest ROC gets rank 0, second highest rank 1...
    ranks = np.argsort(-all_rocs, axis=0)  # (M, n4)

    is_top_leader: Dict[str, np.ndarray] = {}
    for idx_u, p in enumerate(universe):
        in_top_k = np.any(ranks[:cfg.max_concurrent_leaders, :] == idx_u, axis=0)
        valid_px = ~np.isnan(data_4h[p]["close"].to_numpy())
        is_top_leader[p] = in_top_k & valid_px

    # Step 4: Joint Multi-Asset State Machine (Turtle Trend + Pyramiding)
    regime_out: Dict[str, np.ndarray] = {p: np.zeros(n4, dtype=np.float64) for p in universe}
    weights_out: Dict[str, np.ndarray] = {p: np.zeros(n4, dtype=np.float64) for p in universe}
    stops_out: Dict[str, np.ndarray] = {p: np.zeros(n4, dtype=np.float64) for p in universe}
    tranche_out: Dict[str, np.ndarray] = {p: np.zeros(n4, dtype=np.float64) for p in universe}

    pos_state: Dict[str, int] = {p: 0 for p in universe}  # 0: flat, 1, 2, 3: Long tranches, -1: Bear Short
    entry_prices: Dict[str, float] = {p: 0.0 for p in universe}
    trailing_peaks: Dict[str, float] = {p: 0.0 for p in universe}
    trailing_troughs: Dict[str, float] = {p: 1e9 for p in universe}
    current_stops: Dict[str, float] = {p: 0.0 for p in universe}

    warmup = max(lookback_in, atr_period, roc_period, 120)

    for i in range(warmup, n4):
        # Count currently active leaders
        active_longs = sum(1 for p in universe if pos_state[p] > 0)

        for p in universe:
            c4 = data_4h[p]["close"].to_numpy()[i - 1]
            h4 = data_4h[p]["high"].to_numpy()[i - 1]
            l4 = data_4h[p]["low"].to_numpy()[i - 1]
            atr = atr_4h_map[p][i - 1]
            upper_ch = upper_chan_map[p][i - 1]
            lower_ch = lower_chan_map[p][i - 1]
            ex_l = exit_chan_low[p][i - 1]
            ex_h = exit_chan_high[p][i - 1]
            is_leader = is_top_leader[p][i - 1]

            if np.isnan(c4) or np.isnan(atr):
                continue

            state = pos_state[p]

            # ---------------------------
            # State 0: Flat -> Look for Entry
            # ---------------------------
            if state == 0:
                pass_asset = True
                if getattr(cfg, "require_asset_ema50_filter", False):
                    pass_asset = pass_asset and (c4 > ema_50d_map[p][i - 1]) and (ema_slope_map[p][i - 1] > 0)
                if getattr(cfg, "er_filter_enabled", False):
                    pass_asset = pass_asset and (er_map[p][i - 1] >= getattr(cfg, "er_min", 0.28))

                # 1. Bull Mode: Long top momentum leader breakout (positive 30d momentum)
                if is_macro_bull[i - 1] and active_longs < cfg.max_concurrent_leaders and is_leader and roc_30d_map[p][i - 1] > 0 and pass_asset:

                    if c4 > upper_ch:
                        pos_state[p] = 1
                        entry_prices[p] = c4
                        trailing_peaks[p] = c4
                        current_stops[p] = ex_l if not np.isnan(ex_l) else (c4 - cfg.atr_stop_mult * atr)
                        active_longs += 1

                # 2. Bear Mode: Short BTC & ETH on 20-day breakdown (negative 30d momentum)
                elif is_macro_bear[i - 1] and cfg.allow_bear_shorts and p in ("BTC-PERP", "ETH-PERP") and roc_30d_map[p][i - 1] < 0:
                    if c4 < lower_ch:
                        pos_state[p] = -1
                        entry_prices[p] = c4
                        trailing_troughs[p] = c4
                        current_stops[p] = ex_h if not np.isnan(ex_h) else (c4 + cfg.atr_stop_mult * atr)

            # ---------------------------
            # Long States (Tranches 1, 2, 3)
            # ---------------------------
            elif state > 0:
                trailing_peaks[p] = max(trailing_peaks[p], h4)
                peak = trailing_peaks[p]
                entry = entry_prices[p]

                # Pyramiding to Tranche 2 (at +2.5 ATR milestone)
                if state == 1 and cfg.pyramid_enabled and peak >= entry + cfg.pyramid_milestone_1_atr * atr:
                    pos_state[p] = 2

                # Pyramiding to Tranche 3 (at +5.0 ATR milestone)
                elif state == 2 and cfg.pyramid_enabled and peak >= entry + cfg.pyramid_milestone_2_atr * atr:
                    pos_state[p] = 3

                # Trailing exit channel
                current_stops[p] = ex_l if not np.isnan(ex_l) else (peak - cfg.atr_stop_mult * atr)

                # Exit Check: Breached stop channel or macro regime flips bear or asset drops below 50d EMA
                exit_cond = l4 <= current_stops[p] or is_macro_bear[i - 1]
                if getattr(cfg, "require_asset_ema50_filter", False) and c4 < ema_50d_map[p][i - 1]:
                    exit_cond = True

                if exit_cond:
                    pos_state[p] = 0
                    current_stops[p] = 0.0
                    entry_prices[p] = 0.0
                    trailing_peaks[p] = 0.0

            # ---------------------------
            # Short State (BTC/ETH Bear Hedge)
            # ---------------------------
            elif state < 0:
                trailing_troughs[p] = min(trailing_troughs[p], l4)
                current_stops[p] = ex_h if not np.isnan(ex_h) else (trailing_troughs[p] + cfg.atr_stop_mult * atr)

                # Exit Check: Breached stop channel or macro regime flips bull
                if h4 >= current_stops[p] or is_macro_bull[i - 1]:
                    pos_state[p] = 0
                    current_stops[p] = 0.0
                    entry_prices[p] = 0.0
                    trailing_troughs[p] = 1e9

            # Record state outputs
            st = pos_state[p]
            regime_out[p][i] = 1.0 if st > 0 else (-1.0 if st < 0 else 0.0)
            stops_out[p][i] = current_stops[p]
            tranche_out[p][i] = float(st)

            # Cumulative tranche leverage
            if st == 1:
                weights_out[p][i] = cfg.tranche_1_leverage
            elif st == 2:
                weights_out[p][i] = cfg.tranche_1_leverage + cfg.tranche_2_leverage
            elif st == 3:
                weights_out[p][i] = cfg.tranche_1_leverage + cfg.tranche_2_leverage + cfg.tranche_3_leverage
            elif st == -1:
                weights_out[p][i] = -cfg.bear_short_leverage
            else:
                weights_out[p][i] = 0.0

    # Step 5: Map 4h signals forward to 1h bars
    final_signals: Dict[str, AssetSignals] = {}
    ts_1h = common_index

    for p in universe:
        df_sig_4h = pd.DataFrame({
            "regime": regime_out[p],
            "target_weight": weights_out[p],
            "stop_loss_px": stops_out[p],
            "atr": atr_4h_map[p],
            "tranche": tranche_out[p],
            "sma_20": sma_20_map[p],
        }, index=ref_4h_idx)

        df_sig_1h = df_sig_4h.reindex(ts_1h).ffill().fillna(0.0)

        final_signals[p] = AssetSignals(
            product_id=p,
            timestamps=ts_1h,
            close=market_data_1h[p].reindex(ts_1h)["close"].to_numpy(dtype=np.float64),
            high=market_data_1h[p].reindex(ts_1h)["high"].to_numpy(dtype=np.float64),
            low=market_data_1h[p].reindex(ts_1h)["low"].to_numpy(dtype=np.float64),
            regime=df_sig_1h["regime"].to_numpy(dtype=np.float64),
            target_weight=df_sig_1h["target_weight"].to_numpy(dtype=np.float64),
            stop_loss_px=df_sig_1h["stop_loss_px"].to_numpy(dtype=np.float64),
            atr=df_sig_1h["atr"].to_numpy(dtype=np.float64),
            funding_rate=funding_data_1h.get(p, pd.Series(0.0, index=ts_1h)).reindex(ts_1h).ffill().fillna(0.0).to_numpy(dtype=np.float64),
            tranche=df_sig_1h["tranche"].to_numpy(dtype=np.float64),
            sma_20=df_sig_1h["sma_20"].to_numpy(dtype=np.float64),
        )


    return PortfolioSignals(
        universe=universe,
        timestamps=ts_1h,
        asset_signals=final_signals,
    )
