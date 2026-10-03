"""scratch/test_asset_trend_filter.py — Full simulation comparing Asset-Level Filters on 100x Compounding."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals
from core.portfolio_backtester import backtest_portfolio

def test_filtered_strategy(bundle, use_asset_ema=True, use_slope=True, exit_lookback=48, tranche_1_lev=0.60):
    cfg = PortfolioConfig(
        target_annual_vol=0.35, max_portfolio_leverage=3.0, max_single_asset_leverage=2.0,
        allow_bear_shorts=False, exit_lookback_bars=exit_lookback, pyramid_enabled=True,
        tranche_1_leverage=tranche_1_lev, tranche_2_leverage=tranche_1_lev*0.75,
        tranche_3_leverage=tranche_1_lev*0.55, bear_short_leverage=0.0,
    )
    
    universe = cfg.universe
    common_index = bundle.market_data_1h["BTC-PERP"].index
    data_4h = {}
    ref_4h_idx = bundle.market_data_1h[universe[0]].resample("4h", closed="left", label="left").agg({"close": "last"}).index
    n4 = len(ref_4h_idx)

    upper_chan_map = {}
    exit_chan_low = {}
    atr_4h_map = {}
    ema_50d_map = {}
    ema_slope_map = {}
    roc_30d_map = {}

    for p in universe:
        ohlcv = bundle.market_data_1h[p].reindex(common_index)
        df_4h = ohlcv.resample("4h", closed="left", label="left").agg({
            "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
        }).reindex(ref_4h_idx)
        data_4h[p] = df_4h

        c = df_4h["close"].to_numpy(dtype=np.float64)
        h = df_4h["high"].to_numpy(dtype=np.float64)
        l = df_4h["low"].to_numpy(dtype=np.float64)
        n = len(c)

        upper_chan_map[p] = pd.Series(h).rolling(cfg.trend_lookback_bars, min_periods=60).max().shift(1).to_numpy()
        exit_chan_low[p] = pd.Series(l).rolling(cfg.exit_lookback_bars, min_periods=cfg.exit_lookback_bars // 2).min().shift(1).to_numpy()

        tr = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if np.isnan(c[i]) or np.isnan(c[i - 1]):
                tr[i] = np.nan
            else:
                tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        atr_4h_map[p] = pd.Series(tr).ewm(alpha=1.0 / max(cfg.atr_period_bars, 1), adjust=False).mean().to_numpy()

        ema = pd.Series(c).ewm(span=300, adjust=False).mean().to_numpy() # 50-day EMA
        ema_50d_map[p] = ema
        ema_slope_map[p] = pd.Series(ema).diff(6).fillna(0.0).to_numpy()

        roc = (pd.Series(c) / pd.Series(c).shift(cfg.momentum_lookback_bars) - 1.0).fillna(-999.0).to_numpy()
        roc_30d_map[p] = roc

    btc_c = data_4h["BTC-PERP"]["close"].ffill().bfill().to_numpy()
    btc_sma_200 = pd.Series(btc_c).rolling(cfg.macro_btc_sma_bars, min_periods=120).mean().to_numpy()
    is_macro_bull = btc_c >= btc_sma_200

    all_rocs = np.stack([roc_30d_map[p] for p in universe], axis=0)
    ranks = np.argsort(-all_rocs, axis=0)

    is_top_leader = {}
    for idx_u, p in enumerate(universe):
        in_top_k = np.any(ranks[:cfg.max_concurrent_leaders, :] == idx_u, axis=0)
        valid_px = ~np.isnan(data_4h[p]["close"].to_numpy())
        is_top_leader[p] = in_top_k & valid_px

    # State machine
    regime_out = {p: np.zeros(n4, dtype=np.float64) for p in universe}
    weights_out = {p: np.zeros(n4, dtype=np.float64) for p in universe}
    stops_out = {p: np.zeros(n4, dtype=np.float64) for p in universe}
    tranche_out = {p: np.zeros(n4, dtype=np.float64) for p in universe}

    pos_state = {p: 0 for p in universe}
    entry_prices = {p: 0.0 for p in universe}
    trailing_peaks = {p: 0.0 for p in universe}
    current_stops = {p: 0.0 for p in universe}

    warmup = 180

    for i in range(warmup, n4):
        active_longs = sum(1 for p in universe if pos_state[p] > 0)

        for p in universe:
            c4 = data_4h[p]["close"].to_numpy()[i - 1]
            h4 = data_4h[p]["high"].to_numpy()[i - 1]
            l4 = data_4h[p]["low"].to_numpy()[i - 1]
            atr = atr_4h_map[p][i - 1]
            upper_ch = upper_chan_map[p][i - 1]
            ex_l = exit_chan_low[p][i - 1]
            is_leader = is_top_leader[p][i - 1]
            ema_50d = ema_50d_map[p][i - 1]
            slope = ema_slope_map[p][i - 1]

            if np.isnan(c4) or np.isnan(atr):
                continue

            state = pos_state[p]

            if state == 0:
                # Filter check
                pass_asset_filter = True
                if use_asset_ema and c4 <= ema_50d:
                    pass_asset_filter = False
                if use_slope and slope <= 0:
                    pass_asset_filter = False

                if is_macro_bull[i - 1] and active_longs < cfg.max_concurrent_leaders and is_leader and roc_30d_map[p][i - 1] > 0 and pass_asset_filter:
                    if c4 > upper_ch:
                        pos_state[p] = 1
                        entry_prices[p] = c4
                        trailing_peaks[p] = c4
                        current_stops[p] = ex_l if not np.isnan(ex_l) else (c4 - cfg.atr_stop_mult * atr)
                        active_longs += 1

            elif state > 0:
                trailing_peaks[p] = max(trailing_peaks[p], h4)
                peak = trailing_peaks[p]
                entry = entry_prices[p]

                if state == 1 and peak >= entry + cfg.pyramid_milestone_1_atr * atr:
                    pos_state[p] = 2
                elif state == 2 and peak >= entry + cfg.pyramid_milestone_2_atr * atr:
                    pos_state[p] = 3

                # Trailing exit channel
                current_stops[p] = ex_l if not np.isnan(ex_l) else (peak - cfg.atr_stop_mult * atr)

                # Exit if stop hit, OR if macro regime flips bear, OR if asset drops below 50d EMA
                exit_cond = l4 <= current_stops[p] or (~is_macro_bull[i - 1])
                if use_asset_ema and c4 < ema_50d:
                    exit_cond = True

                if exit_cond:
                    pos_state[p] = 0
                    current_stops[p] = 0.0
                    entry_prices[p] = 0.0
                    trailing_peaks[p] = 0.0

            # Record
            curr_state = pos_state[p]
            regime_out[p][i] = 1.0 if curr_state > 0 else 0.0
            stops_out[p][i] = current_stops[p]
            tranche_out[p][i] = float(curr_state)
            if curr_state == 1:
                weights_out[p][i] = cfg.tranche_1_leverage
            elif curr_state == 2:
                weights_out[p][i] = cfg.tranche_1_leverage + cfg.tranche_2_leverage
            elif curr_state == 3:
                weights_out[p][i] = cfg.tranche_1_leverage + cfg.tranche_2_leverage + cfg.tranche_3_leverage
            else:
                weights_out[p][i] = 0.0

    # Build 1h signals
    from core.portfolio_strategy import AssetSignals, PortfolioSignals
    asset_signals_1h = {}
    for p in universe:
        c1 = bundle.market_data_1h[p]["close"].reindex(common_index).ffill().bfill().to_numpy()
        h1 = bundle.market_data_1h[p]["high"].reindex(common_index).ffill().bfill().to_numpy()
        l1 = bundle.market_data_1h[p]["low"].reindex(common_index).ffill().bfill().to_numpy()

        r_s = pd.Series(regime_out[p], index=ref_4h_idx).reindex(common_index).ffill().fillna(0.0).to_numpy()
        w_s = pd.Series(weights_out[p], index=ref_4h_idx).reindex(common_index).ffill().fillna(0.0).to_numpy()
        s_s = pd.Series(stops_out[p], index=ref_4h_idx).reindex(common_index).ffill().fillna(0.0).to_numpy()
        t_s = pd.Series(tranche_out[p], index=ref_4h_idx).reindex(common_index).ffill().fillna(0.0).to_numpy()
        a_s = pd.Series(atr_4h_map[p], index=ref_4h_idx).reindex(common_index).ffill().fillna(0.0).to_numpy()

        asset_signals_1h[p] = AssetSignals(
            product_id=p,
            timestamps=common_index,
            close=c1,
            high=h1,
            low=l1,
            regime=r_s,
            target_weight=w_s,
            stop_loss_px=s_s,
            atr=a_s,
            funding_rate=np.zeros(len(common_index)),
            tranche=t_s
        )

    psig = PortfolioSignals(universe=universe, timestamps=common_index, asset_signals=asset_signals_1h)
    res = backtest_portfolio(psig, cfg)
    return res

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")
    print("--- TESTING ASSET 50D EMA & SLOPE FILTERS ---")
    for lev in [0.30, 0.40, 0.50, 0.60, 0.80, 1.00]:
        for exit_b in [24, 36, 48]:
            res = test_filtered_strategy(bundle, use_asset_ema=True, use_slope=True, exit_lookback=exit_b, tranche_1_lev=lev)
            print(f"Lev={lev:.2f} | Exit={exit_b}b => Final=${res.final_equity:11,.2f} ({res.final_equity/10000:6.2f}x) | Max DD={res.max_drawdown:5.2%} | Sharpe={res.sharpe:4.2f} | Trades={len(res.trades):3d}")

if __name__ == "__main__":
    main()
