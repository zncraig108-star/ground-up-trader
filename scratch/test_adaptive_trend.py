"""scratch/test_adaptive_trend.py — Multi-factor Adaptive Trend Following with Volatility Scaling & ER Filter."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data

def compute_kaufman_er(series, period=30):
    """Kaufman Efficiency Ratio = |Net Change| / Sum of Absolute Changes."""
    net_change = (series - series.shift(period)).abs()
    gross_change = (series - series.shift(1)).abs().rolling(period).sum()
    er = (net_change / np.maximum(gross_change, 1e-8)).fillna(0.0)
    return er

def test_adaptive_portfolio(bundle, base_lev=0.25, er_min=0.35, target_vol=0.20):
    universe = [p for p in bundle.market_data_1h.keys()]
    common_idx = bundle.market_data_1h["BTC-PERP"].index
    n_bars = len(common_idx)
    inst_specs = DEFAULT_PORTFOLIO_INSTRUMENTS

    # Precompute 4h resampled data and indicators
    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()

    # 4h Resample
    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    # 20-day breakout channels (120 4h bars)
    up_4h = h_4h.rolling(120, min_periods=60).max().shift(1)
    # 5-day exit channels (30 4h bars)
    exit_low_4h = l_4h.rolling(30, min_periods=15).min().shift(1)

    # 30-day Momentum (180 4h bars)
    roc_4h = (c_4h / c_4h.shift(180) - 1.0).fillna(-999.0)

    # Kaufman Efficiency Ratio (30 4h bars = 5 days)
    er_4h = pd.DataFrame({p: compute_kaufman_er(c_4h[p], period=30) for p in universe}, index=c_4h.index)

    # 7-day ATR (42 4h bars)
    tr_4h = {}
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.ewm(span=42).mean()
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    # BTC Macro regime (200d SMA = 1200 4h bars)
    btc_sma = c_4h["BTC-PERP"].rolling(1200, min_periods=120).mean().shift(1)
    is_bull_4h = (c_4h["BTC-PERP"] > btc_sma).fillna(False)

    # Map forward to 1h bars
    up_1h = up_4h.reindex(common_idx).ffill()
    exit_low_1h = exit_low_4h.reindex(common_idx).ffill()
    roc_1h = roc_4h.reindex(common_idx).ffill()
    er_1h = er_4h.reindex(common_idx).ffill()
    atr_1h = atr_df.reindex(common_idx).ffill()
    is_bull_1h = is_bull_4h.reindex(common_idx).ffill()

    # Simulation
    cash = 10000.0
    equity = 10000.0
    equity_curve = np.zeros(n_bars)

    pos_qty = {p: 0.0 for p in universe}
    pos_entry = {p: 0.0 for p in universe}
    pos_fees = {p: 0.0 for p in universe}

    trades = []

    for t in range(1200, n_bars):
        current_time = common_idx[t]

        # 1. Hourly Stop Evaluation
        unrealized = 0.0
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                px = c_df[p].iloc[t]
                lo = l_df[p].iloc[t]
                stop = exit_low_1h[p].iloc[t]

                if lo <= stop and stop > 0:
                    exit_px = min(px, stop)
                    spec = inst_specs.get(p, InstrumentSpec(p))
                    fee = abs(q) * exit_px * spec.taker_fee_rate
                    pnl = (exit_px - pos_entry[p]) * q - pos_fees[p] - fee
                    cash += (exit_px - pos_entry[p]) * q - fee
                    trades.append((p, 1, pos_entry[p], exit_px, pnl))
                    pos_qty[p] = 0.0
                    continue

                unrealized += (px - pos_entry[p]) * q

        current_equity = cash + unrealized

        # 2. Discrete 4h Rebalance
        if current_time.hour % 4 == 0 and current_equity > 100.0:
            is_bull = is_bull_1h.iloc[t]

            # In Bear: Liquidate all to cash
            if not is_bull:
                for p in universe:
                    if abs(pos_qty[p]) > 1e-9:
                        px = c_df[p].iloc[t]
                        spec = inst_specs.get(p, InstrumentSpec(p))
                        fee = abs(pos_qty[p]) * px * spec.maker_fee_rate
                        pnl = (px - pos_entry[p]) * pos_qty[p] - pos_fees[p] - fee
                        cash += (px - pos_entry[p]) * pos_qty[p] - fee
                        trades.append((p, 1, pos_entry[p], px, pnl))
                        pos_qty[p] = 0.0
                unrealized = 0.0
                current_equity = cash

            # In Bull: Select Top 2 Momentum Leaders with High Efficiency Ratio (ER > er_min)
            else:
                # Rank universe by momentum
                rocs = {p: roc_1h[p].iloc[t] for p in universe}
                ranked = sorted(universe, key=lambda x: rocs[x], reverse=True)

                # Filter leaders by positive momentum and high ER
                qualified_leaders = [
                    p for p in ranked
                    if rocs[p] > 0 and er_1h[p].iloc[t] >= er_min and c_df[p].iloc[t] > up_1h[p].iloc[t]
                ]

                # Max 2 concurrent positions
                active_leaders = qualified_leaders[:2]

                for p in universe:
                    px = c_df[p].iloc[t]
                    spec = inst_specs.get(p, InstrumentSpec(p))

                    # Open position if leader and flat
                    if p in active_leaders and abs(pos_qty[p]) < 1e-9:
                        # Volatility-scaled position size:
                        # annualized asset vol = (ATR / px) * sqrt(365 * 6)
                        atr = atr_1h[p].iloc[t]
                        if not np.isnan(atr) and atr > 0:
                            ann_vol = (atr / px) * math.sqrt(2190.0)
                            # Target vol scaling
                            vol_weight = min(base_lev, base_lev * (target_vol / max(ann_vol, 0.10)))
                            notional = current_equity * vol_weight
                            q = spec.floor_to_lot(notional / px)

                            if q * px >= spec.min_notional:
                                fee = q * px * spec.maker_fee_rate
                                cash -= fee
                                pos_qty[p] = q
                                pos_entry[p] = px
                                pos_fees[p] = fee

        equity_curve[t] = current_equity

    eq_active = equity_curve[1200:]
    peaks = np.maximum.accumulate(eq_active)
    dds = (peaks - eq_active) / np.maximum(peaks, 1e-6)
    max_dd = float(np.max(dds))
    final_eq = float(eq_active[-1])
    mult = final_eq / 10000.0

    rets = pd.Series(eq_active).pct_change().fillna(0.0)
    sharpe = float((rets.mean() / (rets.std() + 1e-8)) * math.sqrt(8760.0))

    wins = [t for t in trades if t[4] > 0]
    wr = len(wins) / max(len(trades), 1)

    return final_eq, mult, max_dd, sharpe, len(trades), wr

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")

    print("--- TESTING ADAPTIVE TREND + ER FILTER + VOLATILITY SIZING ---")
    for lev in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        for er in [0.25, 0.30, 0.35, 0.40]:
            eq, mult, dd, sh, n, wr = test_adaptive_portfolio(bundle, base_lev=lev, er_min=er, target_vol=0.25)
            print(f"Lev={lev:.2f} | ER_Min={er:.2f} => Final=${eq:,.2f} ({mult:5.2f}x) | Max DD={dd:5.2%} | Sharpe={sh:4.2f} | Trades={n:3d} | WR={wr:4.1%}")

if __name__ == "__main__":
    main()
