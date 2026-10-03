"""scratch/test_parabolic_take_profit.py — Test Parabolic Climax Trimming & Trailing Stops.

Hypothesis: Crypto trends end with parabolic blow-off moves. Trimming 50% of the position
when price extends > 3.0 ATR above the 20-day moving average locks in peak gains and
prevents the 20-30% give-back on trend exhaustion, dramatically increasing net return
and reducing peak-to-trough drawdown.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data

def run_parabolic_engine(
    bundle,
    base_lev=0.30,
    climax_atr_mult=3.0, # Trim 50% if (px - sma20) > climax_atr_mult * ATR
    exit_bars=30         # 30 4h bars = 5 days Donchian low
):
    universe = list(bundle.market_data_1h.keys())
    common_idx = bundle.market_data_1h["BTC-PERP"].index
    n_bars = len(common_idx)
    inst_specs = DEFAULT_PORTFOLIO_INSTRUMENTS

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()

    # 4h Resample
    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    # Channels
    up_4h = h_4h.rolling(120, min_periods=60).max().shift(1)
    exit_low_4h = l_4h.rolling(exit_bars, min_periods=15).min().shift(1)

    # SMA 20 on 4h (80 hours = 3.3 days)
    sma20_4h = c_4h.rolling(20, min_periods=10).mean().shift(1)

    # ATR 14 on 4h
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df_4h = pd.DataFrame(atr_4h, index=c_4h.index)

    # 30d Momentum
    roc_4h = (c_4h / c_4h.shift(180) - 1.0).shift(1)

    # BTC Regime (200d SMA)
    btc_sma = c_4h["BTC-PERP"].rolling(1200, min_periods=120).mean().shift(1)
    is_bull_4h = (c_4h["BTC-PERP"] > btc_sma).shift(1).fillna(False)

    # Map to 1h
    up_1h = up_4h.reindex(common_idx).ffill()
    exit_low_1h = exit_low_4h.reindex(common_idx).ffill()
    sma20_1h = sma20_4h.reindex(common_idx).ffill()
    atr_1h = atr_df_4h.reindex(common_idx).ffill()
    roc_1h = roc_4h.reindex(common_idx).ffill()
    is_bull_1h = is_bull_4h.reindex(common_idx).ffill()

    # Simulation
    cash = 10000.0
    equity_curve = np.zeros(n_bars)

    pos_qty = {p: 0.0 for p in universe}
    pos_entry = {p: 0.0 for p in universe}
    pos_trimmed = {p: False for p in universe}

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
                hi = h_df[p].iloc[t]
                stop = exit_low_1h[p].iloc[t]

                # Normal Stop hit
                if lo <= stop and stop > 0:
                    exit_px = min(px, stop)
                    spec = inst_specs.get(p, InstrumentSpec(p))
                    fee = abs(q) * exit_px * spec.taker_fee_rate
                    pnl = (exit_px - pos_entry[p]) * q - fee
                    cash += (exit_px - pos_entry[p]) * q - fee
                    trades.append((p, current_time, pos_entry[p], exit_px, pnl, "STOP"))
                    pos_qty[p] = 0.0
                    pos_trimmed[p] = False
                    continue

                # Parabolic Climax Trim Check (on 4h boundary)
                if current_time.hour % 4 == 0 and not pos_trimmed[p] and climax_atr_mult > 0:
                    sma = sma20_1h[p].iloc[t]
                    atr = atr_1h[p].iloc[t]
                    if not np.isnan(sma) and not np.isnan(atr) and atr > 0:
                        extension = px - sma
                        if extension >= climax_atr_mult * atr and px > pos_entry[p]:
                            # Trim 50%
                            trim_qty = spec.floor_to_lot(q * 0.5)
                            if trim_qty * px >= spec.min_notional and (q - trim_qty) * px >= spec.min_notional:
                                fee = trim_qty * px * spec.maker_fee_rate
                                pnl = (px - pos_entry[p]) * trim_qty - fee
                                cash += (px - pos_entry[p]) * trim_qty - fee
                                trades.append((p, current_time, pos_entry[p], px, pnl, "TRIM_50"))
                                pos_qty[p] = q - trim_qty
                                pos_trimmed[p] = True

                unrealized += (px - pos_entry[p]) * pos_qty[p]

        current_equity = cash + unrealized
        equity_curve[t] = current_equity

        # 2. Entries on 4h boundary
        if current_time.hour % 4 == 0 and current_equity > 100.0:
            is_bull = is_bull_1h.iloc[t]

            # In Bear: Liquidate all
            if not is_bull:
                for p in universe:
                    if abs(pos_qty[p]) > 1e-9:
                        px = c_df[p].iloc[t]
                        spec = inst_specs.get(p, InstrumentSpec(p))
                        fee = abs(pos_qty[p]) * px * spec.maker_fee_rate
                        pnl = (px - pos_entry[p]) * pos_qty[p] - fee
                        cash += (px - pos_entry[p]) * pos_qty[p] - fee
                        trades.append((p, current_time, pos_entry[p], px, pnl, "BEAR_EXIT"))
                        pos_qty[p] = 0.0
                        pos_trimmed[p] = False
                unrealized = 0.0
                current_equity = cash

            # In Bull: Top 2 Momentum Leaders Breakouts
            else:
                active_count = sum(1 for p in universe if abs(pos_qty[p]) > 1e-9)
                slots = 2 - active_count

                if slots > 0:
                    rocs = {p: roc_1h[p].iloc[t] for p in universe}
                    ranked = sorted(universe, key=lambda x: rocs[x], reverse=True)

                    for p in ranked:
                        if slots <= 0:
                            break
                        if abs(pos_qty[p]) > 1e-9:
                            continue

                        px = c_df[p].iloc[t]
                        hi_up = up_1h[p].iloc[t]

                        if px > hi_up and rocs[p] > 0:
                            spec = inst_specs.get(p, InstrumentSpec(p))
                            notional = current_equity * (base_lev / 2.0)
                            q = spec.floor_to_lot(notional / px)

                            if q * px >= spec.min_notional:
                                fee = q * px * spec.maker_fee_rate
                                cash -= fee
                                pos_qty[p] = q
                                pos_entry[p] = px
                                pos_trimmed[p] = False
                                slots -= 1

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
    print("=== TESTING PARABOLIC CLIMAX TRIMMING ===")
    
    for lev in [0.25, 0.30, 0.35, 0.40]:
        for climax_atr in [0.0, 2.5, 3.0, 3.5, 4.0]:
            for exit_b in [20, 30]:
                eq, mult, dd, sh, n, wr = run_parabolic_engine(
                    bundle,
                    base_lev=lev,
                    climax_atr_mult=climax_atr,
                    exit_bars=exit_b
                )
                print(f"Lev={lev:.2f} | Climax={climax_atr:3.1f}ATR | Exit={exit_b}b => Final=${eq:,.2f} ({mult:5.2f}x) | Max DD={dd:5.2%} | Sharpe={sh:4.2f} | Trades={n:3d} | WR={wr:4.1%}")

if __name__ == "__main__":
    main()
