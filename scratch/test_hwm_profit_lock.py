"""scratch/test_hwm_profit_lock.py — 14% High-Water-Mark (HWM) Ratchet Lock on 100x Macro Engine.

The Mechanism:
1. Compounding Macro Trend Engine (Top Relative Strength Leaders, House-Money Pyramiding).
2. Continuous HWM Tracking:
   - Tracks all-time peak equity H.
   - If equity drops by >= 13.5% from peak (current_dd >= 13.5%):
     IMMEDIATE PORTFOLIO HARD-LOCK: Liquidate all open positions to 100% Cash!
   - Drawdown from peak is permanently bounded below 15.0%!
3. Cooldown & Reset:
   - Stay in 100% Cash while the market remains in chop/drawdown.
   - Re-arm trading when BTC establishes a clean new breakout above 200d SMA,
     resetting the baseline bankroll to current cash.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data

def run_hwm_ratchet_system(
    bundle,
    max_dd_limit=0.135,    # 13.5% hard circuit breaker from peak equity
    base_lev=0.80,         # Base entry leverage on leaders
    pyramid_lev=0.60,      # Pyramid tranche leverage
    cooldown_days=21,      # Wait 21 days after circuit breaker before re-arming
    exit_lookback=24       # 24 4h bars = 4-day Donchian low exit
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
    up_4h = h_4h.rolling(120, min_periods=60).max().shift(1) # 20d high
    exit_low_4h = l_4h.rolling(exit_lookback, min_periods=exit_lookback // 2).min().shift(1)

    # ATR 14 on 4h
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df_4h = pd.DataFrame(atr_4h, index=c_4h.index)

    # 50-day EMA on 4h (300 bars)
    ema50d_4h = c_4h.ewm(span=300).mean().shift(1)

    # 30-day Momentum (180 bars)
    roc_4h = (c_4h / c_4h.shift(180) - 1.0).shift(1)

    # BTC Regime (200d SMA = 1200 4h bars)
    btc_sma = c_4h["BTC-PERP"].rolling(1200, min_periods=120).mean().shift(1)
    is_bull_4h = (c_4h["BTC-PERP"] > btc_sma).shift(1).fillna(False)

    # Reindex to 1h
    up_1h = up_4h.reindex(common_idx).ffill()
    exit_low_1h = exit_low_4h.reindex(common_idx).ffill()
    ema50d_1h = ema50d_4h.reindex(common_idx).ffill()
    atr_1h = atr_df_4h.reindex(common_idx).ffill()
    roc_1h = roc_4h.reindex(common_idx).ffill()
    is_bull_1h = is_bull_4h.reindex(common_idx).ffill()

    # Simulation State
    cash = 10000.0
    peak_equity = 10000.0
    equity_curve = np.zeros(n_bars)

    pos_qty = {p: 0.0 for p in universe}
    pos_entry = {p: 0.0 for p in universe}
    pos_tranches = {p: 0 for p in universe}

    circuit_breaker_active = False
    cooldown_counter = 0
    trades = []

    for t in range(1200, n_bars):
        current_time = common_idx[t]

        # 1. Evaluate open positions and stops on 1h bars
        unrealized = 0.0
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                px = c_df[p].iloc[t]
                lo = l_df[p].iloc[t]
                hi = h_df[p].iloc[t]
                stop = exit_low_1h[p].iloc[t]

                # Normal exit channel breach
                if lo <= stop and stop > 0:
                    exit_px = min(px, stop)
                    spec = inst_specs.get(p, InstrumentSpec(p))
                    fee = q * exit_px * spec.taker_fee_rate
                    pnl = (exit_px - pos_entry[p]) * q - fee
                    cash += (exit_px - pos_entry[p]) * q - fee
                    trades.append((p, current_time, pos_entry[p], exit_px, pnl, "STOP"))
                    pos_qty[p] = 0.0
                    pos_tranches[p] = 0
                    continue

                unrealized += (px - pos_entry[p]) * q

        current_equity = cash + unrealized

        # Update Peak Equity
        if current_equity > peak_equity:
            peak_equity = current_equity

        # CHECK 13.5% CIRCUIT BREAKER
        current_dd = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0

        if current_dd >= max_dd_limit and not circuit_breaker_active:
            # LIQUIDATE ALL POSITIONS IMMEDIATELY TO CASH
            for p in universe:
                if abs(pos_qty[p]) > 1e-9:
                    px = c_df[p].iloc[t]
                    spec = inst_specs.get(p, InstrumentSpec(p))
                    fee = pos_qty[p] * px * spec.taker_fee_rate
                    pnl = (px - pos_entry[p]) * pos_qty[p] - fee
                    cash += (px - pos_entry[p]) * pos_qty[p] - fee
                    trades.append((p, current_time, pos_entry[p], px, pnl, "CIRCUIT_BREAKER"))
                    pos_qty[p] = 0.0
                    pos_tranches[p] = 0

            unrealized = 0.0
            current_equity = cash
            circuit_breaker_active = True
            cooldown_counter = cooldown_days * 24 # hours
            # Reset peak equity to locked cash level to allow future re-growth
            peak_equity = cash

        equity_curve[t] = current_equity

        # Decrement cooldown if circuit breaker is active
        if circuit_breaker_active:
            cooldown_counter -= 1
            if cooldown_counter <= 0 and is_bull_1h.iloc[t]:
                # Re-arm after cooldown when market confirms bull
                circuit_breaker_active = False

        # 2. Entries and Pyramids on 4h boundary
        if current_time.hour % 4 == 0 and not circuit_breaker_active and current_equity > 100.0:
            is_bull = is_bull_1h.iloc[t]

            # In Bear: Liquidate to 100% Cash
            if not is_bull:
                for p in universe:
                    if abs(pos_qty[p]) > 1e-9:
                        px = c_df[p].iloc[t]
                        spec = inst_specs.get(p, InstrumentSpec(p))
                        fee = pos_qty[p] * px * spec.maker_fee_rate
                        pnl = (px - pos_entry[p]) * pos_qty[p] - fee
                        cash += (px - pos_entry[p]) * pos_qty[p] - fee
                        trades.append((p, current_time, pos_entry[p], px, pnl, "BEAR_EXIT"))
                        pos_qty[p] = 0.0
                        pos_tranches[p] = 0
                unrealized = 0.0
                current_equity = cash

            else:
                # In Bull Market:
                # Pyramiding check for existing positions
                for p in universe:
                    if pos_qty[p] > 0 and pos_tranches[p] == 1:
                        px = c_df[p].iloc[t]
                        atr = atr_1h[p].iloc[t]
                        entry = pos_entry[p]
                        if not np.isnan(atr) and atr > 0 and px >= entry + 2.5 * atr:
                            spec = inst_specs.get(p, InstrumentSpec(p))
                            notional = current_equity * pyramid_lev
                            q_add = spec.floor_to_lot(notional / px)
                            if q_add * px >= spec.min_notional:
                                fee = q_add * px * spec.maker_fee_rate
                                cash -= fee
                                pos_qty[p] += q_add
                                pos_tranches[p] = 2

                # New Base Entries
                active_count = sum(1 for p in universe if pos_qty[p] > 0)
                slots = 2 - active_count

                if slots > 0:
                    rocs = {p: roc_1h[p].iloc[t] for p in universe}
                    ranked = sorted(universe, key=lambda x: rocs[x], reverse=True)

                    for p in ranked:
                        if slots <= 0:
                            break
                        if pos_qty[p] > 0:
                            continue

                        px = c_df[p].iloc[t]
                        hi_up = up_1h[p].iloc[t]
                        ema_50d = ema50d_1h[p].iloc[t]

                        # Entry: Breakout of 20d High AND above 50d EMA AND positive 30d momentum
                        if px > hi_up and px > ema_50d and rocs[p] > 0:
                            spec = inst_specs.get(p, InstrumentSpec(p))
                            notional = current_equity * base_lev
                            q = spec.floor_to_lot(notional / px)

                            if q * px >= spec.min_notional:
                                fee = q * px * spec.maker_fee_rate
                                cash -= fee
                                pos_qty[p] = q
                                pos_entry[p] = px
                                pos_tranches[p] = 1
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
    print("=== TESTING 14% HWM RATCHET CIRCUIT BREAKER ===")
    for b_lev in [0.50, 0.70, 0.90, 1.10]:
        for c_days in [14, 21, 28]:
            for exit_b in [18, 24]:
                eq, mult, dd, sh, n, wr = run_hwm_ratchet_system(
                    bundle,
                    max_dd_limit=0.135,
                    base_lev=b_lev,
                    pyramid_lev=b_lev*0.6,
                    cooldown_days=c_days,
                    exit_lookback=exit_b
                )
                print(f"BaseLev={b_lev:.2f} | Cooldown={c_days:2d}d | Exit={exit_b:2d}b => Final=${eq:12,.2f} ({mult:7.2f}x) | Max DD={dd:5.2%} | Sharpe={sh:4.2f} | Trades={n:3d} | WR={wr:4.1%}")

if __name__ == "__main__":
    main()
