"""scratch/test_intraday_asymmetry.py — High-Frequency Intraday Asymmetric Compounding Engine.

Architecture:
1. Hourly execution across 9 crypto perps (46,077 bars, 2021-2026).
2. Entry: 24h High Breakout confirmed by Volume Spike (> 1.5x 24h Vol SMA) & Daily Trend Alignment.
3. Tight Risk Budgeting: 1.0% (1R) risk per trade based on 1h ATR (stop distance ~1.0 - 1.5%).
4. Asymmetric Profit Ratchets:
   - At +1.5R: Move stop to Breakeven (+0.1R fees). Risk = 0%.
   - At +3.0R: Move stop to +1.5R locked profit.
   - At +5.0R: Move stop to +3.5R locked profit.
   - Trail behind 1.5 ATR once in high profit.
5. Delta-Neutral Funding Basis on Idle Cash:
   - Idle cash earns annualized 15% APR (hourly compounding: 0.15 / 8760).
6. Zero lookahead: all signals on closed 1h bars, stop breaches evaluated on intrabar low/high.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data

def run_intraday_engine(
    bundle,
    risk_pct=0.01,        # 1.0% risk per trade (1R)
    atr_stop_mult=1.2,    # Stop distance = 1.2x 1h ATR
    vol_spike_thresh=1.4, # Volume > 1.4x 24h Volume SMA
    be_r=1.5,             # Move stop to BE at +1.5R
    r_step=1.5,           # Ratchet step size
    idle_yield_apr=0.15,  # 15% APR on unallocated cash
    max_active=3,         # Max 3 concurrent positions
    lookback_high=24      # 24h breakout
):
    universe = list(bundle.market_data_1h.keys())
    common_idx = bundle.market_data_1h["BTC-PERP"].index
    n_bars = len(common_idx)
    inst_specs = DEFAULT_PORTFOLIO_INSTRUMENTS

    # Pre-align price and volume data
    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()
    v_df = pd.DataFrame({p: bundle.market_data_1h[p]["volume"] for p in universe}, index=common_idx).ffill().bfill()

    # Precompute indicators (strictly shifted by 1 to prevent lookahead)
    hi_chans = h_df.rolling(lookback_high, min_periods=12).max().shift(1)
    
    # 24h Volume SMA
    vol_sma = v_df.rolling(24, min_periods=12).mean().shift(1)

    # 14-period 1h ATR
    atr_df = {}
    for p in universe:
        tr = np.maximum(h_df[p] - l_df[p], np.maximum((h_df[p] - c_df[p].shift(1)).abs(), (l_df[p] - c_df[p].shift(1)).abs()))
        atr_df[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df = pd.DataFrame(atr_df, index=common_idx)

    # Daily Trend: 24h EMA 20 vs 24h EMA 50
    ema20_df = c_df.ewm(span=24*2).mean().shift(1)
    ema50_df = c_df.ewm(span=24*5).mean().shift(1)
    is_asset_uptrend = (ema20_df > ema50_df)

    # BTC Macro Regime (200d = 4800h SMA)
    btc_sma200d = c_df["BTC-PERP"].rolling(4800, min_periods=480).mean().shift(1)
    is_btc_bull = (c_df["BTC-PERP"] > btc_sma200d).fillna(False)

    # 7-day Relative Momentum (168h ROC)
    roc_7d = (c_df / c_df.shift(168) - 1.0).shift(1)

    # Hourly idle yield
    hourly_idle_rate = idle_yield_apr / 8760.0

    # Simulation State
    cash = 10000.0
    equity_curve = np.zeros(n_bars)

    pos_qty = {p: 0.0 for p in universe}
    pos_entry = {p: 0.0 for p in universe}
    pos_stop = {p: 0.0 for p in universe}
    pos_r_dist = {p: 0.0 for p in universe}
    pos_peak = {p: 0.0 for p in universe}
    pos_ratchet_level = {p: 0 for p in universe}

    trades = []

    # Warmup 4800 bars (~200 days)
    for t in range(4800, n_bars):
        current_time = common_idx[t]

        # 1. Evaluate open positions & stops on 1h bar
        unrealized = 0.0
        used_margin = 0.0

        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                px = c_df[p].iloc[t]
                lo = l_df[p].iloc[t]
                hi = h_df[p].iloc[t]
                entry = pos_entry[p]
                stop = pos_stop[p]
                r_dist = pos_r_dist[p]

                if hi > pos_peak[p]:
                    pos_peak[p] = hi

                # Check Stop Hit
                if lo <= stop:
                    exit_px = min(px, stop)
                    spec = inst_specs.get(p, InstrumentSpec(p))
                    fee = q * exit_px * spec.taker_fee_rate
                    realized_pnl = (exit_px - entry) * q - fee
                    cash += (exit_px - entry) * q - fee
                    r_mult = (exit_px - entry) / max(r_dist, 1e-6)
                    trades.append((p, current_time, entry, exit_px, realized_pnl, r_mult))
                    pos_qty[p] = 0.0
                    pos_stop[p] = 0.0
                    continue

                # Multi-Tier Asymmetric Profit Ratchet
                paper_r = (hi - entry) / max(r_dist, 1e-6)

                # Tier 1: Breakeven (+0.1R buffer) at be_r
                if paper_r >= be_r and pos_ratchet_level[p] < 1:
                    new_stop = entry + 0.1 * r_dist
                    if new_stop > pos_stop[p]:
                        pos_stop[p] = new_stop
                        pos_ratchet_level[p] = 1

                # Tier 2: Lock +1.5R at +3.0R
                if paper_r >= (be_r + r_step) and pos_ratchet_level[p] < 2:
                    new_stop = entry + 1.5 * r_dist
                    if new_stop > pos_stop[p]:
                        pos_stop[p] = new_stop
                        pos_ratchet_level[p] = 2

                # Tier 3: Lock +3.5R at +5.0R
                if paper_r >= (be_r + 2 * r_step) and pos_ratchet_level[p] < 3:
                    new_stop = entry + 3.5 * r_dist
                    if new_stop > pos_stop[p]:
                        pos_stop[p] = new_stop
                        pos_ratchet_level[p] = 3

                # Tier 4: Trailing stop behind 1.5 ATR for massive runners
                if paper_r >= (be_r + 3 * r_step):
                    curr_atr = atr_df[p].iloc[t]
                    if not np.isnan(curr_atr) and curr_atr > 0:
                        trail_stop = pos_peak[p] - 1.5 * curr_atr
                        if trail_stop > pos_stop[p]:
                            pos_stop[p] = trail_stop
                            pos_ratchet_level[p] = 4

                unrealized += (px - entry) * q
                used_margin += q * px * 0.10 # 10% maintenance margin

        # Idle cash basis yield accrual
        idle_cash = max(0.0, cash - used_margin)
        cash += idle_cash * hourly_idle_rate

        current_equity = cash + unrealized
        equity_curve[t] = current_equity

        # 2. Check Entries on 1h bar
        if current_equity > 100.0:
            active_count = sum(1 for p in universe if abs(pos_qty[p]) > 1e-9)
            slots = max_active - active_count

            if slots > 0:
                # Rank universe by 7d relative momentum
                rocs = {p: roc_7d[p].iloc[t] for p in universe}
                ranked = sorted(universe, key=lambda x: rocs[x], reverse=True)

                for p in ranked:
                    if slots <= 0:
                        break
                    if abs(pos_qty[p]) > 1e-9:
                        continue

                    px = c_df[p].iloc[t]
                    hi_break = hi_chans[p].iloc[t]
                    atr = atr_df[p].iloc[t]
                    vol = v_df[p].iloc[t]
                    v_avg = vol_sma[p].iloc[t]

                    if np.isnan(px) or np.isnan(atr) or atr <= 0:
                        continue

                    # Entry Conditions:
                    # 1. Close broke out above 24h High
                    # 2. Volume expansion: Volume > vol_spike_thresh * 24h Vol SMA
                    # 3. Asset in daily uptrend (EMA20 > EMA50)
                    # 4. BTC in macro bull regime
                    is_bull = is_btc_bull.iloc[t]
                    uptrend = is_asset_uptrend[p].iloc[t]
                    vol_ok = (vol > vol_spike_thresh * v_avg) if v_avg > 0 else False

                    if is_bull and uptrend and vol_ok and (px > hi_break):
                        stop_dist = atr_stop_mult * atr
                        initial_stop = px - stop_dist

                        # Fixed 1R risk sizing
                        dollar_risk = current_equity * risk_pct
                        q_target = dollar_risk / stop_dist

                        # Cap position leverage to 1.5x equity max
                        max_q = (current_equity * 1.5) / px
                        q_target = min(q_target, max_q)

                        spec = inst_specs.get(p, InstrumentSpec(p))
                        q = spec.floor_to_lot(q_target)

                        if q * px >= spec.min_notional:
                            fee = q * px * spec.maker_fee_rate
                            cash -= fee
                            pos_qty[p] = q
                            pos_entry[p] = px
                            pos_stop[p] = initial_stop
                            pos_r_dist[p] = stop_dist
                            pos_peak[p] = px
                            pos_ratchet_level[p] = 0
                            slots -= 1

    eq_active = equity_curve[4800:]
    peaks = np.maximum.accumulate(eq_active)
    dds = (peaks - eq_active) / np.maximum(peaks, 1e-6)
    max_dd = float(np.max(dds))
    final_eq = float(eq_active[-1])
    mult = final_eq / 10000.0

    rets = pd.Series(eq_active).pct_change().fillna(0.0)
    sharpe = float((rets.mean() / (rets.std() + 1e-8)) * math.sqrt(8760.0))

    wins = [t for t in trades if t[4] > 0]
    wr = len(wins) / max(len(trades), 1)
    avg_r = np.mean([t[5] for t in trades]) if trades else 0.0

    # In-Sample vs Out-of-Sample metrics
    split_t = 4800 + int((n_bars - 4800) * 0.5) # approx 2024 split
    eq_is = equity_curve[4800:split_t]
    eq_oos = equity_curve[split_t:]

    mult_is = eq_is[-1] / 10000.0 if len(eq_is) > 0 else 1.0
    mult_oos = eq_oos[-1] / eq_oos[0] if len(eq_oos) > 0 else 1.0

    p_is = np.maximum.accumulate(eq_is)
    dd_is = float(np.max((p_is - eq_is) / np.maximum(p_is, 1e-6))) if len(p_is) > 0 else 0.0

    p_oos = np.maximum.accumulate(eq_oos)
    dd_oos = float(np.max((p_oos - eq_oos) / np.maximum(p_oos, 1e-6))) if len(p_oos) > 0 else 0.0

    return final_eq, mult, max_dd, sharpe, len(trades), wr, avg_r, mult_is, dd_is, mult_oos, dd_oos

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")
    print("=== TESTING INTRADAY ASYMMETRIC ENGINE (1h BARS, STRICT 1R RISK) ===")

    for risk in [0.008, 0.010, 0.012, 0.015]:
        for stop_m in [1.0, 1.2, 1.5]:
            for vol_spk in [1.2, 1.5, 1.8]:
                eq, mult, dd, sh, n, wr, ar, m_is, dd_is, m_oos, dd_oos = run_intraday_engine(
                    bundle,
                    risk_pct=risk,
                    atr_stop_mult=stop_m,
                    vol_spike_thresh=vol_spk,
                    be_r=1.5,
                    r_step=1.5,
                    idle_yield_apr=0.15,
                    max_active=3,
                    lookback_high=24
                )
                print(f"Risk={risk*100:4.1f}% | Stop={stop_m:3.1f}ATR | Vol={vol_spk:3.1f}x => Final=${eq:10,.2f} ({mult:5.2f}x) | Max DD={dd:5.2%} | Sh={sh:4.2f} | Trades={n:3d} | WR={wr:4.1%} | IS: {m_is:4.2f}x (DD {dd_is:4.1%}) | OOS: {m_oos:4.2f}x (DD {dd_oos:4.1%})")

if __name__ == "__main__":
    main()
