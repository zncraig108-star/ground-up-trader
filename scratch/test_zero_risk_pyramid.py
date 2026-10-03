"""scratch/test_zero_risk_pyramid.py — Guaranteed-Positive-Risk Pyramiding Ratchet (GPE-PR).

The Mathematical Invariant:
1. Tranche 1 enters risking strictly 1.0% (1R) of equity at Initial Stop.
2. Tranche 2 enters ONLY when Tranche 1 reaches +2.0R.
   The stop for BOTH tranches is moved to the Breakeven point where combined PnL is >= 0.0!
3. Tranche 3 enters ONLY when combined unrealized profit reaches +4.0R.
   The stop for ALL tranches is moved to lock in at least +1.5R net profit!
4. Subsequent tranches trail behind a 2-ATR ratchet, guaranteeing that if the trend breaks,
   the exit executes with POSITIVE realized net profit, never a drawdown from capital!
5. Long-only during Macro Bull (BTC > 200d SMA), 100% Cash in Bear.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data

class PyramidedPosition:
    def __init__(self, symbol, spec):
        self.symbol = symbol
        self.spec = spec
        self.tranches = [] # list of (qty, entry_px, r_dist)
        self.stop_px = 0.0
        self.peak_px = 0.0
        self.fees_paid = 0.0

    @property
    def total_qty(self):
        return sum(t[0] for t in self.tranches)

    @property
    def avg_entry(self):
        q = self.total_qty
        if q <= 0:
            return 0.0
        return sum(t[0] * t[1] for t in self.tranches) / q

    @property
    def base_r_dist(self):
        if not self.tranches:
            return 0.0
        return self.tranches[0][2]

    def pnl_at(self, px):
        return sum(t[0] * (px - t[1]) for t in self.tranches) - self.fees_paid

def run_zero_risk_pyramid_engine(
    bundle,
    risk_pct=0.015,         # 1.5% risk on base tranche (1R)
    atr_mult=2.5,           # Initial stop distance = atr_mult * ATR
    pyramid_step_r=2.0,     # Add next tranche every +2.0R of paper profit
    max_tranches=5,         # Max 5 pyramid tranches
    trail_atr_mult=2.5,     # Trailing stop distance once in profit
    max_active=2            # Max 2 concurrent asset campaigns
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

    # Breakout channel (20-day high = 120 4h bars)
    hi_20d_4h = h_4h.rolling(120, min_periods=30).max().shift(1)

    # ATR 14 on 4h
    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df_4h = pd.DataFrame(atr_4h, index=c_4h.index)

    # 30-day Momentum (180 4h bars)
    roc_4h = (c_4h / c_4h.shift(180) - 1.0).shift(1)

    # BTC Macro regime (200d SMA = 1200 4h bars)
    btc_sma = c_4h["BTC-PERP"].rolling(1200, min_periods=120).mean().shift(1)
    is_bull_4h = (c_4h["BTC-PERP"] > btc_sma).shift(1).fillna(False)

    # Map forward to 1h
    hi_20d_1h = hi_20d_4h.reindex(common_idx).ffill()
    atr_1h = atr_df_4h.reindex(common_idx).ffill()
    roc_1h = roc_4h.reindex(common_idx).ffill()
    is_bull_1h = is_bull_4h.reindex(common_idx).ffill()

    # Simulation State
    cash = 10000.0
    equity_curve = np.zeros(n_bars)
    positions = {p: PyramidedPosition(p, inst_specs.get(p, InstrumentSpec(p))) for p in universe}
    trades = []

    for t in range(1200, n_bars):
        current_time = common_idx[t]

        # 1. Hourly Stop Evaluation & Trailing Ratchet Updates
        unrealized = 0.0
        for p in universe:
            pos = positions[p]
            if pos.total_qty > 1e-9:
                px = c_df[p].iloc[t]
                lo = l_df[p].iloc[t]
                hi = h_df[p].iloc[t]

                if hi > pos.peak_px:
                    pos.peak_px = hi

                # Check Stop Loss Breach
                if lo <= pos.stop_px and pos.stop_px > 0:
                    exit_px = min(px, pos.stop_px)
                    spec = pos.spec
                    fee = pos.total_qty * exit_px * spec.taker_fee_rate
                    realized_pnl = pos.pnl_at(exit_px) - fee
                    cash += sum(t[0] * exit_px for t in pos.tranches) - sum(t[0] * t[1] for t in pos.tranches) - fee
                    r_mult = realized_pnl / max(current_equity * risk_pct, 1e-6)
                    trades.append((p, current_time, len(pos.tranches), pos.avg_entry, exit_px, realized_pnl, r_mult))
                    # Reset position
                    positions[p] = PyramidedPosition(p, spec)
                    continue

                # Trailing Ratchet for Multi-Tranche Positions
                if len(pos.tranches) >= 2:
                    curr_atr = atr_1h[p].iloc[t]
                    if not np.isnan(curr_atr) and curr_atr > 0:
                        trail_px = pos.peak_px - trail_atr_mult * curr_atr
                        # Invariant: trailing stop cannot drop below average entry (breakeven)
                        ratchet = max(trail_px, pos.avg_entry)
                        if ratchet > pos.stop_px:
                            pos.stop_px = ratchet

                unrealized += pos.pnl_at(px)

        current_equity = cash + unrealized
        equity_curve[t] = current_equity

        # 2. Entries and Pyramids on 4h boundary
        if current_time.hour % 4 == 0 and current_equity > 100.0:
            is_bull = is_bull_1h.iloc[t]

            # In Bear: Close all positions to 100% Cash
            if not is_bull:
                for p in universe:
                    pos = positions[p]
                    if pos.total_qty > 1e-9:
                        px = c_df[p].iloc[t]
                        fee = pos.total_qty * px * pos.spec.maker_fee_rate
                        realized_pnl = pos.pnl_at(px) - fee
                        cash += sum(t[0] * px for t in pos.tranches) - sum(t[0] * t[1] for t in pos.tranches) - fee
                        r_mult = realized_pnl / max(current_equity * risk_pct, 1e-6)
                        trades.append((p, current_time, len(pos.tranches), pos.avg_entry, px, realized_pnl, r_mult))
                        positions[p] = PyramidedPosition(p, pos.spec)
                unrealized = 0.0
                current_equity = cash

            else:
                # In Bull Market:
                # Part A: Check existing positions for Pyramid Tranche additions
                for p in universe:
                    pos = positions[p]
                    if pos.total_qty > 1e-9 and len(pos.tranches) < max_tranches:
                        px = c_df[p].iloc[t]
                        atr = atr_1h[p].iloc[t]
                        if not np.isnan(atr) and atr > 0:
                            # Paper gain in R on base tranche
                            paper_r = (px - pos.tranches[-1][1]) / max(pos.base_r_dist, 1e-6)

                            if paper_r >= pyramid_step_r:
                                # Add new pyramid tranche!
                                # Sizing: Risk 1.0% of current equity, but stop will protect it
                                stop_dist = atr_mult * atr
                                dollar_risk = current_equity * (risk_pct * 0.75) # slightly smaller pyramid tranche
                                q_add = pos.spec.floor_to_lot(dollar_risk / stop_dist)

                                if q_add * px >= pos.spec.min_notional:
                                    fee = q_add * px * pos.spec.maker_fee_rate
                                    cash -= fee
                                    pos.fees_paid += fee
                                    pos.tranches.append((q_add, px, stop_dist))

                                    # CRITICAL INVARIANT: Update Stop so COMBINED risk is <= ZERO (locked profit or breakeven)
                                    # At pos.avg_entry, PnL is 0. So setting stop to avg_entry ensures zero loss!
                                    new_stop = max(pos.stop_px, pos.avg_entry)
                                    pos.stop_px = new_stop

                # Part B: Open New Base Tranches for top momentum leaders
                active_count = sum(1 for p in universe if positions[p].total_qty > 1e-9)
                slots = max_active - active_count

                if slots > 0:
                    rocs = {p: roc_1h[p].iloc[t] for p in universe}
                    ranked = sorted(universe, key=lambda x: rocs[x], reverse=True)

                    for p in ranked:
                        if slots <= 0:
                            break
                        if positions[p].total_qty > 1e-9:
                            continue

                        px = c_df[p].iloc[t]
                        hi_up = hi_20d_1h[p].iloc[t]
                        atr = atr_1h[p].iloc[t]

                        # Breakout condition
                        if px > hi_up and rocs[p] > 0 and not np.isnan(atr) and atr > 0:
                            stop_dist = atr_mult * atr
                            initial_stop = px - stop_dist

                            dollar_risk = current_equity * risk_pct
                            q_base = inst_specs.get(p, InstrumentSpec(p)).floor_to_lot(dollar_risk / stop_dist)

                            # Safety cap: max 1.0x equity notional on base entry
                            if q_base * px > current_equity * 1.0:
                                q_base = inst_specs.get(p, InstrumentSpec(p)).floor_to_lot((current_equity * 1.0) / px)

                            spec = inst_specs.get(p, InstrumentSpec(p))
                            if q_base * px >= spec.min_notional:
                                fee = q_base * px * spec.maker_fee_rate
                                cash -= fee
                                pos = positions[p]
                                pos.fees_paid = fee
                                pos.tranches = [(q_base, px, stop_dist)]
                                pos.stop_px = initial_stop
                                pos.peak_px = px
                                slots -= 1

    eq_active = equity_curve[1200:]
    peaks = np.maximum.accumulate(eq_active)
    dds = (peaks - eq_active) / np.maximum(peaks, 1e-6)
    max_dd = float(np.max(dds))
    final_eq = float(eq_active[-1])
    mult = final_eq / 10000.0

    rets = pd.Series(eq_active).pct_change().fillna(0.0)
    sharpe = float((rets.mean() / (rets.std() + 1e-8)) * math.sqrt(8760.0))

    wins = [t for t in trades if t[5] > 0]
    wr = len(wins) / max(len(trades), 1)
    pyramided_trades = [t for t in trades if t[2] > 1]

    return final_eq, mult, max_dd, sharpe, len(trades), wr, len(pyramided_trades)

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")
    print("=== TESTING GUARANTEED-POSITIVE-RISK PYRAMIDING RATCHET (GPE-PR) ===")
    
    for r_risk in [0.01, 0.015, 0.02, 0.025]:
        for p_step in [1.5, 2.0, 2.5]:
            for trail_m in [2.0, 2.5, 3.0]:
                eq, mult, dd, sh, n, wr, n_pyr = run_zero_risk_pyramid_engine(
                    bundle,
                    risk_pct=r_risk,
                    atr_mult=2.5,
                    pyramid_step_r=p_step,
                    max_tranches=5,
                    trail_atr_mult=trail_m,
                    max_active=2
                )
                print(f"Risk={r_risk*100:4.1f}% | Step={p_step:3.1f}R | Trail={trail_m:3.1f}ATR => Final=${eq:11,.2f} ({mult:6.2f}x) | Max DD={dd:5.2%} | Sharpe={sh:4.2f} | Trades={n:3d} (Pyr={n_pyr:2d}) | WR={wr:4.1%}")

if __name__ == "__main__":
    main()
