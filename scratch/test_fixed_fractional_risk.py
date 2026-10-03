"""scratch/test_fixed_fractional_risk.py — Testing Fixed Fractional ATR Risk Sizing & Ratchet Stops."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals

def simulate_fractional_risk(signals, cfg, risk_pct=0.015, stop_atr_mult=2.0, trail_atr_mult=1.5, max_dd_brake=0.12):
    """
    Simulate with:
    - Fixed risk per trade: risk_pct of equity (e.g. 1.5%).
    - Stop distance: stop_atr_mult * ATR.
    - Ratchet trailing stop: once in profit, trail by trail_atr_mult * ATR from peak.
    - Portfolio Drawdown Brake: scale down risk if in drawdown > 5%.
    """
    ts = signals.timestamps
    universe = signals.universe
    n_bars = len(ts)
    inst_specs = DEFAULT_PORTFOLIO_INSTRUMENTS

    cash = float(cfg.initial_equity)
    initial_equity = cash
    equity_curve = np.zeros(n_bars, dtype=np.float64)

    pos_qty = {p: 0.0 for p in universe}
    pos_entry_px = {p: 0.0 for p in universe}
    pos_peak_px = {p: 0.0 for p in universe}
    pos_stop_px = {p: 0.0 for p in universe}
    pos_entry_fees = {p: 0.0 for p in universe}

    trades = []
    peak_equity = initial_equity

    for t in range(n_bars):
        current_time = ts[t]

        # 1. Update equity & check stops on 1h bars
        unrealized = 0.0
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                asig = signals.asset_signals[p]
                px = asig.close[t]
                hi = asig.high[t]
                lo = asig.low[t]
                atr = asig.atr[t]

                # Update peak price for trailing
                if q > 0:
                    pos_peak_px[p] = max(pos_peak_px[p], hi)
                    # Ratchet stop: trail from peak if peak > entry + 1.5*atr
                    if pos_peak_px[p] > pos_entry_px[p] + 1.5 * atr:
                        pos_stop_px[p] = max(pos_stop_px[p], pos_peak_px[p] - trail_atr_mult * atr)
                    
                    # Stop hit?
                    if lo <= pos_stop_px[p]:
                        exit_px = min(px, pos_stop_px[p])
                        fee = abs(q) * exit_px * 0.0005
                        pnl = (exit_px - pos_entry_px[p]) * q - fee - pos_entry_fees[p]
                        cash += (exit_px - pos_entry_px[p]) * q - fee
                        trades.append((p, 1, pos_entry_px[p], exit_px, pnl))
                        pos_qty[p] = 0.0
                        continue

                elif q < 0:
                    pos_peak_px[p] = min(pos_peak_px[p], lo)
                    if pos_peak_px[p] < pos_entry_px[p] - 1.5 * atr:
                        pos_stop_px[p] = min(pos_stop_px[p], pos_peak_px[p] + trail_atr_mult * atr)
                    
                    if hi >= pos_stop_px[p]:
                        exit_px = max(px, pos_stop_px[p])
                        fee = abs(q) * exit_px * 0.0005
                        pnl = (exit_px - pos_entry_px[p]) * q - fee - pos_entry_fees[p]
                        cash += (exit_px - pos_entry_px[p]) * q - fee
                        trades.append((p, -1, pos_entry_px[p], exit_px, pnl))
                        pos_qty[p] = 0.0
                        continue

                unrealized += (px - pos_entry_px[p]) * q

        current_equity = cash + unrealized
        peak_equity = max(peak_equity, current_equity)
        current_dd = (peak_equity - current_equity) / max(peak_equity, 1e-6)

        # 2. Discrete Rebalancing on 4h bars
        if current_time.hour % 4 == 0 and current_equity > 100:
            # Drawdown throttle
            if current_dd > 0.05:
                # Linearly scale down to 0 at max_dd_brake (e.g. 12%)
                dd_scale = max(0.0, (max_dd_brake - current_dd) / (max_dd_brake - 0.05))
            else:
                dd_scale = 1.0

            for p in universe:
                asig = signals.asset_signals[p]
                spec = inst_specs.get(p, InstrumentSpec(p))
                target_tr = int(asig.tranche[t])
                prev_tr = int(asig.tranche[t - 1]) if t > 0 else 0
                px = asig.close[t]
                atr = asig.atr[t]

                if np.isnan(px) or px <= 0 or np.isnan(atr) or atr <= 0:
                    continue

                # Signal Exit
                if target_tr == 0 and abs(pos_qty[p]) > 1e-9:
                    q = pos_qty[p]
                    fee = abs(q) * px * 0.0002
                    pnl = (px - pos_entry_px[p]) * q - fee - pos_entry_fees[p]
                    cash += (px - pos_entry_px[p]) * q - fee
                    trades.append((p, 1 if q > 0 else -1, pos_entry_px[p], px, pnl))
                    pos_qty[p] = 0.0

                # Entry
                elif pos_qty[p] == 0.0 and target_tr != 0 and prev_tr == 0 and dd_scale > 0.01:
                    # Position sizing based on Fixed Risk Fraction
                    dollar_risk = current_equity * risk_pct * dd_scale
                    stop_dist = stop_atr_mult * atr
                    # qty * stop_dist == dollar_risk => qty = dollar_risk / stop_dist
                    q_raw = dollar_risk / stop_dist
                    # cap max notional at 1.5x equity
                    max_q = (current_equity * 1.5) / px
                    q = min(q_raw, max_q)
                    q = spec.floor_to_lot(q)

                    if q * px >= spec.min_notional:
                        fee = q * px * 0.0002
                        cash -= fee
                        pos_qty[p] = q if target_tr > 0 else -q
                        pos_entry_px[p] = px
                        pos_peak_px[p] = px
                        pos_stop_px[p] = px - stop_dist if target_tr > 0 else px + stop_dist
                        pos_entry_fees[p] = fee

        equity_curve[t] = max(0.0, current_equity)

    peaks = np.maximum.accumulate(equity_curve)
    dds = (peaks - equity_curve) / np.maximum(peaks, 1e-6)
    max_dd = float(np.max(dds))
    final_eq = float(equity_curve[-1])

    return final_eq, max_dd, len(trades)

def main():
    cfg = PortfolioConfig()
    bundle = load_portfolio_data(cfg, data_dir="data")
    signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, cfg)

    print("Testing combinations of risk_pct and stop multipliers:")
    for risk in [0.01, 0.015, 0.02, 0.025, 0.03]:
        for stop_m in [1.5, 2.0, 2.5]:
            for trail_m in [1.2, 1.5, 2.0]:
                eq, dd, n_tr = simulate_fractional_risk(signals, cfg, risk_pct=risk, stop_atr_mult=stop_m, trail_atr_mult=trail_m)
                mult = eq / 10000.0
                if dd <= 0.20:
                    print(f"Risk={risk:.1%} | Stop={stop_m:.1f}ATR | Trail={trail_m:.1f}ATR => Final=${eq:,.2f} ({mult:.2f}x) | Max DD={dd:.2%} | Trades={n_tr}")

if __name__ == "__main__":
    main()
