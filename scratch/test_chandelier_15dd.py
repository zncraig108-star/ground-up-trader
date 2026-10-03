"""scratch/test_chandelier_15dd.py — Chandelier Ratchet Trailing Stops + Drawdown Governor."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals

def run_chandelier_test(signals, base_cfg, chandelier_atr=2.2, max_lev=1.2, dd_brake=0.13):
    ts = signals.timestamps
    universe = signals.universe
    n_bars = len(ts)
    inst_specs = DEFAULT_PORTFOLIO_INSTRUMENTS

    initial_equity = float(base_cfg.initial_equity)
    cash = initial_equity
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

        # 1. Hourly Stop Loss Evaluation (Intrabar precision)
        unrealized = 0.0
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                asig = signals.asset_signals[p]
                px = asig.close[t]
                hi = asig.high[t]
                lo = asig.low[t]
                atr = asig.atr[t]

                if q > 0:
                    # Update peak price
                    pos_peak_px[p] = max(pos_peak_px[p], hi)
                    # Chandelier ratchet: stop can ONLY move up
                    chandelier_stop = pos_peak_px[p] - chandelier_atr * atr
                    pos_stop_px[p] = max(pos_stop_px[p], chandelier_stop)

                    # Check hit
                    if lo <= pos_stop_px[p]:
                        exit_px = min(px, pos_stop_px[p])
                        fee = abs(q) * exit_px * 0.0005
                        pnl = (exit_px - pos_entry_px[p]) * q - fee - pos_entry_fees[p]
                        cash += (exit_px - pos_entry_px[p]) * q - fee
                        trades.append((p, 1, pos_entry_px[p], exit_px, pnl, "chandelier_stop"))
                        pos_qty[p] = 0.0
                        continue

                elif q < 0:
                    pos_peak_px[p] = min(pos_peak_px[p], lo)
                    chandelier_stop = pos_peak_px[p] + chandelier_atr * atr
                    pos_stop_px[p] = min(pos_stop_px[p], chandelier_stop)

                    if hi >= pos_stop_px[p]:
                        exit_px = max(px, pos_stop_px[p])
                        fee = abs(q) * exit_px * 0.0005
                        pnl = (exit_px - pos_entry_px[p]) * q - fee - pos_entry_fees[p]
                        cash += (exit_px - pos_entry_px[p]) * q - fee
                        trades.append((p, -1, pos_entry_px[p], exit_px, pnl, "chandelier_stop"))
                        pos_qty[p] = 0.0
                        continue

                unrealized += (px - pos_entry_px[p]) * q

        current_equity = cash + unrealized
        peak_equity = max(peak_equity, current_equity)
        current_dd = (peak_equity - current_equity) / max(peak_equity, 1e-6)

        # 2. Discrete 4h Rebalance
        if current_time.hour % 4 == 0 and current_equity > 50.0:
            # Drawdown governor multiplier:
            # If DD < 5%, mult = 1.0. If DD between 5% and dd_brake, scale down.
            if current_dd <= 0.05:
                risk_scale = 1.0
            elif current_dd < dd_brake:
                risk_scale = (dd_brake - current_dd) / (dd_brake - 0.05)
            else:
                risk_scale = 0.0

            # If DD hits dd_brake (e.g. 13%), emergency close all open positions to guarantee < 15% Max DD!
            if current_dd >= dd_brake:
                for p in universe:
                    if abs(pos_qty[p]) > 1e-9:
                        q = pos_qty[p]
                        px = signals.asset_signals[p].close[t]
                        fee = abs(q) * px * 0.0005
                        pnl = (px - pos_entry_px[p]) * q - fee - pos_entry_fees[p]
                        cash += (px - pos_entry_px[p]) * q - fee
                        trades.append((p, 1 if q > 0 else -1, pos_entry_px[p], px, pnl, "dd_circuit_breaker"))
                        pos_qty[p] = 0.0
                unrealized = 0.0
                current_equity = cash

            for p in universe:
                asig = signals.asset_signals[p]
                spec = inst_specs.get(p, InstrumentSpec(p))
                target_tr = int(asig.tranche[t])
                prev_tr = int(asig.tranche[t - 1]) if t > 0 else 0
                px = asig.close[t]
                atr = asig.atr[t]

                if np.isnan(px) or px <= 0:
                    continue

                # Signal exit (e.g. regime flipped)
                if target_tr == 0 and abs(pos_qty[p]) > 1e-9:
                    q = pos_qty[p]
                    fee = abs(q) * px * 0.0002
                    pnl = (px - pos_entry_px[p]) * q - fee - pos_entry_fees[p]
                    cash += (px - pos_entry_px[p]) * q - fee
                    trades.append((p, 1 if q > 0 else -1, pos_entry_px[p], px, pnl, "signal_exit"))
                    pos_qty[p] = 0.0

                # Entry
                elif pos_qty[p] == 0.0 and target_tr != 0 and prev_tr == 0 and risk_scale > 0.05:
                    # Allocate portion of equity per leader
                    weight = (max_lev / 2.0) * risk_scale  # 2 leaders
                    notional = current_equity * weight
                    q = spec.floor_to_lot(notional / px)

                    if q * px >= spec.min_notional:
                        fee = q * px * 0.0002
                        cash -= fee
                        pos_qty[p] = q if target_tr > 0 else -q
                        pos_entry_px[p] = px
                        pos_peak_px[p] = px
                        pos_stop_px[p] = px - chandelier_atr * atr if target_tr > 0 else px + chandelier_atr * atr
                        pos_entry_fees[p] = fee

        equity_curve[t] = max(0.0, current_equity)

    peaks = np.maximum.accumulate(equity_curve)
    dds = (peaks - equity_curve) / np.maximum(peaks, 1e-6)
    max_dd = float(np.max(dds))
    final_eq = float(equity_curve[-1])

    # Sharpe
    eq_s = pd.Series(equity_curve, index=ts)
    rets = eq_s.pct_change().fillna(0.0).to_numpy()
    std = np.std(rets)
    sharpe = float(np.mean(rets) / std * math.sqrt(8760.0)) if std > 1e-9 else 0.0

    return final_eq, max_dd, sharpe, len(trades)

def main():
    base_cfg = PortfolioConfig()
    bundle = load_portfolio_data(base_cfg, data_dir="data")
    signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, base_cfg)

    print("Evaluating Chandelier Ratchet Stop + DD Circuit Breaker:")
    for chan in [1.5, 2.0, 2.5, 3.0]:
        for lev in [0.8, 1.0, 1.2, 1.5, 2.0]:
            for brake in [0.12, 0.13, 0.14]:
                eq, dd, sh, n = run_chandelier_test(signals, base_cfg, chandelier_atr=chan, max_lev=lev, dd_brake=brake)
                mult = eq / 10000.0
                if dd <= 0.15:
                    print(f"Chan={chan:.1f}ATR | Lev={lev:.1f}x | Brake={brake:.0%} => Final=${eq:,.2f} ({mult:.2f}x) | Max DD={dd:.2%} | Sharpe={sh:.2f} | Tr={n}")

if __name__ == "__main__":
    main()
