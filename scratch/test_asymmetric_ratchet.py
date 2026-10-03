"""scratch/test_asymmetric_ratchet.py — Asymmetric profit locking on mature trends."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals

def test_ratchet_mechanic(signals, base_cfg, lock_threshold_atr=4.0, lock_trail_atr=2.5):
    """
    Keep baseline Donchian exit for normal trades, BUT if a trade exceeds lock_threshold_atr
    (e.g. +4.0 ATR profit), switch to a tighter trailing stop (Peak - lock_trail_atr * ATR)
    to prevent huge winners from giving back all their profits!
    """
    ts = signals.timestamps
    universe = signals.universe
    n_bars = len(ts)
    inst_specs = DEFAULT_PORTFOLIO_INSTRUMENTS

    cash = float(base_cfg.initial_equity)
    initial_equity = cash
    equity_curve = np.zeros(n_bars, dtype=np.float64)

    pos_qty = {p: 0.0 for p in universe}
    pos_entry_px = {p: 0.0 for p in universe}
    pos_peak_px = {p: 0.0 for p in universe}
    pos_entry_fees = {p: 0.0 for p in universe}
    pos_tranche = {p: 0 for p in universe}

    trades = []

    for t in range(n_bars):
        current_time = ts[t]

        # 1. Hourly Stop Loss Evaluation
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
                    pos_peak_px[p] = max(pos_peak_px[p], hi)
                    peak = pos_peak_px[p]
                    entry = pos_entry_px[p]
                    
                    # Normal Donchian stop
                    stop_px = asig.stop_loss_px[t]
                    
                    # If profit has exceeded lock_threshold_atr, ratchet stop up to Peak - lock_trail_atr * ATR
                    if peak >= entry + lock_threshold_atr * atr:
                        ratchet_stop = peak - lock_trail_atr * atr
                        stop_px = max(stop_px, ratchet_stop)

                    if lo <= stop_px and stop_px > 0:
                        exit_px = min(px, stop_px)
                        fee = abs(q) * exit_px * 0.0005
                        pnl = (exit_px - entry) * q - fee - pos_entry_fees[p]
                        cash += (exit_px - entry) * q - fee
                        trades.append((p, 1, entry, exit_px, pnl))
                        pos_qty[p] = 0.0
                        pos_tranche[p] = 0
                        continue

                elif q < 0:
                    pos_peak_px[p] = min(pos_peak_px[p], lo)
                    trough = pos_peak_px[p]
                    entry = pos_entry_px[p]
                    stop_px = asig.stop_loss_px[t]

                    if trough <= entry - lock_threshold_atr * atr:
                        ratchet_stop = trough + lock_trail_atr * atr
                        stop_px = min(stop_px, ratchet_stop)

                    if hi >= stop_px and stop_px > 0:
                        exit_px = max(px, stop_px)
                        fee = abs(q) * exit_px * 0.0005
                        pnl = (exit_px - entry) * q - fee - pos_entry_fees[p]
                        cash += (exit_px - entry) * q - fee
                        trades.append((p, -1, entry, exit_px, pnl))
                        pos_qty[p] = 0.0
                        pos_tranche[p] = 0
                        continue

                unrealized += (px - pos_entry_px[p]) * q

        current_equity = cash + unrealized

        # 2. Discrete 4h Rebalance
        if current_time.hour % 4 == 0 and current_equity > 100.0:
            for p in universe:
                asig = signals.asset_signals[p]
                spec = inst_specs.get(p, InstrumentSpec(p))
                target_tr = int(asig.tranche[t])
                prev_tr = int(asig.tranche[t - 1]) if t > 0 else 0
                current_tr = pos_tranche[p]
                px = asig.close[t]

                if np.isnan(px) or px <= 0:
                    continue

                # Signal Exit
                if target_tr == 0 and current_tr != 0 and abs(pos_qty[p]) > 1e-9:
                    q = abs(pos_qty[p])
                    side = 1 if pos_qty[p] > 0 else -1
                    fee = q * px * spec.maker_fee_rate
                    pnl = (px - pos_entry_px[p]) * pos_qty[p] - (pos_entry_fees[p] + fee)
                    cash += (px - pos_entry_px[p]) * pos_qty[p] - fee
                    trades.append((p, side, pos_entry_px[p], px, pnl))
                    pos_qty[p] = 0.0
                    pos_tranche[p] = 0

                # Open
                elif current_tr == 0 and target_tr != 0 and prev_tr == 0:
                    target_w = asig.target_weight[t]
                    dollar_notional = current_equity * abs(target_w)
                    q = spec.floor_to_lot(dollar_notional / px)
                    if q * px >= spec.min_notional:
                        fee = q * px * spec.maker_fee_rate
                        cash -= fee
                        pos_qty[p] = q if target_tr > 0 else -q
                        pos_entry_px[p] = px
                        pos_peak_px[p] = px
                        pos_entry_fees[p] = fee
                        pos_tranche[p] = target_tr

                # Pyramid
                elif target_tr > current_tr and current_tr > 0:
                    add_w = base_cfg.tranche_2_leverage if target_tr == 2 else base_cfg.tranche_3_leverage
                    dollar_notional = current_equity * add_w
                    add_q = spec.floor_to_lot(dollar_notional / px)
                    if add_q * px >= spec.min_notional:
                        fee = add_q * px * spec.maker_fee_rate
                        cash -= fee
                        old_val = abs(pos_qty[p]) * pos_entry_px[p]
                        new_val = add_q * px
                        tot_q = abs(pos_qty[p]) + add_q
                        pos_entry_px[p] = (old_val + new_val) / tot_q
                        pos_qty[p] = tot_q
                        pos_entry_fees[p] += fee
                        pos_tranche[p] = target_tr

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

    print("Testing Asymmetric Profit Locking Ratchet on mature trends:")
    for thresh in [3.0, 4.0, 5.0, 6.0]:
        for trail in [1.5, 2.0, 2.5, 3.0]:
            eq, dd, n = test_ratchet_mechanic(signals, cfg, lock_threshold_atr=thresh, lock_trail_atr=trail)
            mult = eq / 10000.0
            print(f"Thresh={thresh:.1f}ATR | Trail={trail:.1f}ATR => Final=${eq:,.2f} ({mult:6.2f}x) | Max DD={dd:6.2%} | Trades={n}")

if __name__ == "__main__":
    main()
