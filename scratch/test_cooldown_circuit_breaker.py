"""scratch/test_cooldown_circuit_breaker.py — Dynamic Drawdown Circuit Breaker with Cooldown & Reset."""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, InstrumentSpec, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data
from core.portfolio_strategy import compute_portfolio_signals

def simulate_with_cooldown(signals, cfg, max_dd_limit=0.12, cooldown_days=14):
    ts = signals.timestamps
    universe = signals.universe
    n_bars = len(ts)
    inst_specs = DEFAULT_PORTFOLIO_INSTRUMENTS

    initial_equity = float(cfg.initial_equity)
    cash = initial_equity
    equity_curve = np.zeros(n_bars, dtype=np.float64)

    pos_qty = {p: 0.0 for p in universe}
    pos_entry_px = {p: 0.0 for p in universe}
    pos_entry_bar = {p: -1 for p in universe}
    pos_entry_fees = {p: 0.0 for p in universe}
    pos_funding_accrued = {p: 0.0 for p in universe}
    pos_tranche = {p: 0 for p in universe}

    peak_equity = initial_equity
    in_cooldown_until = -1
    cooldown_bars = cooldown_days * 24

    trades = []

    for t in range(n_bars):
        current_time = ts[t]

        # 1. Hourly Funding Settlement
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                px = signals.asset_signals[p].close[t]
                rate = signals.asset_signals[p].funding_rate[t]
                if not np.isnan(px) and px > 0:
                    cf = -q * px * rate
                    cash += cf
                    pos_funding_accrued[p] += cf

        # 2. Check Stop-Losses on 1h Bar Extremes
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                asig = signals.asset_signals[p]
                sl_px = asig.stop_loss_px[t]
                spec = inst_specs.get(p, InstrumentSpec(p))
                hit_stop = False
                exit_price = 0.0

                if q > 0 and sl_px > 0 and asig.low[t] <= sl_px:
                    exit_price = min(asig.close[t], sl_px) - spec.tick_size
                    hit_stop = True
                elif q < 0 and sl_px > 0 and asig.high[t] >= sl_px:
                    exit_price = max(asig.close[t], sl_px) + spec.tick_size
                    hit_stop = True

                if hit_stop and exit_price > 0:
                    notional = abs(q) * exit_price
                    exit_fee = notional * spec.taker_fee_rate
                    pnl = (exit_price - pos_entry_px[p]) * q - (pos_entry_fees[p] + exit_fee)
                    cash += (exit_price - pos_entry_px[p]) * q - exit_fee
                    trades.append((p, 1 if q > 0 else -1, exit_price, pnl, "stop_loss"))
                    pos_qty[p] = 0.0
                    pos_entry_px[p] = 0.0
                    pos_tranche[p] = 0

        # 3. Mark to Market Equity
        unrealized = 0.0
        for p in universe:
            q = pos_qty[p]
            if abs(q) > 1e-9:
                px = signals.asset_signals[p].close[t]
                if not np.isnan(px) and px > 0:
                    unrealized += (px - pos_entry_px[p]) * q

        current_equity = cash + unrealized

        # 4. Check Drawdown from Peak
        if current_equity > peak_equity:
            peak_equity = current_equity

        current_dd = (peak_equity - current_equity) / max(peak_equity, 1e-6)

        # 5. Circuit Breaker Trigger: If DD hits limit, LIQUIDATE TO CASH and enter Cooldown
        if current_dd >= max_dd_limit and in_cooldown_until < t:
            # Emergency exit to 100% cash
            for p in universe:
                q = pos_qty[p]
                if abs(q) > 1e-9:
                    px = signals.asset_signals[p].close[t]
                    spec = inst_specs.get(p, InstrumentSpec(p))
                    fee = abs(q) * px * spec.taker_fee_rate
                    pnl = (px - pos_entry_px[p]) * q - (pos_entry_fees[p] + fee)
                    cash += (px - pos_entry_px[p]) * q - fee
                    trades.append((p, 1 if q > 0 else -1, px, pnl, "circuit_breaker"))
                    pos_qty[p] = 0.0
                    pos_tranche[p] = 0

            unrealized = 0.0
            current_equity = cash
            in_cooldown_until = t + cooldown_bars
            # Reset peak equity anchor to current equity so we can resume trading after cooldown
            peak_equity = current_equity

        # Cooldown active check
        is_cooling_down = (t < in_cooldown_until)

        # 6. Discrete 4h Rebalancing
        is_rebal_hour = (current_time.hour % cfg.signal_timeframe_hours == 0) and not is_cooling_down

        if is_rebal_hour and current_equity > 100.0:
            for p in universe:
                asig = signals.asset_signals[p]
                spec = inst_specs.get(p, InstrumentSpec(p))
                target_tr = int(asig.tranche[t])
                prev_tr = int(asig.tranche[t - 1]) if t > 0 else 0
                current_tr = pos_tranche[p]
                px = asig.close[t]

                if np.isnan(px) or px <= 0:
                    continue

                # Close
                if target_tr == 0 and current_tr != 0 and abs(pos_qty[p]) > 1e-9:
                    q = abs(pos_qty[p])
                    fee = q * px * spec.maker_fee_rate
                    pnl = (px - pos_entry_px[p]) * pos_qty[p] - (pos_entry_fees[p] + fee)
                    cash += (px - pos_entry_px[p]) * pos_qty[p] - fee
                    trades.append((p, 1 if pos_qty[p] > 0 else -1, px, pnl, "signal"))
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
                        pos_entry_bar[p] = t
                        pos_entry_fees[p] = fee
                        pos_tranche[p] = target_tr

                # Pyramid
                elif target_tr > current_tr and current_tr > 0 and cfg.pyramid_enabled:
                    add_w = cfg.tranche_2_leverage if target_tr == 2 else cfg.tranche_3_leverage
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

    # Calculate overall max drawdown from the true global running peak
    # (NOT the reset peak, but the absolute true global peak)
    true_peaks = np.maximum.accumulate(equity_curve)
    true_dds = (true_peaks - equity_curve) / np.maximum(true_peaks, 1e-6)
    true_max_dd = float(np.max(true_dds))
    final_eq = float(equity_curve[-1])

    eq_s = pd.Series(equity_curve, index=ts)
    rets = eq_s.pct_change().fillna(0.0).to_numpy()
    std = np.std(rets)
    sharpe = float(np.mean(rets) / std * math.sqrt(8760.0)) if std > 1e-9 else 0.0

    return final_eq, true_max_dd, sharpe, len(trades)

def main():
    bundle = load_portfolio_data(PortfolioConfig(), data_dir="data")

    # Test circuit breaker settings
    print("Testing Drawdown Circuit Breaker with Cooldown:")
    for dd_lim in [0.08, 0.10, 0.12, 0.14]:
        for c_days in [3, 7, 14, 21]:
            for lev in [0.70, 0.85, 1.00]:
                cfg = PortfolioConfig(
                    tranche_1_leverage=lev,
                    tranche_2_leverage=0.50 * lev,
                    tranche_3_leverage=0.35 * lev,
                    allow_bear_shorts=False, # 100% Cash in Bear
                    exit_lookback_bars=48, # 8 days
                )
                signals = compute_portfolio_signals(bundle.market_data_1h, bundle.funding_data_1h, cfg)
                eq, true_dd, sh, n = simulate_with_cooldown(signals, cfg, max_dd_limit=dd_lim, cooldown_days=c_days)
                mult = eq / 10000.0
                print(f"DD Limit={dd_lim:.0%} | Cooldown={c_days:2d}d | Lev={lev:.2f} => Final=${eq:,.2f} ({mult:6.2f}x) | True Max DD={true_dd:6.2%} | Sharpe={sh:4.2f} | Tr={n}")

if __name__ == "__main__":
    main()
