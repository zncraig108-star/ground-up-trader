"""scratch/test_500_macro_hedged.py — $500 Account with Smart Hedged Shorting & Hard Capital Floor.

1. Longs: Top RS Leaders (SOL, AVAX, NEAR, etc.) in Bull Market (BTC > 200d SMA).
2. Shorts: Restricted to BTC & ETH in Bear Market (BTC < 200d SMA) to avoid altcoin short squeezes.
3. Sizing: Starting capital $500.00.
4. Hard Floor: Once equity >= $580, hard floor locks at $500.00.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import math
import numpy as np
import pandas as pd
from config import PortfolioConfig, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_data import load_portfolio_data

def run_hedged_sim():
    cfg = PortfolioConfig()
    bundle = load_portfolio_data(cfg, data_dir="data")
    universe = list(bundle.market_data_1h.keys())
    common_idx = bundle.market_data_1h["BTC-PERP"].index
    specs = DEFAULT_PORTFOLIO_INSTRUMENTS

    c_df = pd.DataFrame({p: bundle.market_data_1h[p]["close"] for p in universe}, index=common_idx).ffill().bfill()
    h_df = pd.DataFrame({p: bundle.market_data_1h[p]["high"] for p in universe}, index=common_idx).ffill().bfill()
    l_df = pd.DataFrame({p: bundle.market_data_1h[p]["low"] for p in universe}, index=common_idx).ffill().bfill()

    c_4h = c_df.resample("4h", closed="left", label="left").last()
    h_4h = h_df.resample("4h", closed="left", label="left").max()
    l_4h = l_df.resample("4h", closed="left", label="left").min()

    up_4h = h_4h.rolling(120, min_periods=60).max().shift(1)
    dn_4h = l_4h.rolling(120, min_periods=60).min().shift(1)
    ex_l_4h = l_4h.rolling(24, min_periods=12).min().shift(1)
    ex_h_4h = h_4h.rolling(24, min_periods=12).max().shift(1)

    ema50_4h = c_4h.ewm(span=300).mean().shift(1)
    slope_4h = ema50_4h.diff(6).shift(1)
    sma20_4h = c_4h.rolling(20, min_periods=10).mean().shift(1)

    atr_4h = {}
    for p in universe:
        tr = np.maximum(h_4h[p] - l_4h[p], np.maximum((h_4h[p] - c_4h[p].shift(1)).abs(), (l_4h[p] - c_4h[p].shift(1)).abs()))
        atr_4h[p] = tr.rolling(14, min_periods=7).mean().shift(1)
    atr_df = pd.DataFrame(atr_4h, index=c_4h.index)

    er_4h = ((c_4h - c_4h.shift(30)).abs() / np.maximum((c_4h - c_4h.shift(1)).abs().rolling(30).sum(), 1e-8)).shift(1).fillna(0.0)
    roc_30d = (c_4h / c_4h.shift(180) - 1.0).shift(1)

    btc_sma = c_4h["BTC-PERP"].rolling(1200, min_periods=120).mean().shift(1)
    is_bull_4h = (c_4h["BTC-PERP"] > btc_sma).shift(1).fillna(False)

    # 1h alignments
    up_1h = up_4h.reindex(common_idx).ffill()
    dn_1h = dn_4h.reindex(common_idx).ffill()
    ex_l_1h = ex_l_4h.reindex(common_idx).ffill()
    ex_h_1h = ex_h_4h.reindex(common_idx).ffill()
    ema50_1h = ema50_4h.reindex(common_idx).ffill()
    slope_1h = slope_4h.reindex(common_idx).ffill()
    sma20_1h = sma20_4h.reindex(common_idx).ffill()
    atr_1h = atr_df.reindex(common_idx).ffill()
    er_1h = er_4h.reindex(common_idx).ffill()
    roc_1h = roc_30d.reindex(common_idx).ffill()
    is_bull_1h = is_bull_4h.reindex(common_idx).ffill()

    for lev_scale in [0.4, 0.6, 0.8, 1.0, 1.2]:
        initial_capital = 500.0
        cash = initial_capital
        equity_curve = []
        trades = []
        min_equity_ever = cash
        floor_locked = False

        pos_qty = {p: 0.0 for p in universe}
        pos_entry = {p: 0.0 for p in universe}
        pos_stop = {p: 0.0 for p in universe}
        pos_trimmed = {p: False for p in universe}

        n_bars = len(common_idx)

        for t in range(1200, n_bars):
            current_time = common_idx[t]
            unrealized = 0.0

            for p in universe:
                q = pos_qty[p]
                if abs(q) > 1e-8:
                    px = c_df[p].iloc[t]
                    hi = h_df[p].iloc[t]
                    lo = l_df[p].iloc[t]
                    stop = pos_stop[p]
                    spec = specs.get(p, DEFAULT_PORTFOLIO_INSTRUMENTS[p])

                    hit_stop = False
                    exit_px = 0.0
                    if q > 0 and lo <= stop and stop > 0:
                        exit_px = min(px, stop)
                        hit_stop = True
                    elif q < 0 and hi >= stop and stop > 0:
                        exit_px = max(px, stop)
                        hit_stop = True

                    if hit_stop:
                        side = 1 if q > 0 else -1
                        fee = abs(q) * exit_px * spec.taker_fee_rate
                        pnl = (exit_px - pos_entry[p]) * q - fee
                        cash += (exit_px - pos_entry[p]) * q - fee
                        trades.append((p, "STOP", pnl, current_time))
                        pos_qty[p] = 0.0
                        pos_entry[p] = 0.0
                        pos_stop[p] = 0.0
                        pos_trimmed[p] = False
                        continue

                    # Climax trim (50% at +2.8 ATR)
                    if not pos_trimmed[p]:
                        s20 = sma20_1h[p].iloc[t]
                        atr = atr_1h[p].iloc[t]
                        if not np.isnan(s20) and not np.isnan(atr) and atr > 0:
                            if q > 0 and (px - s20) >= 2.8 * atr and px > pos_entry[p]:
                                trim_q = spec.floor_to_lot(abs(q) * 0.5)
                                if trim_q * px >= spec.min_notional:
                                    fee = trim_q * px * spec.maker_fee_rate
                                    pnl = (px - pos_entry[p]) * trim_q - fee
                                    cash += (px - pos_entry[p]) * trim_q - fee
                                    trades.append((p, "CLIMAX", pnl, current_time))
                                    pos_qty[p] = q - trim_q
                                    pos_trimmed[p] = True
                                    pos_stop[p] = pos_entry[p]

                    # Trailing stop
                    if q > 0:
                        ex_l = ex_l_1h[p].iloc[t]
                        if not np.isnan(ex_l) and ex_l > pos_stop[p]:
                            pos_stop[p] = ex_l
                    elif q < 0:
                        ex_h = ex_h_1h[p].iloc[t]
                        if not np.isnan(ex_h) and (pos_stop[p] == 0 or ex_h < pos_stop[p]):
                            pos_stop[p] = ex_h

                    # Floor lock protection
                    if floor_locked:
                        hard_stop = pos_entry[p] + (500.0 - cash) / (q + 1e-9)
                        if q > 0:
                            pos_stop[p] = max(pos_stop[p], hard_stop)
                        else:
                            pos_stop[p] = min(pos_stop[p], hard_stop)

                    unrealized += (px - pos_entry[p]) * q

            current_equity = cash + unrealized
            min_equity_ever = min(min_equity_ever, current_equity)
            equity_curve.append(current_equity)

            if current_equity >= 600.0:
                floor_locked = True

            # Floor stopout
            if floor_locked and current_equity <= 500.0:
                for p in universe:
                    q = pos_qty[p]
                    if abs(q) > 1e-8:
                        px = c_df[p].iloc[t]
                        spec = specs.get(p, DEFAULT_PORTFOLIO_INSTRUMENTS[p])
                        fee = abs(q) * px * spec.taker_fee_rate
                        pnl = (px - pos_entry[p]) * q - fee
                        cash += pnl
                        pos_qty[p] = 0.0
                current_equity = cash
                unrealized = 0.0

            # Entries on 4h
            if current_time.hour % 4 == 0:
                if floor_locked and current_equity <= 500.0:
                    continue

                is_bull = is_bull_1h.iloc[t]

                for p in universe:
                    q = pos_qty[p]
                    if abs(q) > 1e-8:
                        px = c_df[p].iloc[t]
                        e50 = ema50_1h[p].iloc[t]
                        spec = specs.get(p, DEFAULT_PORTFOLIO_INSTRUMENTS[p])

                        if q > 0 and (px < e50 or not is_bull):
                            fee = abs(q) * px * spec.maker_fee_rate
                            cash += (px - pos_entry[p]) * q - fee
                            trades.append((p, "LONG_EXIT", (px - pos_entry[p]) * q - fee, current_time))
                            pos_qty[p] = 0.0
                            pos_trimmed[p] = False
                        elif q < 0 and (px > e50 or is_bull):
                            fee = abs(q) * px * spec.maker_fee_rate
                            cash += (px - pos_entry[p]) * q - fee
                            trades.append((p, "SHORT_EXIT", (px - pos_entry[p]) * q - fee, current_time))
                            pos_qty[p] = 0.0
                            pos_trimmed[p] = False

                active_cnt = sum(1 for p in universe if abs(pos_qty[p]) > 1e-8)
                slots = 2 - active_cnt

                if slots > 0 and current_equity >= 400.0:
                    rocs = {p: roc_1h[p].iloc[t] for p in universe if not np.isnan(roc_1h[p].iloc[t])}
                    if len(rocs) >= 4:
                        ranked_desc = sorted(rocs.keys(), key=lambda x: rocs[x], reverse=True)

                        # A. Bull Market: Long Top Momentum Leaders
                        if is_bull:
                            for p in ranked_desc[:3]:
                                if slots <= 0:
                                    break
                                if abs(pos_qty[p]) > 1e-8:
                                    continue

                                px = c_df[p].iloc[t]
                                up = up_1h[p].iloc[t]
                                e50 = ema50_1h[p].iloc[t]
                                slp = slope_1h[p].iloc[t]
                                er_val = er_1h[p].iloc[t]
                                atr = atr_1h[p].iloc[t]
                                spec = specs.get(p, DEFAULT_PORTFOLIO_INSTRUMENTS[p])

                                if px > up and px > e50 and slp > 0 and er_val >= 0.28 and rocs[p] > 0 and atr > 0:
                                    notional = current_equity * (0.35 * lev_scale)
                                    target_qty = spec.floor_to_lot(notional / px)
                                    if target_qty * px >= spec.min_notional:
                                        fee = target_qty * px * spec.maker_fee_rate
                                        cash -= fee
                                        pos_qty[p] = target_qty
                                        pos_entry[p] = px
                                        pos_stop[p] = px - 2.5 * atr
                                        pos_trimmed[p] = False
                                        slots -= 1

                        # B. Bear Market: Short only BTC or ETH
                        else:
                            for p in ["BTC-PERP", "ETH-PERP"]:
                                if slots <= 0:
                                    break
                                if abs(pos_qty[p]) > 1e-8:
                                    continue

                                px = c_df[p].iloc[t]
                                dn = dn_1h[p].iloc[t]
                                e50 = ema50_1h[p].iloc[t]
                                slp = slope_1h[p].iloc[t]
                                atr = atr_1h[p].iloc[t]
                                spec = specs.get(p, DEFAULT_PORTFOLIO_INSTRUMENTS[p])

                                if px < dn and px < e50 and slp < 0 and atr > 0:
                                    notional = current_equity * (0.35 * lev_scale)
                                    target_qty = spec.floor_to_lot(notional / px)
                                    if target_qty * px >= spec.min_notional:
                                        fee = target_qty * px * spec.maker_fee_rate
                                        cash -= fee
                                        pos_qty[p] = -target_qty
                                        pos_entry[p] = px
                                        pos_stop[p] = px + 2.5 * atr
                                        pos_trimmed[p] = False
                                        slots -= 1

        eq_arr = np.array(equity_curve)
        final_eq = eq_arr[-1]
        mult = final_eq / initial_capital
        peaks = np.maximum.accumulate(eq_arr)
        dds = (peaks - eq_arr) / peaks
        max_dd = np.max(dds)
        deposit_dd = max(0.0, (initial_capital - min_equity_ever) / initial_capital)
        wins = [tr for tr in trades if tr[2] > 0]
        wr = len(wins) / max(len(trades), 1)

        print(f"LevScale={lev_scale:.1f} => Final=${final_eq:9,.2f} ({mult:5.2f}x) | Peak MaxDD={max_dd:5.2%} | Deposit DD={deposit_dd:5.2%} (MinEq=${min_equity_ever:.2f}) | Trades={len(trades)} | WR={wr:4.1%}")

if __name__ == "__main__":
    run_hedged_sim()
