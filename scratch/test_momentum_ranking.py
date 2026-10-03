"""scratch/test_momentum_ranking.py — Test relative strength leader ranking in micro futures.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from config import MicroFuturesConfig, PortfolioConfig
from core.portfolio_data import load_portfolio_data
from core.micro_futures_strategy import compute_micro_futures_signals
from core.micro_futures_backtester import MICRO_CONTRACT_SIZES, MARGIN_RATES

p_cfg = PortfolioConfig()
bundle = load_portfolio_data(p_cfg, data_dir="data")

def test_rs_ranking(step=200):
    cfg_1h = MicroFuturesConfig(
        signal_timeframe_hours=1,
        trend_lookback_days=20,
        exit_lookback_days=4,
        atr_stop_mult=2.0,
        er_min_normal=0.28,
        er_min_defensive=0.38,
    )
    sig_1h = compute_micro_futures_signals(bundle.market_data_1h, cfg_1h)
    
    ts = sig_1h.timestamps
    n_bars = len(ts)
    cash = 500.0
    active_floor = 0.0
    min_eq = 500.0
    eq_curve = np.zeros(n_bars)
    
    pos_cnt = {p: 0 for p in sig_1h.universe}
    pos_entry = {p: 0.0 for p in sig_1h.universe}
    pos_stop = {p: 0.0 for p in sig_1h.universe}
    pos_be = {p: False for p in sig_1h.universe}
    
    warmup = min(n_bars // 4, 1200)
    trades = []
    
    for t in range(warmup, n_bars):
        unrealized = 0.0
        used_margin = 0.0
        
        # Stops
        for p in sig_1h.universe:
            cnt = pos_cnt[p]
            if cnt != 0:
                asig = sig_1h.asset_signals[p]
                px = asig.close[t]
                hi = asig.high[t]
                lo = asig.low[t]
                stop = pos_stop[p]
                mult = MICRO_CONTRACT_SIZES[p]
                qty = abs(cnt) * mult
                side = 1 if cnt > 0 else -1
                
                hit_stop = False
                exit_px = 0.0
                if cnt > 0 and lo <= stop and stop > 0:
                    exit_px = min(px, stop)
                    hit_stop = True
                elif cnt < 0 and hi >= stop and stop > 0:
                    exit_px = max(px, stop)
                    hit_stop = True
                    
                if hit_stop:
                    fee = qty * exit_px * 0.0005
                    pnl = (exit_px - pos_entry[p]) * qty * side - fee
                    cash += pnl
                    trades.append((p, side, pnl))
                    pos_cnt[p] = 0
                    pos_be[p] = False
                    continue
                    
                entry = pos_entry[p]
                atr = asig.atr[t]
                if not pos_be[p] and atr > 0:
                    if cnt > 0 and px >= entry + 2.0 * atr:
                        pos_stop[p] = max(pos_stop[p], entry + 0.05 * atr)
                        pos_be[p] = True
                    elif cnt < 0 and px <= entry - 2.0 * atr:
                        pos_stop[p] = min(pos_stop[p], entry - 0.05 * atr) if pos_stop[p] > 0 else entry - 0.05 * atr
                        pos_be[p] = True
                        
                if cnt > 0:
                    ex_l = asig.trailing_exit_low[t]
                    if not np.isnan(ex_l) and ex_l > pos_stop[p]:
                        pos_stop[p] = ex_l
                elif cnt < 0:
                    ex_h = asig.trailing_exit_high[t]
                    if not np.isnan(ex_h) and (pos_stop[p] == 0 or ex_h < pos_stop[p]):
                        pos_stop[p] = ex_h
                        
                unrealized += (px - pos_entry[p]) * qty * side
                used_margin += qty * px * MARGIN_RATES[p]
                
        # USDC 5%
        idle = max(0.0, cash - used_margin)
        cash += idle * (0.05 / 8760.0)
        
        eq = cash + unrealized
        min_eq = min(min_eq, eq)
        
        if eq >= 650.0 and active_floor < 500.0:
            active_floor = 500.0
        elif eq >= 1500.0 and active_floor < 1000.0:
            active_floor = 1000.0
        elif eq >= 3000.0 and active_floor < 2000.0:
            active_floor = 2000.0
            
        if active_floor > 0 and eq <= active_floor:
            for p in sig_1h.universe:
                cnt = pos_cnt[p]
                if cnt != 0:
                    px = sig_1h.asset_signals[p].close[t]
                    mult = MICRO_CONTRACT_SIZES[p]
                    qty = abs(cnt) * mult
                    side = 1 if cnt > 0 else -1
                    fee = qty * px * 0.0005
                    cash += (px - pos_entry[p]) * qty * side - fee
                    trades.append((p, side, (px - pos_entry[p]) * qty * side - fee))
                    pos_cnt[p] = 0
            eq = cash
            unrealized = 0.0
            
        # Hourly Trend Exit & Entry
        for p in sig_1h.universe:
            cnt = pos_cnt[p]
            if cnt != 0:
                asig = sig_1h.asset_signals[p]
                px = asig.close[t]
                e50 = asig.ema_50d[t]
                mult = MICRO_CONTRACT_SIZES[p]
                qty = abs(cnt) * mult
                side = 1 if cnt > 0 else -1
                if (cnt > 0 and px < e50) or (cnt < 0 and px > e50):
                    fee = qty * px * 0.0002
                    pnl = (px - pos_entry[p]) * qty * side - fee
                    cash += pnl
                    trades.append((p, side, pnl))
                    pos_cnt[p] = 0
                    pos_be[p] = False
                    
        # Entries
        if (active_floor == 0 or eq > active_floor) and (eq >= 450.0):
            is_defensive = (eq < 500.0)
            active_cnt = sum(1 for p in sig_1h.universe if pos_cnt[p] != 0)
            max_slots = 1 if is_defensive else 2
            slots = max_slots - active_cnt
            
            if slots > 0:
                excess = max(0.0, eq - 500.0)
                add_cnt = int(excess // step)
                base_cnt = 1 if is_defensive else (1 + add_cnt)
                
                tradable = ["SOL-PERP", "ETH-PERP"]
                if eq >= 1000.0:
                    tradable.append("BTC-PERP")
                    
                # Rank candidates by 20-day momentum (close / close_20d_ago - 1)
                scored = []
                for p in tradable:
                    if pos_cnt[p] != 0:
                        continue
                    asig = sig_1h.asset_signals[p]
                    px = asig.close[t]
                    px_prev = asig.close[max(0, t - 480)]
                    mom = (px - px_prev) / px_prev if px_prev > 0 else 0.0
                    scored.append((abs(mom), p))
                scored.sort(reverse=True)
                
                for _, p in scored:
                    if slots <= 0:
                        break
                    asig = sig_1h.asset_signals[p]
                    px = asig.close[t]
                    up = asig.upper_breakout[t]
                    dn = asig.lower_breakout[t]
                    e50 = asig.ema_50d[t]
                    slp = asig.ema_slope[t]
                    er_val = asig.er[t]
                    atr_val = asig.atr[t]
                    mult = MICRO_CONTRACT_SIZES[p]
                    
                    if np.isnan(px) or np.isnan(atr_val) or atr_val <= 0:
                        continue
                    req_er = 0.38 if is_defensive else 0.28
                    
                    stop_dist = max(2.0 * atr_val, px * 0.035)
                    cnt_target = max(1, base_cnt // 2) if p == "BTC-PERP" else base_cnt
                    if is_defensive:
                        cnt_target = 1
                        
                    notional = cnt_target * mult * px
                    margin = notional * MARGIN_RATES[p]
                    
                    if margin <= eq * 0.45 and (notional / eq) <= 2.8:
                        if er_val >= req_er and px > up and px > e50 and slp > 0:
                            fee = notional * 0.0002
                            cash -= fee
                            pos_cnt[p] = cnt_target
                            pos_entry[p] = px
                            pos_stop[p] = px - stop_dist
                            slots -= 1
                        elif px < dn and px < e50 and slp < 0 and er_val >= (req_er + 0.05):
                            fee = notional * 0.0002
                            cash -= fee
                            pos_cnt[p] = -cnt_target
                            pos_entry[p] = px
                            pos_stop[p] = px + stop_dist
                            slots -= 1
                            
        eq_curve[t] = eq
        
    final_eq = eq_curve[-1]
    peaks = np.maximum.accumulate(eq_curve[warmup:])
    dds = (peaks - eq_curve[warmup:]) / peaks
    max_dd = np.max(dds)
    dep_dd = max(0.0, (500.0 - min_eq) / 500.0)
    print(f"Momentum Ranked (Step={step}) => Final=${final_eq:8.2f} ({final_eq/500:5.2f}x) | MaxDD={max_dd:5.1%} | DepDD={dep_dd:5.1%} (MinEq=${min_eq:.2f}) | Trades={len(trades)}")

for s in [100, 150, 200, 250]:
    test_rs_ranking(step=s)
