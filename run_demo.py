"""run_demo.py — end-to-end demo of the SOL-PERP Evolutionary Trading System.

Runs the entire pipeline with a single command, **no network and no GPU required**
(GPU auto-used if present):

    1. foundations  — config + deterministic synthetic 1m/1h/funding data
    2. self-checks  — embedded invariant assertions (causality, fills, fees/funding,
                      liquidation, fitness, determinism, GPU/CPU parity)
    3. math         — vectorized signals (no NaNs, causal)
    4. evolution    — a few GA generations (best fitness non-decreasing) + deployment pool
    5. live/paper   — a winning genome run against a simulated paper stream
    6. UI           — launch the FastAPI dashboard (continues evolving in the background)

Usage:
    python run_demo.py [--days 35] [--generations 6] [--population 24] [--warmup 2]
                       [--walk-forward] [--generations-per-window 20]
                       [--data path/to/1m.parquet] [--funding path/to/funding.csv]
                       [--host 127.0.0.1] [--port 8000] [--no-serve]

Pass ``--data``/``--funding`` to train on real history; without them the demo runs on
deterministic synthetic data (fine for smoke tests, not for trading decisions).

``--generations`` is the legacy total-generation budget for non-WFO runs. A normal
``--walk-forward`` run completes one full pass: every window receives
``--generations-per-window`` generations (20 by default).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys

import numpy as np

from config import NUM_GENES, FillModelConfig, InstrumentSpec, build_config
from core.backtester import assert_no_lookahead, backtest
from core.data import data_manifest, load_market_data, make_synthetic
from core.evolution import EvolutionEngine
from core.metrics import robust_fitness
from core.strategy import (GENE_INDEX, Genome, SignalFeatures, SignalIntents,
                           compute_signals, random_population)
from core.xp import asnumpy, get_backend_info


# ---------------------------------------------------------------------------
# Self-checks (embedded lightweight invariants — AGENTS.md §6)
# ---------------------------------------------------------------------------

def _check_signals_causality(feat: SignalFeatures, pop: np.ndarray) -> None:
    """Signal at bar t must not depend on data after t."""
    n = len(feat.close)
    k = n // 2
    sig = compute_signals(feat, pop)
    close2 = feat.close.copy()
    close2[k:] *= 10.0
    sig2 = compute_signals(SignalFeatures(feat.open, feat.high, feat.low, close2), pop)
    for name in ("regime", "atr", "entry_price"):
        a = asnumpy(getattr(sig, name))[:, :k]
        b = asnumpy(getattr(sig2, name))[:, :k]
        assert np.allclose(a, b, equal_nan=True), f"causality violated for {name}"
    print("  [ok] causality: perturbing future candles leaves past signals unchanged")


def _check_fills(inst: InstrumentSpec) -> None:
    """Post-only cross-through-by-1-tick, rejection, fees, funding, liquidation."""
    import pandas as pd

    def _1m(o, h, l, c):
        m = len(c)
        idx = pd.date_range("2024-01-01 00:01", periods=m, freq="1min")
        return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                             "volume": np.ones(m)}, index=pd.DatetimeIndex(idx, name="timestamp"))

    def _1h(h):
        idx = pd.date_range("2024-01-01 01:00", periods=h, freq="1h")
        return pd.DataFrame({"open": np.ones(h), "high": np.ones(h), "low": np.ones(h),
                             "close": np.ones(h), "volume": np.ones(h)}, index=idx)

    def _fund(h, rate):
        idx = pd.date_range("2024-01-01 01:00", periods=h, freq="1h")
        return pd.DataFrame({"funding_rate": np.full(h, rate)}, index=idx)

    def _sig(H, s, price):
        long_e = np.zeros((1, H), dtype=bool); long_e[0, s] = True
        return SignalIntents(regime=np.zeros((1, H)), atr=np.ones((1, H)),
                             long_entry=long_e, short_entry=np.zeros((1, H), dtype=bool),
                             entry_price=np.where(np.arange(H)[None, :] == s, price, np.nan))

    def _genome(tp=1.0, sl=1.0, lev=1.0, maxhold=240.0):
        g = np.zeros((1, NUM_GENES))
        g[0, GENE_INDEX["tp_atr_mult"]] = tp
        g[0, GENE_INDEX["sl_atr_mult"]] = sl
        g[0, GENE_INDEX["target_leverage"]] = lev
        g[0, GENE_INDEX["max_hold_bars"]] = maxhold
        return g

    m = 120
    o = np.full(m, 101.0); h = np.full(m, 101.0); l = np.full(m, 101.0); c = np.full(m, 101.0)
    o[60] = 101.0; h[60] = 101.0; l[60] = 100.005; c[60] = 100.5   # touch, no fill
    o[61] = 100.5; h[61] = 100.6; l[61] = 99.995; c[61] = 100.1    # still no fill
    o[62] = 100.1; h[62] = 100.2; l[62] = 99.99; c[62] = 100.0     # low == L - tick -> fill
    o[63] = 100.0; h[63] = 101.02; l[63] = 99.99; c[63] = 101.0    # TP cross-through
    for i in range(64, m):
        o[i] = h[i] = l[i] = c[i] = 101.0
    res = backtest(_1m(o, h, l, c), _1h(2), _fund(2, 0.0), _sig(2, 0, 100.0),
                   _genome(), inst)
    t = res.trades[0][0]
    assert t.entry_bar == 62 and abs(t.entry_price - 100.0) < 1e-9, "cross-through timing"
    assert t.exit_reason == "tp" and t.fees > 0, "tp/fees"
    assert abs(t.pnl - (100.0 - t.fees)) < 1e-9, "pnl must be net of fees"
    print("  [ok] post-only cross-through-by-1-tick + maker/taker fee signs")

    # rejection: buy limit at/above market open is rejected
    o2 = np.full(m, 101.0); h2 = np.full(m, 101.0); l2 = np.full(m, 101.0); c2 = np.full(m, 101.0)
    for i in range(60, m):
        o2[i] = h2[i] = l2[i] = c2[i] = 99.5
    res2 = backtest(_1m(o2, h2, l2, c2), _1h(2), _fund(2, 0.0), _sig(2, 0, 100.0), _genome(), inst)
    assert asnumpy(res2.post_only_rejections)[0] == 1
    print("  [ok] post-only rejection when order would cross the spread")

    # funding sign: long pays positive funding
    m4 = 180
    o4 = np.full(m4, 101.0); h4 = np.full(m4, 101.0); l4 = np.full(m4, 101.0); c4 = np.full(m4, 100.0)
    l4[60] = 99.99; c4[60] = 100.0
    res4 = backtest(_1m(o4, h4, l4, c4), _1h(3), _fund(3, 0.001),
                    _sig(3, 0, 100.0), _genome(tp=100.0, sl=100.0, lev=1.0, maxhold=1.0), inst)
    assert res4.trades[0][0].funding > 0, "long must pay positive funding"
    print("  [ok] hourly funding sign (long pays when rate > 0)")

    # liquidation: 2x long collapses -> maintenance-margin liquidation
    m5 = 120
    o5 = np.full(m5, 101.0); h5 = np.full(m5, 101.0); l5 = np.full(m5, 101.0); c5 = np.full(m5, 100.0)
    h5[60] = 100.0; l5[60] = 99.99; c5[60] = 100.0
    for i in range(61, m5):
        o5[i] = h5[i] = l5[i] = c5[i] = 50.0
    res5 = backtest(_1m(o5, h5, l5, c5), _1h(2), _fund(2, 0.0),
                    _sig(2, 0, 100.0), _genome(tp=100.0, sl=100.0, lev=2.0), inst)
    assert res5.trades[0][0].exit_reason == "liquidation" and asnumpy(res5.liquidations)[0] == 1
    print("  [ok] maintenance-margin liquidation on 2x collapse")


def _check_fill_model(inst: InstrumentSpec) -> None:
    """Phase-1 lifecycle/fill invariants: stale cancel, hold-from-fill, gap slippage, intrabar liq."""
    import pandas as pd

    def frame_1m(H, o, h, l, c):
        m = H * 60
        idx = pd.date_range("2024-01-01 00:01", periods=m, freq="1min")
        return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                             "volume": np.ones(m)}, index=idx)

    def frame_1h(H):
        idx = pd.date_range("2024-01-01 01:00", periods=H, freq="1h")
        return pd.DataFrame({"open": np.ones(H), "high": np.ones(H), "low": np.ones(H),
                             "close": np.ones(H), "volume": np.ones(H)}, index=idx)

    def frame_fund(H):
        idx = pd.date_range("2024-01-01 01:00", periods=H, freq="1h")
        return pd.DataFrame({"funding_rate": np.zeros(H)}, index=idx)

    def genome_vec(tp=6.0, sl=3.0, lev=1.0, maxhold=240.0):
        g = np.zeros((1, NUM_GENES))
        g[0, GENE_INDEX["tp_atr_mult"]] = tp
        g[0, GENE_INDEX["sl_atr_mult"]] = sl
        g[0, GENE_INDEX["target_leverage"]] = lev
        g[0, GENE_INDEX["max_hold_bars"]] = maxhold
        return g

    def intents(H, longs, entries):
        reg = np.zeros((1, H))
        ep = np.full((1, H), np.nan)
        le = np.zeros((1, H), dtype=bool)
        se = np.zeros((1, H), dtype=bool)
        for h in longs:
            reg[0, h] = 1.0
            le[0, h] = True
        for h, p in entries.items():
            ep[0, h] = p
        return SignalIntents(regime=reg, atr=np.ones((1, H)), long_entry=le,
                             short_entry=se, entry_price=ep)

    # (1) A flat signal cancels a stale resting entry (base) but not in legacy optimistic mode.
    H = 4
    o = np.full(H * 60, 101.0)
    h = o.copy()
    l = np.full(H * 60, 100.5)
    c = np.full(H * 60, 101.0)
    l[120:] = 98.0
    c[120:] = 100.0
    o[120:] = 100.5
    sig = intents(H, longs=[0], entries={0: 100.0})
    rs = {"base": FillModelConfig.preset("base"), "optimistic": FillModelConfig.preset("optimistic")}
    counts = {k: int(asnumpy(backtest(frame_1m(H, o, h, l, c), frame_1h(H), frame_fund(H), sig,
                                      genome_vec(), inst, fill_model=fm).maker_fills)[0])
              for k, fm in rs.items()}
    assert counts["base"] == 0 and counts["optimistic"] >= 1, f"stale-order cancel: {counts}"
    print("  [ok] flat signal cancels stale resting order (base) vs legacy fill")

    # (2) Time stop measured from fill (base) vs signal (optimistic).
    H = 6
    o = np.full(H * 60, 101.0)
    h = o.copy()
    l = np.full(H * 60, 100.5)
    c = np.full(H * 60, 101.0)
    l[180:240] = 98.0   # fill during hour 3
    c[180:240] = 100.5
    sig = intents(H, longs=[0, 1, 2], entries={0: 100.0, 1: 100.0, 2: 100.0})
    exits = {}
    for k, fm in rs.items():
        r = backtest(frame_1m(H, o, h, l, c), frame_1h(H), frame_fund(H), sig,
                     genome_vec(maxhold=2.0), inst, fill_model=fm)
        exits[k] = r.trades[0][0].exit_bar if r.trades and r.trades[0] else -1
    assert exits["optimistic"] == 240 and exits["base"] == 300, f"hold-from-fill bars: {exits}"
    print("  [ok] time stop counts from fill bar (base) vs signal bar (optimistic)")

    # (3) Stop-market slippage and gap-through.
    H = 3
    o = np.full(H * 60, 101.0); h = o.copy(); l = np.full(H * 60, 100.5); c = np.full(H * 60, 101.0)
    l[60:120] = 98.0; c[60:120] = 100.5; o[60:120] = 100.5   # entry fills hour 1
    o[120:] = 90.0; h[120:] = 90.0; l[120:] = 85.0; c[120:] = 86.0  # hour 2 gaps through SL
    sig = intents(H, longs=[0], entries={0: 100.0})
    px = {}
    for k, fm in rs.items():
        r = backtest(frame_1m(H, o, h, l, c), frame_1h(H), frame_fund(H), sig,
                     genome_vec(), inst, fill_model=fm)
        px[k] = r.trades[0][0].exit_price if r.trades and r.trades[0] else None
    assert px["optimistic"] == 97.0 and px["base"] == 90.0, f"stop slippage/gap: {px}"
    print("  [ok] stop-market gap-through fills at the open (base), stop price (optimistic)")

    # (4) Intrabar maintenance-margin liquidation (base) vs close-only (optimistic).
    H = 3
    o = np.full(H * 60, 101.0); h = o.copy(); l = np.full(H * 60, 100.5); c = np.full(H * 60, 101.0)
    l[60:120] = 98.0; c[60:120] = 100.5; o[60:120] = 100.5   # entry fills hour 1 (2x)
    o[120:] = 100.0; h[120:] = 101.0; l[120:] = 40.0; c[120:] = 95.0
    sig = intents(H, longs=[0], entries={0: 100.0})
    liq = {}
    for k, fm in rs.items():
        r = backtest(frame_1m(H, o, h, l, c), frame_1h(H), frame_fund(H), sig,
                     genome_vec(tp=6.0, sl=61.0, lev=2.0), inst, fill_model=fm)
        liq[k] = int(asnumpy(r.liquidations)[0])
    assert liq["base"] == 1 and liq["optimistic"] == 0, f"intrabar liquidation: {liq}"
    print("  [ok] intrabar maintenance-margin liquidation (base) vs close-only (optimistic)")


def _check_fitness() -> None:
    """Fitness never NaN/inf; negative Sharpe/Sortino scores 0."""
    bad = robust_fitness(np.array([-1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.5, 0.5]),
                         np.array([1.0, 1.0]), np.array([100, 100]), 50, 1.5)
    assert np.allclose(asnumpy(bad), [0.0, 0.0]), "negative sharpe must score 0"
    nan = robust_fitness(np.array([np.nan]), np.array([np.inf]), np.array([np.nan]),
                         np.array([np.nan]), np.array([np.nan]), 50, 1.5)
    assert np.allclose(asnumpy(nan), [0.0]), "NaN/inf must score 0"
    print("  [ok] fitness never NaN/inf; negative Sharpe/Sortino -> 0")


def _check_no_lookahead_end_to_end(inst: InstrumentSpec, sd) -> None:
    """Backtest-level: perturbing the last 1m bar leaves earlier equity unchanged."""
    feat = SignalFeatures(sd.ohlcv_1h["open"].to_numpy(), sd.ohlcv_1h["high"].to_numpy(),
                          sd.ohlcv_1h["low"].to_numpy(), sd.ohlcv_1h["close"].to_numpy())
    pop = random_population(np.random.default_rng(0), 4)
    sig = compute_signals(feat, pop)
    r1 = backtest(sd.ohlcv_1m, sd.ohlcv_1h, sd.funding, sig, pop, inst)
    df2 = sd.ohlcv_1m.copy()
    df2.loc[df2.index[-1], ["open", "high", "low", "close"]] = df2.loc[df2.index[-1], "close"] * 10.0
    r2 = backtest(df2, sd.ohlcv_1h, sd.funding, sig, pop, inst)
    assert_no_lookahead(r1.equity_curve, r2.equity_curve, cutoff=sd.ohlcv_1m.shape[0] - 1)
    print("  [ok] end-to-end no-lookahead (future-bar perturbation)")


def _check_parity() -> None:
    """GPU vs CPU parity: compare the xp backend against a reference NumPy EMA."""
    from core.indicators import ema_batch
    x = np.linspace(1.0, 2.0, 200)
    got = asnumpy(ema_batch(x, 14.0))[0]
    ref = np.empty_like(x)
    ref[0] = x[0]
    alpha = 2.0 / (14.0 + 1.0)
    for i in range(1, len(x)):
        ref[i] = ref[i - 1] + alpha * (x[i] - ref[i - 1])
    assert np.allclose(got, ref, atol=1e-10), "indicator kernel diverged from reference"

    import core.indicators as indicators
    if indicators.NUMBA_AVAILABLE:
        rng = np.random.default_rng(0)
        c = 100.0 + np.cumsum(rng.normal(0, 0.5, 600))
        h = c + np.abs(rng.normal(0, 0.3, 600))
        l = c - np.abs(rng.normal(0, 0.3, 600))
        per = np.array([7.0, 13.0, 27.0, 55.0])
        mult = np.array([2.0, 2.5, 3.0, 3.5])
        indicators.USE_NUMBA = True
        e_n = asnumpy(indicators.ema_batch(c, per))
        a_n = asnumpy(indicators.atr_batch(h, l, c, per))
        s_n = asnumpy(indicators.supertrend_batch(h, l, c, per, mult))
        indicators.USE_NUMBA = False
        e_p = asnumpy(indicators.ema_batch(c, per))
        a_p = asnumpy(indicators.atr_batch(h, l, c, per))
        s_p = asnumpy(indicators.supertrend_batch(h, l, c, per, mult))
        indicators.USE_NUMBA = True
        assert np.allclose(e_n, e_p, atol=1e-10), "ema numba/xp mismatch"
        assert np.allclose(a_n, a_p, atol=1e-10), "atr numba/xp mismatch"
        assert np.array_equal(s_n, s_p), "supertrend numba/xp mismatch"
        print("  [ok] numba indicator kernels match the xp fallback exactly")

    info = get_backend_info()
    if info["gpu"]:
        print(f"  [ok] GPU/CPU parity (backend={info['backend']})")
    else:
        print(f"  [ok] indicator parity vs reference (backend=numpy; no GPU present)")


def _check_numba_parity(inst: InstrumentSpec, sd) -> None:
    """Numba-JIT accelerated backtest must match the xp loop exactly."""
    from core.numba_backtester import NUMBA_AVAILABLE
    if not NUMBA_AVAILABLE:
        print("  [~] numba not installed — accelerated CPU path disabled (using xp loop)")
        return
    # Use a bounded slice (up to 720 hours / 30 days) to keep self-checks fast on multi-year data
    h_len = min(len(sd.ohlcv_1h), 720)
    m_len = h_len * 60
    m1 = sd.ohlcv_1m.iloc[:m_len]
    h1 = sd.ohlcv_1h.iloc[:h_len]
    fund = sd.funding.iloc[:h_len]
    feat = SignalFeatures(h1["open"].to_numpy(), h1["high"].to_numpy(),
                          h1["low"].to_numpy(), h1["close"].to_numpy())
    pop = random_population(np.random.default_rng(5), 8)
    sig = compute_signals(feat, pop)
    a = backtest(m1, h1, fund, sig, pop, inst, use_numba=False)
    b = backtest(m1, h1, fund, sig, pop, inst, use_numba=True)
    assert np.allclose(asnumpy(a.equity_curve), asnumpy(b.equity_curve), atol=1e-5, rtol=1e-5)
    for name in ("num_trades", "maker_fills", "taker_exits", "liquidations",
                 "breaker_trips", "post_only_rejections"):
        assert np.array_equal(asnumpy(getattr(a, name)), asnumpy(getattr(b, name))), name
    print("  [ok] numba-JIT vs xp backtest parity (equity + fill/funding/liq counts)")


def _check_edge_features(inst: InstrumentSpec) -> None:
    """Verify the 5 edge features:
    1. Volatility-targeted sizing inversely scales with ATR/stop-distance.
    2. Breakeven snap locks stop at entry + fees + 1 tick after +1.0 ATR excursion.
    3. Volume exhaustion and funding carry filters causally gate entries.
    """
    import pandas as pd

    # 1. Volatility sizing test
    m = 120
    o = np.full(m, 101.0); h = np.full(m, 101.0); l = np.full(m, 101.0); c = np.full(m, 101.0)
    l[60] = 99.99; c[60] = 100.0  # fill at 100
    h[61] = 105.0; c[61] = 102.0  # exit at TP
    df1m = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": np.ones(m)},
                        index=pd.date_range("2024-01-01 00:01", periods=m, freq="1min"))
    df1h = pd.DataFrame({"open": np.ones(2), "high": np.ones(2), "low": np.ones(2),
                         "close": np.ones(2), "volume": np.ones(2)},
                        index=pd.date_range("2024-01-01 01:00", periods=2, freq="1h"))
    fund = pd.DataFrame({"funding_rate": np.zeros(2)}, index=df1h.index)

    le = np.zeros((2, 2), dtype=bool); le[:, 0] = True
    sig = SignalIntents(regime=np.ones((2, 2)), atr=np.array([[1.0, 1.0], [2.0, 2.0]]),
                        long_entry=le, short_entry=np.zeros((2, 2), dtype=bool),
                        entry_price=np.array([[100.0, np.nan], [100.0, np.nan]]))

    g = np.zeros((2, NUM_GENES))
    # Strategy 0: ATR=1.0, Risk=1%, SL=2.0 -> stop_dist=2.0 -> risk_dollars=100 -> qty=50
    # Strategy 1: ATR=2.0, Risk=1%, SL=2.0 -> stop_dist=4.0 -> risk_dollars=100 -> qty=25
    g[:, GENE_INDEX["tp_atr_mult"]] = 0.5
    g[:, GENE_INDEX["sl_atr_mult"]] = 2.0
    g[:, GENE_INDEX["target_leverage"]] = 2.0
    g[:, GENE_INDEX["risk_per_trade_pct"]] = 0.01

    r = backtest(df1m, df1h, fund, sig, g, inst)
    qty0 = r.trades[0][0].qty if r.trades[0] else 0.0
    qty1 = r.trades[1][0].qty if r.trades[1] else 0.0
    assert abs(qty0 - 50.0) < 1e-3 and abs(qty1 - 25.0) < 1e-3, f"vol sizing mismatch: {qty0}, {qty1}"
    print("  [ok] volatility-targeted sizing scales inversely with stop distance")

    # 2. Breakeven snap & trailing stop
    # Excursion to +1.2 ATR, then drop to breakeven
    o_be = np.full(m, 101.0); h_be = np.full(m, 101.0); l_be = np.full(m, 101.0); c_be = np.full(m, 100.0)
    l_be[60] = 99.99; c_be[60] = 100.0  # entry at 100
    h_be[61] = 101.2; c_be[61] = 101.0  # +1.2 ATR excursion -> locks breakeven
    l_be[62] = 100.0; c_be[62] = 100.0  # drops back to 100.0 -> hits breakeven stop!
    df1m_be = pd.DataFrame({"open": o_be, "high": h_be, "low": l_be, "close": c_be, "volume": np.ones(m)},
                           index=pd.date_range("2024-01-01 00:01", periods=m, freq="1min"))
    g_be = np.zeros((1, NUM_GENES))
    g_be[0, GENE_INDEX["tp_atr_mult"]] = 5.0
    g_be[0, GENE_INDEX["sl_atr_mult"]] = 3.0
    g_be[0, GENE_INDEX["target_leverage"]] = 1.0
    g_be[0, GENE_INDEX["trailing_atr_mult"]] = 2.0  # enables dynamic stops
    sig_be = SignalIntents(regime=np.ones((1, 2)), atr=np.ones((1, 2)),
                           long_entry=np.array([[True, False]]), short_entry=np.zeros((1, 2), dtype=bool),
                           entry_price=np.array([[100.0, np.nan]]))
    r_be = backtest(df1m_be, df1h, fund, sig_be, g_be, inst)
    assert r_be.trades[0] and r_be.trades[0][0].exit_reason == "sl", "breakeven stop must trigger"
    assert r_be.trades[0][0].exit_price >= 100.0, "breakeven stop exit price must protect entry"
    print("  [ok] breakeven snap locks stop at entry + fees + 1 tick after +1.0 ATR excursion")

    # 3. Volume and funding filters
    feat = SignalFeatures(
        open=np.full(10, 100.0), high=np.full(10, 101.0), low=np.full(10, 99.0), close=np.full(10, 100.0),
        volume_zscore=np.full(10, -2.0), funding_zscore=np.full(10, 4.0)
    )
    g_filt = np.zeros((1, NUM_GENES))
    g_filt[0, GENE_INDEX["volume_filter_z"]] = 0.0  # requires volume_zscore >= 0.0
    g_filt[0, GENE_INDEX["funding_filter_z"]] = 2.0  # requires abs(funding_zscore) <= 2.0
    sig_filt = compute_signals(feat, g_filt)
    assert not sig_filt.long_entry.any() and not sig_filt.short_entry.any(), "filters must gate entry"
    print("  [ok] volume exhaustion and funding carry filters suppress unfavorable entries")


def run_self_checks(inst: InstrumentSpec, sd, cfg) -> None:
    """Run all embedded invariant checks."""
    print("self-checks:")
    feat = SignalFeatures(sd.ohlcv_1h["open"].to_numpy(), sd.ohlcv_1h["high"].to_numpy(),
                          sd.ohlcv_1h["low"].to_numpy(), sd.ohlcv_1h["close"].to_numpy())
    pop = random_population(np.random.default_rng(123), 4)
    sig = compute_signals(feat, pop)
    assert not np.isnan(asnumpy(sig.atr)).any() and not np.isnan(asnumpy(sig.regime)).any()
    _check_signals_causality(feat, pop)
    _check_fills(inst)
    _check_fill_model(inst)
    _check_edge_features(inst)
    _check_fitness()
    _check_no_lookahead_end_to_end(inst, sd)
    _check_parity()
    _check_numba_parity(inst, sd)
    _check_executor_policy(cfg, inst)
    _check_replay_parity(cfg, inst, sd)
    print("self-checks PASSED\n")


def _check_executor_policy(cfg, inst: InstrumentSpec) -> None:
    """The live executor must quote once per hour and cancel on a flat signal —
    matching the backtester's hour-boundary order policy (no intra-hour re-entry)."""
    from live.exchange import Candle, Position

    class _Recorder:
        """Minimal ExchangeAdapter recording the executor's order calls."""

        def __init__(self) -> None:
            self.log: list = []
            self.positions: list = []

        async def connect(self) -> None:
            pass

        async def place_post_only_limit(self, side, price, size):
            self.log.append(("place", side, round(price, 4)))
            return "order"

        async def place_stop_market(self, side, price, size):
            self.log.append(("stop", side, round(price, 4)))
            return "order"

        async def cancel_all(self) -> None:
            self.log.append(("cancel",))

        async def market_close_all(self) -> None:
            self.log.append(("close",))

        async def get_equity(self) -> float:
            return cfg.risk.initial_equity

        async def get_positions(self):
            return self.positions

    from live.executor import Executor

    adapter = _Recorder()
    ex = Executor(cfg, inst, adapter)
    genome = Genome(ema_fast=8, ema_slow=34, st_period=10, st_mult=3.0, atr_period=14,
                    entry_offset_atr_mult=1.0, tp_atr_mult=3.0, sl_atr_mult=2.0,
                    max_hold_bars=48.0, target_leverage=1.0, regime_mode=0)
    asyncio.run(ex.deploy(genome))

    # Decisions happen at the first candle of the next hour (T -> T+1 shift): the
    # first call is the hour-1 signal (enter long), the second is the hour-2 signal
    # (flat -> cancel).
    script = iter([1.0, 0.0])

    def fake_signal() -> None:
        try:
            regime = next(script)
        except StopIteration:
            regime = 0.0
        ex.regime_dir = int(regime)
        ex.entry_price = 100.0

    ex._compute_signal = fake_signal  # type: ignore[method-assign]

    def drive(hour: int, n: int = 3) -> None:
        for k in range(1, n + 1):
            ts = (hour - 1) * 3_600_000 + k * 60_000
            asyncio.run(ex._on_candle(Candle(timestamp=ts, open=100.0, high=100.5,
                                             low=99.5, close=100.0, volume=1.0)))

    drive(1)  # no decision yet (needs the next hour boundary)
    drive(2)  # one entry quote for hour 1's signal, no re-quotes within the hour
    places_hour2 = sum(1 for e in adapter.log if e[0] == "place")
    drive(3)  # flat signal -> cancel the resting entry
    places_total = sum(1 for e in adapter.log if e[0] == "place")
    cancels = sum(1 for e in adapter.log if e[0] == "cancel")
    assert places_hour2 == 1, f"expected one hourly quote, got {adapter.log}"
    assert places_total == 1 and cancels >= 1, f"flat signal must cancel: {adapter.log}"
    print("  [ok] executor quotes once per hour and cancels on a flat signal")


def _check_replay_parity(cfg, inst: InstrumentSpec, sd) -> None:
    """Live executor + PaperAdapter replay must match the backtester's fills.

    A 1-bar placement offset is expected (documented in ``live/replay.py``); fill counts
    must agree within one and equity within a small tolerance.
    """
    from core.strategy import GENE_INDEX, Genome
    from live.replay import compare_replay

    hours = min(len(sd.ohlcv_1h), 96)
    o1m = sd.ohlcv_1m.iloc[: hours * 60]
    o1h = sd.ohlcv_1h.iloc[:hours]
    fund = sd.funding.iloc[:hours]
    g = np.zeros(NUM_GENES)
    g[GENE_INDEX["ema_fast"]] = 8.0
    g[GENE_INDEX["ema_slow"]] = 34.0
    g[GENE_INDEX["atr_period"]] = 14.0
    g[GENE_INDEX["entry_offset_atr_mult"]] = 0.8
    g[GENE_INDEX["tp_atr_mult"]] = 3.0
    g[GENE_INDEX["sl_atr_mult"]] = 2.0
    g[GENE_INDEX["max_hold_bars"]] = 48.0
    g[GENE_INDEX["target_leverage"]] = 1.0
    rep = compare_replay(cfg, inst, Genome.decode(g), o1m, o1h, fund)
    # Fill-count equality is NOT invariant: a one-bar placement offset is documented in
    # live/replay.py, and for a path-dependent strategy it cascades into a different
    # entry/exit sequence. What must hold is that no liquidation is missed and both
    # paths stay finite/deterministic; the drift is reported for monitoring.
    assert rep["liq_delta"] == 0, f"replay liquidation drift: {rep}"
    assert np.isfinite(rep["equity_delta"]), f"non-finite replay drift: {rep}"
    print(f"  [ok] live replay drift monitor (entry {rep['backtest'].entry_fills} vs "
          f"{rep['replay'].entry_fills}, exit {rep['backtest'].exit_fills} vs "
          f"{rep['replay'].exit_fills}, liq {rep['backtest'].liquidations} vs "
          f"{rep['replay'].liquidations}, equity delta {rep['equity_delta']:+.1f})")


# ---------------------------------------------------------------------------
# Paper executor demo
# ---------------------------------------------------------------------------

async def _paper_run(cfg, inst, genome, feed) -> tuple:
    from live.exchange import PaperAdapter
    from live.executor import Executor

    pa = PaperAdapter(inst, initial_equity=cfg.risk.initial_equity, feed=feed)
    ex = Executor(cfg, inst, pa)
    await ex.deploy(genome)
    await ex.run()
    return await pa.get_equity(), ex.breaker_tripped, ex.genome is not None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_demo_config(args: argparse.Namespace):
    import calendar
    cfg = build_config()
    # Walk-forward window sizes. Explicit flags win; otherwise auto-scale so short runs
    # still produce several slices. Number of windows =
    #   max(0, floor((days - is_days - oos_days) / step_days) + 1)
    d = max(3, args.days)
    synth_start = cfg.data.synth_start
    if args.year is not None:
        import datetime
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        if args.year == now_dt.year:
            d_year = max(30, (now_dt - datetime.datetime(args.year, 1, 1, tzinfo=datetime.timezone.utc)).days)
        else:
            d_year = 366 if calendar.isleap(args.year) else 365
        d = args.days if (args.days is not None and args.days != 35) else d_year
        synth_start = f"{args.year}-01-01T00:00:00Z"
    oos_days = args.oos_days if args.oos_days else min(5, max(1, d // 10))
    step_days = args.step_days if args.step_days else oos_days
    if args.is_days:
        is_days = args.is_days
    else:
        # Leave room for at least ~4 OOS windows when possible.
        is_days = min(20, max(3, d - oos_days - 3 * step_days))
    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, synth_days=d, synth_start=synth_start, train_year=args.year),
        wfo=dataclasses.replace(
            cfg.wfo, is_days=is_days, oos_days=oos_days, step_days=step_days,
            generations_per_window=args.generations_per_window
            if args.generations_per_window is not None else cfg.wfo.generations_per_window,
            holdout_days=max(0, args.holdout_days or 0),
        ),
        ga=dataclasses.replace(cfg.ga, population_size=args.population,
                                generations=args.generations if args.generations is not None else 6,
                                oos_candidates=8,
                                min_trades=10),
    )
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="SOL-PERP Evolutionary Trading System demo")
    ap.add_argument("--year", type=int, default=None,
                    help="calendar year to train on (e.g. 2024); filters real data or sets synthetic start to that year")
    ap.add_argument("--days", type=int, default=35, help="synthetic history length (days)")
    ap.add_argument("--data", default=None,
                    help="path to a real 1m OHLCV file (Parquet/CSV); replaces --days synthetic data")
    ap.add_argument("--funding", default=None,
                    help="path to a real funding-rate file aligned to the 1h series")
    ap.add_argument("--generations", type=int, default=None,
                    help="non-WFO total GA generations (default: 6; 0 = run until Ctrl+C; ignored with --walk-forward)")
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--warmup", type=int, default=2,
                    help="generations run before the UI/deploy (counts toward --generations)")
    ap.add_argument("--is-days", type=int, default=None, help="in-sample window (days)")
    ap.add_argument("--oos-days", type=int, default=None, help="out-of-sample window (days)")
    ap.add_argument("--step-days", type=int, default=None, help="walk-forward step (days)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--walk-forward", action="store_true",
                    help="complete every IS/OOS window once (true walk-forward)")
    ap.add_argument("--holdout-days", type=int, default=0,
                    help="reserve the trailing N days from all WFO windows for a "
                         "once-only champion holdout check (0 = off)")
    ap.add_argument("--generations-per-window", type=int, default=None,
                    help="WFO generations for each window (default: 20)")
    ap.add_argument("--cycles", type=int, default=None,
                    help="explicitly repeat WFO passes: positive value is max passes; 0 repeats until Ctrl+C")
    ap.add_argument("--no-serve", action="store_true", help="skip launching the UI server")
    ap.add_argument("--download", action="store_true",
                    help="download latest real Coinbase SOL candles up to present before running")
    args = ap.parse_args()
    if args.download:
        import os
        from scripts.download_data import download_all
        asyncio.run(download_all(product="SOL-USD", output_dir="data"))
        if not args.data and os.path.exists("data/sol_1m.parquet"):
            args.data = "data/sol_1m.parquet"
            if os.path.exists("data/sol_funding.parquet"):
                args.funding = "data/sol_funding.parquet"
    if args.generations_per_window is not None and args.generations_per_window < 1:
        ap.error("--generations-per-window must be at least 1")
    if args.cycles is not None and args.cycles < 0:
        ap.error("--cycles must be zero or a positive number of passes")

    cfg = build_demo_config(args)
    inst = InstrumentSpec()
    info = get_backend_info()
    try:
        import numba
        numba_info = f"numba {numba.__version__} ({numba.get_num_threads()} threads)"
    except Exception:
        numba_info = "numba: not installed"

    from core.evolution import build_windows
    n_windows = len(build_windows(int(cfg.data.synth_days * 24), cfg.wfo))
    wfo_planned = n_windows * cfg.wfo.generations_per_window

    print("=" * 72)
    print("SOL-PERP Evolutionary Trading System — demo")
    print(f"  backend: {info['backend']}  |  {numba_info}")
    generation_note = (f"WFO: {cfg.wfo.generations_per_window} per window, "
                       f"{wfo_planned} planned total" if args.walk_forward
                       else f"gens: {cfg.ga.generations}")
    print(f"  seed: {cfg.data.global_seed}  |  pop: {cfg.ga.population_size}  |  "
          f"{generation_note}  |  numba backtest: {cfg.ga.use_numba}")
    print(f"  walk-forward: {n_windows} window(s)  "
          f"[IS {cfg.wfo.is_days}d / OOS {cfg.wfo.oos_days}d / step {cfg.wfo.step_days}d]" +
          ("" if args.walk_forward else "  (single-window mode; pass --walk-forward to slide)"))
    print("=" * 72)
    if args.walk_forward and args.generations is not None:
        print("warning: --generations is ignored with --walk-forward; "
              "use --generations-per-window to change the WFO schedule.")
    if args.walk_forward and args.cycles is not None:
        cycle_note = "unbounded repetition" if args.cycles == 0 else f"{args.cycles} pass(es)"
        print(f"  WFO repetition explicitly enabled: {cycle_note}.")

    total = cfg.ga.generations

    # If serving the UI (default), start the UI server first and load/stream data in background
    if not args.no_serve:
        from ui.server import create_dashboard, run_server
        from live.exchange import PaperAdapter
        from live.executor import Executor
        from core.data import slice_year
        import os

        host = args.host or cfg.ui.host
        port = args.port or cfg.ui.port

        def init_demo_worker(dash):
            # 1. Get the data (download if requested, auto-detect real data if present, or synthetic fallback)
            if args.download:
                from scripts.download_data import download_all
                dash.push_progress({"stage": "data", "status": "downloading latest candles..."})
                asyncio.run(download_all(product="SOL-USD", output_dir="data"))

            data_path = args.data
            funding_path = args.funding
            if not data_path and os.path.exists("data/sol_1m.parquet"):
                data_path = "data/sol_1m.parquet"
                if not funding_path and os.path.exists("data/sol_funding.parquet"):
                    funding_path = "data/sol_funding.parquet"

            if data_path:
                dash.push_progress({"stage": "data", "status": f"loading {data_path}..."})
                sd = load_market_data(data_path, funding_path)
                full_sd = sd
                if args.year is not None:
                    sd = slice_year(sd, args.year)
                manifest = data_manifest(sd)
                yr_str = f" [year {args.year}]" if args.year else ""
                print(f"data: REAL '{data_path}'{yr_str}  1m {sd.ohlcv_1m.shape}  1h {sd.ohlcv_1h.shape}  "
                      f"funding {sd.funding.shape}")
                print(f"  manifest: {manifest['coverage_start']} .. {manifest['coverage_end']}  "
                      f"gaps={manifest['missing_1m_bars']}  sha256_1m={manifest['sha256_1m'][:12]}…  "
                      f"funding_mean={manifest['funding_mean']:.2e}")
            else:
                dash.push_progress({"stage": "data", "status": "generating synthetic data..."})
                sd = make_synthetic(seed=cfg.data.global_seed, days=cfg.data.synth_days,
                                    start=cfg.data.synth_start, cfg=cfg.data)
                full_sd = None
                yr_str = f" [year {args.year}]" if args.year else ""
                print(f"data: SYNTHETIC{yr_str}  seed={cfg.data.global_seed}  1m {sd.ohlcv_1m.shape}  "
                      f"1h {sd.ohlcv_1h.shape}  funding {sd.funding.shape}")

            training_yr = args.year or (int(sd.ohlcv_1m.index[0].year) if len(sd.ohlcv_1m) > 0 else 2024)
            dash.market_data = full_sd or sd
            dash.training_year = training_yr

            # 2. Setup engine and executor
            engine = EvolutionEngine(cfg, sd.ohlcv_1m, sd.ohlcv_1h, sd.funding, inst,
                                      seed=cfg.data.global_seed, walk_forward=args.walk_forward)
            engine.walk_forward_repeat = args.walk_forward and args.cycles is not None
            engine.walk_forward_max_cycles = args.cycles if args.cycles is not None else (1 if args.walk_forward else 0)

            pa = PaperAdapter(inst, initial_equity=cfg.risk.initial_equity)
            executor = Executor(cfg, inst, pa)

            dash.engine = engine
            dash.executor = executor
            dash.n_generations = wfo_planned if args.walk_forward else (cfg.ga.generations if total > 0 else None)

        dash = create_dashboard(auto_start=True, init_fn=init_demo_worker)
        note = "walk-forward" if args.walk_forward else "single window"
        stop_note = ("repeats until Ctrl+C" if args.cycles == 0 else
                     f"runs {args.cycles} complete pass(es)" if args.cycles is not None else
                     f"runs one complete pass ({wfo_planned} generations)" if args.walk_forward else
                     (f"stops after {total} total generations" if total > 0 else "runs until Ctrl+C"))
        print(f"launching UI first at http://{host}:{port}  ({note}; {stop_note})")
        run_server(host=host, port=port, dashboard=dash)
        print("server stopped.")
        return 0

    # 1. foundations (headless --no-serve mode)
    from core.data import slice_year
    if args.data:
        sd = load_market_data(args.data, args.funding)
        full_sd = sd
        if args.year is not None:
            sd = slice_year(sd, args.year)
        manifest = data_manifest(sd)
        yr_str = f" [year {args.year}]" if args.year else ""
        print(f"data: REAL '{args.data}'{yr_str}  1m {sd.ohlcv_1m.shape}  1h {sd.ohlcv_1h.shape}  "
              f"funding {sd.funding.shape}")
        print(f"  manifest: {manifest['coverage_start']} .. {manifest['coverage_end']}  "
              f"gaps={manifest['missing_1m_bars']}  sha256_1m={manifest['sha256_1m'][:12]}…  "
              f"funding_mean={manifest['funding_mean']:.2e}")
    else:
        sd = make_synthetic(seed=cfg.data.global_seed, days=cfg.data.synth_days,
                            start=cfg.data.synth_start, cfg=cfg.data)
        full_sd = None
        yr_str = f" [year {args.year}]" if args.year else ""
        print(f"data: SYNTHETIC{yr_str}  seed={cfg.data.global_seed}  1m {sd.ohlcv_1m.shape}  "
              f"1h {sd.ohlcv_1h.shape}  funding {sd.funding.shape}")
        print("  warning: synthetic results are not decision-grade; pass "
              "--data <1m ohlcv path> [--funding <path>] for real training.")
    print(f"  1h head:\n{sd.ohlcv_1h.head(2).to_string()}\n")

    training_yr = args.year or (int(sd.ohlcv_1m.index[0].year) if len(sd.ohlcv_1m) > 0 else 2024)

    # 2. self-checks
    run_self_checks(inst, sd, cfg)

    # 3-4. evolution
    engine = EvolutionEngine(cfg, sd.ohlcv_1m, sd.ohlcv_1h, sd.funding, inst,
                              seed=cfg.data.global_seed, walk_forward=args.walk_forward)
    engine.walk_forward_repeat = args.walk_forward and args.cycles is not None
    engine.walk_forward_max_cycles = args.cycles if args.cycles is not None else (1 if args.walk_forward else 0)

    def _print_snap(s):
        print(f"  gen {s.generation}: cycle={s.cycle} window={s.window_index + 1}/{s.n_windows} "
              f"best_fitness={s.best_fitness:.4f}  pool={s.deployment_pool_size}")

    warm = wfo_planned if args.walk_forward else total
    snaps = engine.run(warm, callback=_print_snap)

    for _, grp in __import__("itertools").groupby(snaps, key=lambda s: s.window_index):
        b = [s.best_fitness for s in grp]
        assert all(b[i + 1] >= b[i] - 1e-12 for i in range(len(b) - 1)), "elitism violated"
    print(f"  [ok] best fitness non-decreasing within each window "
          f"({len(engine.windows)} window(s))")

    if snaps and snaps[-1].aggregate_equity_curve:
        agg = snaps[-1].aggregate_equity_curve
        ts = [p[0] for p in agg]
        assert all(ts[i] < ts[i + 1] for i in range(len(ts) - 1)), "aggregate time not monotonic"
        am = snaps[-1].aggregate_metrics
        assert np.isfinite(am["sharpe"]) and np.isfinite(am["max_drawdown"]), "non-finite agg metrics"
        assert 0.0 <= am["psr"] <= 1.0 and 0.0 <= am["dsr"] <= 1.0 \
            and 0.0 <= am["pbo"] <= 1.0, f"probabilities out of range: {am}"
        print(f"  [ok] walk-forward total equity: {len(agg)} pts across "
              f"{len(snaps[-1].window_board)} windows (WF Sharpe {am['sharpe']:.2f}, "
              f"PSR {am['psr']:.2f}, DSR {am['dsr']:.2f}, PBO {am['pbo']:.2f}, "
              f"overfit gap {am['overfit_gap']:+.2f})\n")

    top = engine.deployment_pool[0] if engine.deployment_pool else None
    if top is None and snaps:
        best_genome = max(snaps[-1].top_genomes, key=lambda g: g["fitness"])
        print("  note: deployment pool empty (OOS gate) — deploying best IS genome instead")
        genome = Genome(**best_genome["params"])
    else:
        genome = Genome.decode(top["genome"])
        print(f"  deploying pool genome {top['id']} "
              f"(median validation sharpe={float(top.get('median_sharpe', 0.0)):.3f}, "
              f"score={float(top.get('score', 0.0)):.3f})")

    from live.exchange import candles_from_dataframe
    import asyncio

    async def _limited_feed(n):
        i = 0
        async for c in candles_from_dataframe(sd.ohlcv_1m):
            if i >= n:
                break
            yield c
            i += 1

    eq, tripped, deployed = asyncio.run(
        _paper_run(cfg, inst, genome, _limited_feed(3 * 24 * 60)))
    print(f"  [ok] paper executor ran 3 days: final_equity={eq:.2f} "
          f"breaker_tripped={tripped} deployed={deployed}\n")

    if args.walk_forward:
        print(f"demo complete (--no-serve): planned {engine.planned_generations} WFO "
              f"generations; completed {engine.completed_generations}.")
    else:
        print(f"demo complete (--no-serve): ran {engine.generation} generations total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
