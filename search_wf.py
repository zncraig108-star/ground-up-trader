"""search_wf.py - unattended walk-forward strategy search for the $500 -> $5,000 goal.

Data     : data/{eth,sol,btc}_15m.parquet from fetch_cb.py (real Coinbase candles).
Universe : ETH/SOL/BTC micro futures (NYS-legal), whole contracts, long+short, one position at a time.
Search   : random (seeded) configs over 5 signal families x 4 timeframes x sizing/exit params.
Protocol : for each quarterly test window from 2024-01, pick the config with the best TRAIN-only growth
           (data strictly before the window), then run it on the window carrying equity+floor forward.
           Stitched OOS must reach >=10x with >=40 trades/yr, zero floor breach, yield <5% of PnL.
Causality: bar i is decided at its close, orders fill at the open of bar i+1 (entries AND exits);
           stops are checked before favourable updates inside a bar; open positions are closed at window end.
Output   : prints progress; writes data/wf_result_<seed>.json (read by the monitor agent).
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, pandas as pd
from numba import njit

ASSETS = {"eth": 0.10, "sol": 1.00, "btc": 0.01}  # MET, SOL, BTI contract multipliers
MULTS = np.array(list(ASSETS.values()))
FEE, SLIP, LEV = 0.0005, 0.0002, 3.0  # ASSUMPTION: taker 5bp + 2bp slip per side; 3x margin cap
LT, LF = np.array([650.0, 1500.0, 3000.0]), np.array([500.0, 1000.0, 2000.0])  # ratchets
BPH = {"15min": 4.0, "30min": 2.0, "1h": 1.0, "4h": 0.25}  # bars per hour


@njit(cache=True)
def _net(side, n, m, ent, px, fee, slip):
    return (px * (1 - slip * side) - ent) * side * n * m - n * m * px * fee


@njit(cache=True)
def run(O, H, L, C, ATR, SIDE, SCORE, EXL, EXS, lo, hi, eq0, floor0, k, trail, r_lo, r_x,
        mults, ybar, fee, slip, lt, lf, lev, dthr):
    eq = cash = eq0
    floor = floor0
    pa = -1; ps = 0; pn = 0; pe = 0.0; pst = 0.0; pex = 0.0; pat = 0.0
    qa = -1; qs = 0; qatr = 0.0; qexit = False
    trades = wins = breach = 0
    mineq = eq0; yld = 0.0; peak = eq0; mdd = 0.0
    for i in range(lo, hi):
        if pa >= 0 and qexit:  # exit decided at close i-1 -> open i
            m = mults[pa]; p = O[pa, i]
            net = _net(ps, pn, m, pe, p, fee, slip); cash += net
            if net > 0: wins += 1
            pa = -1
        qexit = False
        if qa >= 0 and pa < 0:  # entry decided at close i-1 -> open i
            m = mults[qa]; px = O[qa, i] * (1 + slip * qs)
            sd = k * qatr
            free = eq - floor
            budget = min(r_lo * eq + r_x * max(0.0, eq - 500.0), free / 1.5)  # gap-adjusted floor guard
            n = int(budget // (sd * m)) if eq >= 500.0 else 1  # defensive posture: exactly 1 contract
            n = max(0, min(n, int(lev * eq // (px * m))))
            if n >= 1 and free > 0:
                cash -= n * m * O[qa, i] * fee; trades += 1
                pa = qa; ps = qs; pn = n; pe = px; pst = px - qs * sd; pex = px; pat = qatr
        qa = -1
        if pa >= 0:  # stops first (conservative), then favourable updates
            m = mults[pa]
            hit = -1.0
            if ps * (O[pa, i] - pst) <= 0: hit = O[pa, i]
            elif ps > 0 and L[pa, i] <= pst: hit = pst
            elif ps < 0 and H[pa, i] >= pst: hit = pst
            if hit > 0:
                net = _net(ps, pn, m, pe, hit, fee, slip); cash += net
                if net > 0: wins += 1
                pa = -1
            else:
                pex = max(pex, H[pa, i]) if ps > 0 else min(pex, L[pa, i])
                new = pex - ps * trail * pat
                if ps * (pex - pe) >= 2.0 * pat and ps * (new - pe) <= 0:  # breakeven lock +0.05 ATR
                    new = pe + ps * 0.05 * pat
                if ps * (new - pst) > 0: pst = new
        if pa >= 0 and i == hi - 1:  # flatten at window end
            m = mults[pa]; net = _net(ps, pn, m, pe, C[pa, i], fee, slip); cash += net
            if net > 0: wins += 1
            pa = -1
        unreal = used = 0.0
        if pa >= 0:
            c = C[pa, i]; m = mults[pa]
            unreal = (c - pe) * ps * pn * m; used = pn * m * c * 0.33
        y = max(0.0, cash + unreal - used) * ybar
        cash += y; yld += y
        eq = cash + unreal
        for j in range(lt.shape[0]):
            if eq >= lt[j] and lf[j] > floor: floor = lf[j]
        mineq = min(mineq, eq)
        if eq < floor - 1e-9: breach += 1
        peak = max(peak, eq); mdd = max(mdd, (peak - eq) / peak)
        if pa >= 0:
            if (ps > 0 and EXL[pa, i]) or (ps < 0 and EXS[pa, i]): qexit = True
        elif i < hi - 1 and eq > floor:
            best = -1.0; ba = -1
            for a in range(O.shape[0]):
                s = SIDE[a, i]
                if s != 0 and ATR[a, i] > 0 and SCORE[a, i] >= (dthr if eq < 500.0 else 0.0) and SCORE[a, i] > best:
                    best = SCORE[a, i]; ba = a
            if ba >= 0:
                qa = ba; qs = SIDE[ba, i]; qatr = ATR[ba, i]
    return eq, floor, trades, wins, breach, mineq, yld, mdd


def load_tf(tf: str, nrows: int | None = None) -> dict:
    """Resample real 15m -> tf for all assets on a common index; ATR(14) and Kaufman ER(30) are causal."""
    fr = []
    for a in ASSETS:
        d = pd.read_parquet(f"data/{a}_15m.parquet")  # REAL candles; the 1m files are interpolated (leak)
        d = d.iloc[:nrows] if nrows else d
        fr.append(d.resample(tf).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                      "volume": "sum"}).dropna())
    idx = fr[0].index
    for f in fr[1:]:
        idx = idx.intersection(f.index)
    fr = [f.reindex(idx) for f in fr]
    out = dict(idx=idx, fr=fr, tf=tf)
    for key, col in (("O", "open"), ("H", "high"), ("L", "low"), ("C", "close")):
        out[key] = np.ascontiguousarray(np.stack([f[col].values for f in fr]))
    atr, er = [], []
    for f in fr:
        c = f.close
        tr = np.maximum(f.high - f.low, np.maximum((f.high - c.shift()).abs(), (f.low - c.shift()).abs()))
        atr.append(tr.rolling(14).mean().fillna(0).values)
        er.append(((c - c.shift(30)).abs() / c.diff().abs().rolling(30).sum().clip(lower=1e-9)).fillna(0).values)
    out["ATR"], out["ER"] = np.stack(atr), np.stack(er)
    return out


def signals(T: dict, fam: str, p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-asset causal signals at bar close: entry side (+1/-1/0), exit-long, exit-short flags."""
    sides, exl, exs = [], [], []
    for a, f in enumerate(T["fr"]):
        c, h, l = f.close, f.high, f.low
        er = pd.Series(T["ER"][a], index=c.index)
        okL, okS = er >= p.get("er", 0.0), er >= p.get("er", 0.0) + 0.05  # shorts need extra efficiency
        if fam == "donch":
            n, m = p["n"], max(2, p["n"] // 3)
            lg, sh = c > h.rolling(n).max().shift(1), c < l.rolling(n).min().shift(1)
            xl, xs = c < l.rolling(m).min().shift(1), c > h.rolling(m).max().shift(1)
        elif fam == "ema":
            tr = np.sign(c.ewm(span=p["f"], adjust=False).mean() - c.ewm(span=p["s"], adjust=False).mean())
            ch = tr != tr.shift(1)
            lg, sh, xl, xs = (tr > 0) & ch, (tr < 0) & ch, tr < 0, tr > 0
        elif fam == "mr":
            z = (c - c.rolling(p["n"]).mean()) / c.rolling(p["n"]).std()
            lg, sh, xl, xs = z < -p["z"], z > p["z"], z > 0, z < 0
            okL = okS = er <= 1.0
        elif fam == "mom":
            r = c / c.shift(p["n"]) - 1
            thr = p["z"] * pd.Series(T["ATR"][a], index=c.index) / c * np.sqrt(p["n"])
            st = np.where(r > thr, 1, np.where(r < -thr, -1, 0))
            st = pd.Series(st, index=c.index)
            lg, sh, xl, xs = (st > 0) & (st.shift(1) <= 0), (st < 0) & (st.shift(1) >= 0), r < 0, r > 0
        else:  # squeeze: ATR compression then range break
            atr = pd.Series(T["ATR"][a], index=c.index)
            comp = (atr / atr.rolling(p["n"] * 4).mean()).shift(1) < p["q"]
            n, m = p["n"], max(2, p["n"] // 3)
            lg = (c > h.rolling(n).max().shift(1)) & comp
            sh = (c < l.rolling(n).min().shift(1)) & comp
            xl, xs = c < l.rolling(m).min().shift(1), c > h.rolling(m).max().shift(1)
        side = np.where(lg & okL, 1, np.where(sh & okS, -1, 0)).astype(np.int64)
        sides.append(side); exl.append(xl.fillna(False).values); exs.append(xs.fillna(False).values)
    return np.stack(sides), np.stack(exl), np.stack(exs)


def sim(T, sig, lo, hi, eq0, floor0, sz) -> dict:
    side, exl, exs = sig
    r = run(T["O"], T["H"], T["L"], T["C"], T["ATR"], side, T["ER"], exl, exs, lo, hi, eq0, floor0,
            sz["k"], sz["trail"], sz["r_lo"], sz["r_x"], MULTS, 0.05 / (8760 * BPH[T["tf"]]), FEE, SLIP,
            LT, LF, LEV, 0.38)
    return dict(zip(("eq", "floor", "trades", "wins", "breach", "mineq", "yld", "mdd"), r))


FAMS = {"donch": dict(n=[20, 40, 80, 160, 320], er=[0.0, 0.2, 0.3, 0.4]),
        "ema": dict(fs=[(10, 50), (20, 100), (30, 150), (50, 200)], er=[0.0, 0.2, 0.3]),
        "mr": dict(n=[20, 50, 100], z=[1.5, 2.0, 2.5, 3.0]),
        "mom": dict(n=[24, 48, 96, 192], z=[0.5, 1.0, 1.5], er=[0.0, 0.2]),
        "sq": dict(n=[20, 40, 80], q=[0.6, 0.8], er=[0.0, 0.2])}
SIZING = dict(k=[1.0, 1.5, 2.0, 3.0, 4.0], trail=[2.0, 3.0, 5.0, 8.0], r_lo=[0.01, 0.02, 0.04],
              r_x=[0.1, 0.25, 0.5, 1.0])


def sample(rng) -> dict:
    fam = str(rng.choice(list(FAMS)))
    p = {}
    for key, vals in FAMS[fam].items():
        v = vals[int(rng.integers(len(vals)))]
        if key == "fs":
            p["f"], p["s"] = v
        else:
            p[key] = v
    return dict(fam=fam, tf=str(rng.choice(list(BPH))), p=p,
                sz={k: vals[int(rng.integers(len(vals)))] for k, vals in SIZING.items()})


def assert_no_lookahead() -> None:
    """Signals at bars well before a cut must be identical with and without later data."""
    full, part = load_tf("1h"), load_tf("1h", nrows=100000)
    n = len(part["idx"]) - 100
    for fam, p in (("donch", dict(n=40, er=0.2)), ("ema", dict(f=20, s=100, er=0.2)), ("mr", dict(n=50, z=2.0)),
                   ("mom", dict(n=48, z=1.0, er=0.0)), ("sq", dict(n=40, q=0.8, er=0.0))):
        a, b = signals(full, fam, p), signals(part, fam, p)
        for x, y in zip(a, b):
            assert np.array_equal(x[:, :n], y[:, :n]), f"lookahead in {fam}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    assert_no_lookahead()
    print("no-lookahead OK", flush=True)
    Ts = {tf: load_tf(tf) for tf in BPH}
    starts = list(pd.date_range("2024-01-01", "2026-07-01", freq="QS"))
    end_ts = Ts["1h"]["idx"][-1]
    bounds = starts + [end_ts]
    rng = np.random.default_rng(a.seed)
    cfgs = [sample(rng) for _ in range(a.n)]
    score = np.full((len(cfgs), len(starts)), -np.inf)
    t0 = time.time()
    for ci, c in enumerate(cfgs):
        T = Ts[c["tf"]]
        sig = signals(T, c["fam"], c["p"])
        for wi, s in enumerate(starts):
            hi = T["idx"].searchsorted(s)
            r = sim(T, sig, 400, hi, 500.0, 450.0, c["sz"])
            yrs = (T["idx"][hi - 1] - T["idx"][400]).days / 365.25
            if r["breach"] == 0 and r["trades"] / yrs >= 40 and r["trades"] >= 50 and r["eq"] > 500:
                score[ci, wi] = np.log(r["eq"] / 500) / yrs
        if ci % 25 == 0:
            print(f"cfg {ci}/{len(cfgs)} {time.time() - t0:.0f}s", flush=True)
    eq, floor, tot_tr, wins, breach, yld, windows = 500.0, 450.0, 0, 0, 0, 0.0, []
    for wi, s in enumerate(starts):
        best = int(np.argmax(score[:, wi]))
        if not np.isfinite(score[best, wi]):
            windows.append(dict(window=str(s.date()), cfg=None)); continue
        c = cfgs[best]; T = Ts[c["tf"]]
        lo, hi = T["idx"].searchsorted(s), T["idx"].searchsorted(bounds[wi + 1]) if wi + 1 < len(starts) else len(T["idx"])
        r = sim(T, signals(T, c["fam"], c["p"]), lo, hi, eq, floor, c["sz"])
        eq, floor = r["eq"], r["floor"]; tot_tr += r["trades"]; breach += r["breach"]; yld += r["yld"]
        windows.append(dict(window=str(s.date()), cfg=c, eq=eq, trades=r["trades"], mdd=r["mdd"]))
    yrs = (end_ts - starts[0]).days / 365.25
    res = dict(seed=a.seed, n=a.n, final_mult=eq / 500, trades_per_year=tot_tr / yrs, breach_bars=breach,
               yield_share=yld / max(eq - 500, 1e-9), windows=windows)
    res["SUSPECT"] = bool(eq / 500 > 1000)  # absurd compounding = data/sim bug, never a pass`n    res["PASS"] = bool(not res["SUSPECT"] and eq / 500 >= 10 and res["trades_per_year"] >= 40 and breach == 0 and res["yield_share"] < 0.05)
    json.dump(res, open(f"data/wf_result_{a.seed}.json", "w"), indent=1, default=str)
    print({k: v for k, v in res.items() if k != "windows"})


if __name__ == "__main__":
    main()
