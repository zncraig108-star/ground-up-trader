"""research_freq.py - 15m-trigger / 1h-gate frequency + ladder research (variants B..E).

Inputs : data/{eth,sol,btc}_1m.parquet (real). Output: IS (2021-07..2023) vs OOS (2024+) table,
each period restarted at $500. Whole contracts, one position at a time, long+short.

Causality: bar i is decided at its CLOSE; orders fill at open of bar i+1. 1h gate values are shifted
one 1h bar (only completed hours) before being forward-filled onto the 15m grid.
Stops are checked before any favourable update inside a bar (conservative intrabar order).
"""
from __future__ import annotations
import itertools, sys
import numpy as np, pandas as pd

ASSETS = {"eth": 0.10, "sol": 1.00, "btc": 0.01}  # contract multipliers (MET, SOL, BTI)
FEE, SLIP, MARGIN, LEV_CAP = 0.0005, 0.0002, 0.33, 3.0  # ASSUMPTION: taker fee+1.4bp slip per side
YIELD = 0.05 / (8760 * 4)  # 5% APR per 15m bar on idle cash
LOCKS = [(650, 500), (1500, 1000), (3000, 2000)]  # (equity trigger, locked floor)


def features(df1m: pd.DataFrame) -> dict[str, np.ndarray]:
    """Causal features on the 15m grid from 1m OHLCV (all values known at bar close)."""
    b = df1m.resample("15min").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"}).dropna()
    c, h, l = b.close, b.high, b.low
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    rng = (h - l).replace(0, np.nan)
    delta = (b.volume * (2 * c - h - l) / rng).fillna(0.0)
    d4 = delta.rolling(4).sum()
    dz = d4 / d4.rolling(384).std()
    h1 = b.close.resample("1h").last().dropna()
    er = (h1 - h1.shift(30)).abs() / h1.diff().abs().rolling(30).sum().clip(lower=1e-9)
    ema = h1.ewm(span=100, adjust=False).mean()
    gate = pd.DataFrame({"up": ((h1 > ema) & (ema.diff(24) > 0)).astype(float),
                         "dn": ((h1 < ema) & (ema.diff(24) < 0)).astype(float), "er": er})
    gate = gate.shift(1).reindex(b.index, method="ffill")  # completed hours only
    return dict(idx=b.index, o=b.open.values, h=h.values, l=l.values, c=c.values,
                ema20=c.ewm(span=20, adjust=False).mean().values, atr=tr.rolling(14).mean().values,
                dz=dz.values, up=gate.up.values, dn=gate.dn.values, er=gate.er.values,
                ret=(c / c.shift(384) - 1).values)


def load() -> list[dict]:
    fs = [features(pd.read_parquet(f"data/{a}_1m.parquet")) for a in ASSETS]
    idx = fs[0]["idx"]
    for f in fs[1:]:
        idx = idx.intersection(f["idx"])
    out = []
    for f in fs:
        pos = f["idx"].get_indexer(idx)
        out.append({k: (idx if k == "idx" else v[pos]) for k, v in f.items()})
    return out


def assert_no_lookahead(df1m: pd.DataFrame, cut: int = 40000) -> None:
    """Features at every bar < cut must not change when all later data is removed."""
    full, part = features(df1m), features(df1m.iloc[: cut * 15])
    n = len(part["c"]) - 8  # last bars sit on a partial 15m/1h window
    for k in ("ema20", "atr", "dz", "up", "dn", "er", "ret"):
        assert np.allclose(full[k][:n], part[k][:n], equal_nan=True), f"lookahead in {k}"


def floor_for(eq: float, floor: float) -> float:
    for trig, lock in LOCKS:
        if eq >= trig:
            floor = max(floor, lock)
    return floor


def sim(S: list[dict], t0: str, t1: str, *, trigger: str = "pullback", flow: bool = False, rs: bool = False,
        ladder: bool = False, er_min: float = 0.3, k: float = 2.0, trail: float = 3.0, r_lo: float = 0.02,
        r_hi: float = 0.08) -> dict:
    idx = S[0]["idx"]
    lo, hi = idx.searchsorted(pd.Timestamp(t0)), idx.searchsorted(pd.Timestamp(t1))
    names, mults = list(ASSETS), list(ASSETS.values())
    eq = cash = 500.0
    floor, pos, pend, min_eq, breach, ntr, yld, fees = 450.0, None, None, 500.0, 0, 0, 0.0, 0.0
    wins = 0

    def close_pos(px: float) -> None:
        nonlocal cash, pos, wins, fees
        a, side, n, ent = pos["a"], pos["side"], pos["n"], pos["ent"]
        fee = n * mults[a] * px * FEE
        pnl = (px * (1 - SLIP * side) - ent) * side * n * mults[a] - fee
        cash += pnl; fees += fee; wins += pnl > 0; pos = None

    for i in range(max(lo, 400), min(hi, len(idx)) - 1):
        # 1) fill pending order at this bar's open
        if pend and pos is None:
            a, side, atr = pend
            px = S[a]["o"][i] * (1 + SLIP * side)
            stop_d = k * atr
            free = eq - floor
            budget = min((r_hi if ladder else r_lo) * eq, free / 1.5)  # gap-adjusted floor guard
            n = int(budget // (stop_d * mults[a])) if eq >= 500 else 1  # defensive: exactly 1 contract
            n = max(0, min(n, int(LEV_CAP * eq // (px * mults[a]))))
            if n >= 1:
                fee = n * mults[a] * px * FEE
                cash -= fee; fees += fee; ntr += 1
                pos = dict(a=a, side=side, n=n, ent=px, stop=px - side * stop_d, ext=px, atr=atr)
        pend = None
        # 2) stop check, then favourable updates (stop first = conservative)
        if pos:
            P, a, side = S[pos["a"]], pos["a"], pos["side"]
            if side * (P["o"][i] - pos["stop"]) <= 0:
                close_pos(P["o"][i])
            elif side * (P["l" if side > 0 else "h"][i] - pos["stop"]) <= 0:
                close_pos(pos["stop"])
            else:
                pos["ext"] = max(pos["ext"], P["h"][i]) if side > 0 else min(pos["ext"], P["l"][i])
                new = pos["ext"] - side * trail * pos["atr"]
                if side * (pos["ext"] - pos["ent"]) >= 2 * pos["atr"]:  # breakeven lock +0.05 ATR
                    new = new if side * (new - pos["ent"]) > 0 else pos["ent"] + side * 0.05 * pos["atr"]
                if side * (new - pos["stop"]) > 0:
                    pos["stop"] = new
        # 3) mark to market at close, yield, floor ratchet
        unreal, used = 0.0, 0.0
        if pos:
            c = S[pos["a"]]["c"][i]
            unreal = (c - pos["ent"]) * pos["side"] * pos["n"] * mults[pos["a"]]
            used = pos["n"] * mults[pos["a"]] * c * MARGIN
        y = max(0.0, cash + unreal - used) * YIELD
        cash += y; yld += y
        eq = cash + unreal
        floor = floor_for(eq, floor)
        min_eq = min(min_eq, eq); breach += eq < floor - 1e-9
        # 4) exit on gate flip, else look for entry at this close (acts at i+1)
        if pos:
            P = S[pos["a"]]
            if (P["up"][i] if pos["side"] > 0 else P["dn"][i]) < 1:
                close_pos(P["c"][i])  # ASSUMPTION: gate-flip exit at close price (decided from completed hour)
            continue
        if eq - floor <= 0:
            continue
        thr = er_min + (0.1 if eq < 500 else 0.0)
        cands = []
        for a, P in enumerate(S):
            if not (P["er"][i] >= thr) or not np.isfinite(P["atr"][i]):
                continue
            for side, g in ((1, "up"), (-1, "dn")):
                if side < 0 and P["er"][i] < thr + 0.05:
                    continue  # short ER boost
                if P[g][i] < 1:
                    continue
                if trigger == "pullback":
                    ok = (P["c"][i - 1] - P["ema20"][i - 1]) * side < 0 and (P["c"][i] - P["ema20"][i]) * side > 0
                else:  # 'orb': 4h range break (bars i-16..i-1), completed bars only
                    ok = side * (P["c"][i] - (P["h"][i - 16:i].max() if side > 0 else P["l"][i - 16:i].min())) > 0
                if flow and not side * P["dz"][i] > 0:
                    ok = False
                if rs:
                    rets = [Q["ret"][i] for Q in S]
                    ok = ok and P["ret"][i] == (max(rets) if side > 0 else min(rets))
                if ok:
                    cands.append((P["er"][i], a, side, P["atr"][i]))
        if cands:
            _, a, side, atr = max(cands)
            pend = (a, side, atr)
    yrs = (idx[min(hi, len(idx)) - 1] - idx[max(lo, 400)]).days / 365.25
    return dict(mult=eq / 500, trades=ntr, tpy=ntr / yrs, win=wins / max(ntr, 1), min_eq=min_eq,
                breach=breach, yield_usd=yld)


VARIANTS = {"B pullback": {}, "C +flow": dict(flow=True), "D +relstr": dict(flow=True, rs=True),
            "D' relstr only": dict(rs=True), "E +ladder": dict(ladder=True),
            "E full": dict(flow=True, rs=True, ladder=True), "ORB+ladder": dict(trigger="orb", ladder=True)}
GRID = dict(k=[1.5, 2.5], er_min=[0.25, 0.35], r_hi=[0.05, 0.10])


def main() -> None:
    assert_no_lookahead(pd.read_parquet("data/sol_1m.parquet"))
    print("no-lookahead check OK")
    S = load()
    IS, OOS = ("2021-07-01", "2024-01-01"), ("2024-01-01", "2027-01-01")
    for name, kw in VARIANTS.items():
        best = None
        for k, er, r in itertools.product(GRID["k"], GRID["er_min"], GRID["r_hi"]):  # tuned on IS only
            res = sim(S, *IS, **kw, k=k, er_min=er, r_hi=r)
            if best is None or res["mult"] > best[0]["mult"]:
                best = (res, dict(k=k, er_min=er, r_hi=r))
        oos = sim(S, *OOS, **kw, **best[1])
        f = lambda d: (f"{d['mult']:6.2f}x trades/yr {d['tpy']:5.1f} win {d['win']:.0%} minEq {d['min_eq']:6.1f} "
                       f"breach {d['breach']} yield ${d['yield_usd']:.0f}")
        print(f"{name:15s} {best[1]}\n   IS : {f(best[0])}\n   OOS: {f(oos)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
