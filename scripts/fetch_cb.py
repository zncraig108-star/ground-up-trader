"""fetch_cb.py - download REAL 15m candles (Coinbase Exchange public API) to data/{asset}_15m.parquet.

Why: data/*_1m.parquet is linearly interpolated from hourly bars (90% of 1m diffs identical, ~2 unique volumes
per hour), which leaks each hour's close into earlier sub-hour bars. Spot USD pairs are the proxy for the
regulated futures. # ASSUMPTION: spot candles ~ futures price path. # VERIFY vs live Coinbase spec.
"""
import json, sys, time, urllib.request
from datetime import datetime, timedelta, timezone
import pandas as pd

START, STEP = datetime(2021, 6, 30, tzinfo=timezone.utc), timedelta(seconds=900 * 300)


def get(url: str) -> list:
    for t in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gut"})
            return json.load(urllib.request.urlopen(req, timeout=30))
        except Exception:
            time.sleep(1 + t)
    raise RuntimeError(url)


def fetch(asset: str) -> pd.DataFrame:
    rows, t, now = [], START, datetime.now(timezone.utc)
    while t < now:
        e = min(t + STEP, now)
        url = (f"https://api.exchange.coinbase.com/products/{asset.upper()}-USD/candles?granularity=900"
               f"&start={t.strftime('%Y-%m-%dT%H:%M:%SZ')}&end={e.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        rows += get(url); t = e; time.sleep(0.15)
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df.t, unit="s")
    return df.drop(columns="t").drop_duplicates("timestamp").set_index("timestamp").sort_index()[
        ["open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    for a in sys.argv[1:] or ["eth", "sol", "btc"]:
        d = fetch(a); d.to_parquet(f"data/{a}_15m.parquet"); print(a, len(d), d.index[0], d.index[-1], flush=True)
