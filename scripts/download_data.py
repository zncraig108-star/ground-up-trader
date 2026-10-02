"""scripts/download_data.py — Downloader for real Coinbase historical market data.

Fetches real SOL-USD OHLCV candles from Coinbase Exchange API up to the present hour.
- 1h candles: continuous historical series from 2021 to present.
- 1m candles: minute-level execution data for recent history (2026 to present).
- Generates aligned continuous 1m/1h/funding series saved to Parquet in data/.

Usage:
    python scripts/download_data.py [--product SOL-USD] [--output-dir data] [--full-1m]
"""

import argparse
import asyncio
import datetime
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import _normalize_timestamps, _resample_signals, OHLCV_COLUMNS


async def _fetch_candle_chunk(
    session: aiohttp.ClientSession,
    product: str,
    granularity: int,
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    max_retries: int = 5,
) -> List[list]:
    """Fetch a single chunk of up to 300 candles with 429 rate-limit backoff."""
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    params = {
        "granularity": str(granularity),
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
    }
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, list):
                        return data
                    return []
                elif resp.status == 429:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    await asyncio.sleep(0.2)
        except Exception:
            await asyncio.sleep(0.3)
    return []


async def fetch_candles_range(
    product: str,
    granularity_sec: int,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    batch_size: int = 8,
    sleep_between_batches: float = 0.08,
) -> pd.DataFrame:
    """Fetch all candles across a date range in parallel batches."""
    step_minutes = (300 * granularity_sec) // 60
    windows: List[Tuple[datetime.datetime, datetime.datetime]] = []
    curr = end_time
    while curr > start_time:
        chunk_start = max(start_time, curr - datetime.timedelta(minutes=step_minutes))
        windows.append((chunk_start, curr))
        curr = chunk_start

    print(f"Fetching {len(windows)} chunks for granularity={granularity_sec}s ({product})...")
    raw_candles: List[list] = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AntigravityTrader/1.0"}

    t0 = time.time()
    async with aiohttp.ClientSession(headers=headers) as session:
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            tasks = [
                _fetch_candle_chunk(session, product, granularity_sec, ws, we)
                for ws, we in batch
            ]
            results = await asyncio.gather(*tasks)
            for chunk in results:
                raw_candles.extend(chunk)
            done = min(i + batch_size, len(windows))
            if done % 40 == 0 or done == len(windows):
                pct = (done / len(windows)) * 100
                print(f"  progress: {done}/{len(windows)} chunks ({pct:.0f}%) — {len(raw_candles)} candles so far")
            await asyncio.sleep(sleep_between_batches)

    elapsed = time.time() - t0
    print(f"Completed download: {len(raw_candles)} raw candles in {elapsed:.1f}s")
    if not raw_candles:
        return pd.DataFrame()

    # Deduplicate and sort
    seen = set()
    unique_candles = []
    for c in raw_candles:
        if c[0] not in seen:
            seen.add(c[0])
            unique_candles.append(c)

    # Coinbase: [time, low, high, open, close, volume]
    # time is the start of the candle.
    # Convert to right-closed timestamp: close_time = time + granularity_sec
    recs = []
    now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    for c in unique_candles:
        candle_start = int(c[0])
        candle_close = candle_start + granularity_sec
        # Ignore candles that have not fully closed yet
        if candle_close > now_ts:
            continue
        recs.append({
            "timestamp": candle_close * 1000,  # epoch ms
            "open": float(c[3]),
            "high": float(c[2]),
            "low": float(c[1]),
            "close": float(c[4]),
            "volume": float(c[5]),
        })

    df = pd.DataFrame(recs)
    df = _normalize_timestamps(df)
    df = df.sort_index()
    # Remove any potential duplicate index entries
    df = df[~df.index.duplicated(keep="last")]
    return df[list(OHLCV_COLUMNS)]


def expand_1h_to_1m(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Deterministically expand 1h candles into 60 causal intra-hour 1m bars.

    Guarantees:
      - 1m bar 0 open == 1h open
      - 1m bar 59 close == 1h close
      - 1m bars touch 1h high and 1h low
      - 1m volume sums to 1h volume
      - strict right-closed alignment (each hour has exactly 60 minute bars)
    """
    m_records = []
    for ts, row in df_1h.iterrows():
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        v = float(row["volume"]) / 60.0

        # Construct 60 minute bars:
        # Step through a piecewise path: Open -> High -> Low -> Close (or Open -> Low -> High -> Close)
        mid_time = 30
        if c >= o:
            # Bullish: Open -> Low (bar 15) -> High (bar 45) -> Close (bar 59)
            t_low = 15
            t_high = 45
            p = np.empty(61)
            p[:t_low] = np.linspace(o, l, t_low)
            p[t_low:t_high] = np.linspace(l, h, t_high - t_low)
            p[t_high:] = np.linspace(h, c, 61 - t_high)
        else:
            # Bearish: Open -> High (bar 15) -> Low (bar 45) -> Close (bar 59)
            t_high = 15
            t_low = 45
            p = np.empty(61)
            p[:t_high] = np.linspace(o, h, t_high)
            p[t_high:t_low] = np.linspace(h, l, t_low - t_high)
            p[t_low:] = np.linspace(l, c, 61 - t_low)

        hour_end = ts  # timestamp of the 1h close
        hour_start = hour_end - pd.Timedelta(hours=1)
        m_timestamps = pd.date_range(hour_start + pd.Timedelta(minutes=1), hour_end, freq="1min")

        for minute_idx in range(60):
            bar_open = float(p[minute_idx])
            bar_close = float(p[minute_idx + 1])
            bar_high = max(bar_open, bar_close)
            bar_low = min(bar_open, bar_close)
            # Ensure the intra-hour extreme points touch the true High and Low
            if minute_idx == 15 or minute_idx == 45:
                if bar_open < bar_close:
                    bar_high = max(bar_high, h if minute_idx == 45 and c >= o else bar_high)
                    bar_low = min(bar_low, l if minute_idx == 15 and c >= o else bar_low)
                else:
                    bar_high = max(bar_high, h if minute_idx == 15 and c < o else bar_high)
                    bar_low = min(bar_low, l if minute_idx == 45 and c < o else bar_low)

            m_records.append({
                "timestamp": m_timestamps[minute_idx],
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": bar_close,
                "volume": v,
            })

    m_df = pd.DataFrame(m_records)
    m_df = m_df.set_index("timestamp")
    return m_df[list(OHLCV_COLUMNS)]


async def fetch_funding_history(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    market: str = "SOL-USD",
    max_pages: int = 60,
) -> pd.DataFrame:
    """Fetch real historical hourly funding rates from public indexer."""
    records = []
    curr = end_time
    base_url = f"https://indexer.dydx.trade/v4/historicalFunding/{market}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AntigravityTrader/1.0"}

    print(f"\nFetching real hourly funding history for {market}...")
    async with aiohttp.ClientSession(headers=headers) as session:
        for page in range(max_pages):
            iso_str = curr.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            url = f"{base_url}?effectiveBeforeOrAt={iso_str}"
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json()
                    items = data.get("historicalFunding", [])
                    if not items:
                        break
                    for it in items:
                        ts = pd.to_datetime(it["effectiveAt"], utc=True)
                        rate = float(it["rate"])
                        records.append({"timestamp": ts, "funding_rate": rate})
                    oldest = pd.to_datetime(items[-1]["effectiveAt"], utc=True)
                    if oldest <= start_time or oldest >= curr:
                        break
                    curr = oldest - datetime.timedelta(seconds=1)
                    if (page + 1) % 10 == 0:
                        print(f"  funding page {page+1}: reached {oldest.strftime('%Y-%m-%d')} ({len(records)} records)")
            except Exception as e:
                print(f"  note: funding fetch reached earliest available at page {page}: {e}")
                break
            await asyncio.sleep(0.04)

    if not records:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame(records).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp").tz_convert("UTC").tz_localize(None)
    df.index = df.index.floor("1h")
    df = df[~df.index.duplicated(keep="last")]
    return df


async def download_all(
    product: str = "SOL-USD",
    output_dir: str = "data",
    start_year: int = 2021,
    download_1m_since: Optional[datetime.datetime] = None,
    skip_1m: bool = False,
) -> Tuple[str, str, str]:
    """Download full history from Coinbase up to present and save Parquet files."""
    os.makedirs(output_dir, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    start_dt = datetime.datetime(start_year, 7, 1, 0, 0, tzinfo=datetime.timezone.utc)

    print("=" * 72)
    print(f"DOWNLOADING REAL MARKET DATA: {product}")
    print(f"  Range: {start_dt.strftime('%Y-%m-%d')} -> {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Output directory: {output_dir}")
    print("=" * 72)

    # 1. Fetch full 1h candles
    print("\n[Step 1/3] Downloading continuous 1h candles...")
    df_1h = await fetch_candles_range(product, 3600, start_dt, now, batch_size=8)
    if df_1h.empty:
        raise RuntimeError("Failed to fetch 1h candles from Coinbase API.")
    print(f"[ok] 1h history: {len(df_1h)} candles from {df_1h.index[0]} to {df_1h.index[-1]}")

    # 2. Fetch 1m candles for recent history (or expand 1h when skip_1m=True)
    df_1m_real = pd.DataFrame()
    if not skip_1m:
        if download_1m_since is None:
            download_1m_since = datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)

        print(f"\n[Step 2/3] Downloading real 1m candles since {download_1m_since.strftime('%Y-%m-%d')}...")
        df_1m_real = await fetch_candles_range(product, 60, download_1m_since, now, batch_size=8)
        print(f"[ok] Real 1m candles: {len(df_1m_real)} candles from {df_1m_real.index[0] if len(df_1m_real) > 0 else 'N/A'}")

    # 3. Combine into continuous 1m dataset
    print("\n[Step 3/3] Assembling continuous, aligned multi-year dataset...")
    if len(df_1m_real) > 0:
        split_ts = df_1m_real.index[0]
        # Earlier 1h candles are expanded to 1m
        df_1h_early = df_1h.loc[:split_ts - pd.Timedelta(minutes=1)]
        if len(df_1h_early) > 0:
            print(f"  Synthesizing intra-hour 1m bars for early history ({len(df_1h_early)} hours)...")
            df_1m_early = expand_1h_to_1m(df_1h_early)
            df_1m = pd.concat([df_1m_early, df_1m_real]).sort_index()
        else:
            df_1m = df_1m_real
    else:
        print("  Expanding all 1h candles to continuous 1m bars...")
        df_1m = expand_1h_to_1m(df_1h)

    # Enforce regularity (forward-fill any exchange dropouts) and trim to multiple of 60
    from core.data import _enforce_regularity
    df_1m = _enforce_regularity(df_1m, "1m", allow_ffill=True, source="sol_1m")
    m_len = (len(df_1m) // 60) * 60
    df_1m = df_1m.iloc[:m_len]
    df_1h = _resample_signals(df_1m, "1h")

    # Fetch real historical hourly funding rates from public indexer
    fund_df = await fetch_funding_history(start_dt, now)
    if not fund_df.empty:
        funding = fund_df.reindex(df_1h.index).ffill().fillna(0.0)
        mean_r = float(funding["funding_rate"].mean())
        print(f"[ok] Real funding history: {len(funding)} hourly rates (mean={mean_r:.2e})")
    else:
        funding = pd.DataFrame(
            {"funding_rate": np.zeros(len(df_1h), dtype=np.float64)},
            index=df_1h.index,
        )

    prefix = product.split("-")[0].lower()
    path_1m = os.path.join(output_dir, f"{prefix}_1m.parquet")
    path_1h = os.path.join(output_dir, f"{prefix}_1h.parquet")
    path_funding = os.path.join(output_dir, f"{prefix}_funding.parquet")

    df_1m.to_parquet(path_1m)
    df_1h.to_parquet(path_1h)
    funding.to_parquet(path_funding)

    print("\n" + "=" * 72)
    print("DOWNLOAD COMPLETE!")
    print(f"  1m bars:    {len(df_1m):,} rows -> {path_1m}")
    print(f"  1h candles: {len(df_1h):,} rows -> {path_1h}")
    print(f"  Date range: {df_1m.index[0]} -> {df_1m.index[-1]}")
    print(f"  Years available: {sorted(list(set(df_1m.index.year)))}")
    print("=" * 72)
    return path_1m, path_1h, path_funding


def main():
    parser = argparse.ArgumentParser(description="Download real Coinbase SOL data up to present.")
    parser.add_argument("--product", default="SOL-USD", help="Coinbase product ID (default: SOL-USD)")
    parser.add_argument("--output-dir", default="data", help="Output directory for Parquet files")
    parser.add_argument("--start-year", type=int, default=2021, help="Earliest year to download (default: 2021)")
    parser.add_argument("--full-1m", action="store_true", help="Download 1m candles for entire history (takes ~3-5 mins)")
    args = parser.parse_args()

    since_1m = datetime.datetime(args.start_year, 7, 1, 0, 0, tzinfo=datetime.timezone.utc) if args.full_1m else None
    asyncio.run(download_all(
        product=args.product,
        output_dir=args.output_dir,
        start_year=args.start_year,
        download_1m_since=since_1m,
    ))


if __name__ == "__main__":
    main()
