"""scripts/download_portfolio.py — Multi-Asset Historical Data Downloader.

Downloads continuous historical 1h and 1m candles for the portfolio universe
from Coinbase Public Exchange API and saves them to data/:
  - BTC-USD, ETH-USD, SOL-USD, AVAX-USD, DOGE-USD, LINK-USD, SUI-USD, NEAR-USD, XRP-USD.

Usage:
  python scripts/download_portfolio.py [--products BTC,ETH,SOL] [--days 730]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import os
import sys
import aiohttp
import pandas as pd
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_data import download_all

PORTFOLIO_PRODUCTS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "DOGE-USD",
    "LINK-USD", "SUI-USD", "NEAR-USD", "XRP-USD"
]


async def download_portfolio(products: list[str], output_dir: str = "data", skip_1m: bool = True) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading portfolio universe: {', '.join(products)} (skip_1m={skip_1m})")
    for prod in products:
        print(f"\n--- Fetching {prod} ---")
        try:
            await download_all(product=prod, output_dir=output_dir, skip_1m=skip_1m)
        except Exception as e:
            print(f"Error fetching {prod}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download Multi-Asset Portfolio Data")
    parser.add_argument("--products", default=None, help="Comma-separated product list")
    parser.add_argument("--output-dir", default="data", help="Output directory for parquets")
    args = parser.parse_args()

    products = args.products.split(",") if args.products else PORTFOLIO_PRODUCTS
    asyncio.run(download_portfolio(products, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
