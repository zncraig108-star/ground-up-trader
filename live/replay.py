"""live/replay.py — executor-vs-backtester replay parity harness.

Feeds a historical 1m series through the **real** ``Executor`` + ``PaperAdapter`` path
(the same code that would trade live) and compares its fills with the batched
backtester on the same data and the same ``FillModelConfig``.

Timing ASSUMPTION: the backtester places a resting entry at the first 1m bar of hour
``h`` (the hour boundary) and may fill within that same bar; the executor observes the
whole hour before it can act, so its order becomes fillable one bar later. A 1-bar
placement offset is therefore expected and the parity check tolerates it; everything
else (stale-order cancellation, stop slippage, intrabar liquidation, funding) should
match structurally. Use this harness to monitor live-vs-backtest drift, not as a
bit-exact equality assertion.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import InstrumentSpec
from core.backtester import backtest
from core.strategy import Genome, SignalFeatures, compute_signals
from core.xp import asnumpy
from live.exchange import PaperAdapter, candles_from_dataframe
from live.executor import Executor

__all__ = ["ReplayResult", "replay_executor", "backtest_reference", "compare_replay"]


@dataclass(frozen=True)
class ReplayResult:
    """Fill summary used to compare the live path against the backtester."""

    entry_fills: int
    exit_fills: int
    liquidations: int
    final_equity: float


async def replay_executor(config: Any, instrument: InstrumentSpec, genome: Genome,
                          ohlcv_1m: pd.DataFrame) -> ReplayResult:
    """Run the live executor over a historical 1m series (paper fills, no network)."""
    adapter = PaperAdapter(instrument, initial_equity=config.risk.initial_equity,
                           feed=candles_from_dataframe(ohlcv_1m),
                           fill_model=config.fills)
    executor = Executor(config, instrument, adapter)
    await executor.deploy(genome)
    await executor.run()
    return ReplayResult(entry_fills=adapter.entry_fills, exit_fills=adapter.exit_fills,
                        liquidations=adapter.liquidations,
                        final_equity=float(await adapter.get_equity()))


def backtest_reference(config: Any, instrument: InstrumentSpec, genome: Genome,
                       ohlcv_1m: pd.DataFrame, ohlcv_1h: pd.DataFrame,
                       funding: pd.DataFrame) -> ReplayResult:
    """Backtest the same genome with the same fill model; entries are maker fills that
    are not take-profit exits."""
    features = SignalFeatures(open=ohlcv_1h["open"].to_numpy(),
                              high=ohlcv_1h["high"].to_numpy(),
                              low=ohlcv_1h["low"].to_numpy(),
                              close=ohlcv_1h["close"].to_numpy())
    batch = np.asarray([genome.encode()], dtype=np.float64)
    signals = compute_signals(features, batch)
    res = backtest(ohlcv_1m, ohlcv_1h, funding, signals, batch, instrument,
                   initial_equity=config.risk.initial_equity, fill_model=config.fills,
                   use_numba=False)
    maker = int(asnumpy(res.maker_fills)[0])
    tp = sum(1 for t in res.trades[0] if t.exit_reason == "tp")
    return ReplayResult(entry_fills=maker - tp,
                        exit_fills=int(asnumpy(res.taker_exits)[0]) + tp,
                        liquidations=int(asnumpy(res.liquidations)[0]),
                        final_equity=float(asnumpy(res.final_equity)[0]))


def compare_replay(config: Any, instrument: InstrumentSpec, genome: Genome,
                   ohlcv_1m: pd.DataFrame, ohlcv_1h: pd.DataFrame,
                   funding: pd.DataFrame) -> Dict[str, Any]:
    """Run both paths and return a drift report (deterministic)."""
    replay = asyncio.run(replay_executor(config, instrument, genome, ohlcv_1m))
    bt = backtest_reference(config, instrument, genome, ohlcv_1m, ohlcv_1h, funding)
    return {
        "backtest": bt,
        "replay": replay,
        "entry_delta": replay.entry_fills - bt.entry_fills,
        "exit_delta": replay.exit_fills - bt.exit_fills,
        "liq_delta": replay.liquidations - bt.liquidations,
        "equity_delta": replay.final_equity - bt.final_equity,
    }
