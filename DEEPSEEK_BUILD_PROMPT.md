# BUILD PROMPT — SOL-PERP Evolutionary Trading System

## 0. ROLE & PRIME DIRECTIVE

You are an expert quantitative systems engineer specializing in low-latency async
Python, **strict zero-lookahead backtesting**, GPU-accelerated numerical computing,
and automated evolutionary optimization.

Build a production-grade, highly modular algorithmic trading system for **Coinbase
Advanced Trade perpetual futures**, target asset **SOL-PERP**. Runs on a Linux/Windows
**PC with an NVIDIA CUDA GPU** (not a Raspberry Pi).

**Prime directive: correctness and zero lookahead bias over cleverness.** If any
instruction is ambiguous, choose the interpretation that most rigorously prevents
future information from leaking into a decision, and leave a `# ASSUMPTION:` comment
explaining the choice.

**Non-negotiable global rules**
- Python 3.11+. Full type hints. Docstrings on every public class/function.
- No lookahead anywhere. A decision made at time `T` may use data timestamped `<= T`
  only, and orders resulting from it may only act at `T+1` or later.
- All money math uses explicit tick/lot rounding and integer-safe rounding helpers.
  Never compare raw floats for fills; round to the instrument tick size first.
- Every module must be runnable/importable in isolation and must not crash on import
  if the GPU or network is absent (degrade gracefully; see fallbacks below).
- Deterministic: every stochastic component (GA, data synth) takes an explicit
  `seed`. Same seed + same config = identical run.

---

## 1. SYSTEM OVERVIEW

Four decoupled sub-systems communicating through plain dataclasses / typed dicts and
a shared config. No hidden global state.

```
config.py                 # single source of truth for all tunables + instrument spec
core/
  data.py                 # data loader interface + Coinbase fetcher + synthetic gen
  indicators.py           # GPU/CPU indicator kernels (EMA, ATR, Supertrend, etc.)
  strategy.py             # genome -> signals (the concrete strategy template)
  backtester.py           # GPU-vectorized population backtest, zero-lookahead
  metrics.py              # Sharpe/Sortino/DD/WFE + robust fitness function
  evolution.py            # GA + Walk-Forward Optimization loop
ui/
  server.py               # FastAPI + WebSocket dashboard host
  index.html              # single-file Tailwind + Chart.js frontend
live/
  exchange.py             # ExchangeAdapter interface + Coinbase CDP implementation
  executor.py             # live/paper async execution engine
run_demo.py               # end-to-end runnable demo (synthetic data -> GA -> UI)
requirements.txt
AGENTS.md                 # (already provided — follow it)
```

Data flow:
`data.py -> indicators.py -> strategy.py -> backtester.py -> metrics.py -> evolution.py`
then `evolution.py <-> ui/server.py` (live broadcast) and
`ui/server.py -> live/executor.py` (deploy a winning genome to paper trading).

---

## 2. COMPUTE ARCHITECTURE (GPU-VECTORIZED POPULATION BACKTEST)

The GA evaluates hundreds/thousands of genomes per generation. To make this fast:

- **Backend abstraction.** Create `core/xp.py` exposing an array namespace `xp` that
  is **CuPy** when a CUDA GPU is available and **NumPy** otherwise. All numeric kernels
  import from here so the exact same code runs on GPU or CPU. Detect once at startup,
  log which backend is active. Custom reductions may use **Numba** (`@numba.njit` for
  CPU, `numba.cuda` kernels for GPU) where a vectorized form is impractical.
- **Population batching.** The backtester evaluates an entire population in **one
  batched pass**: arrays are shaped `(P, N)` where `P` = population size and `N` =
  number of bars. Indicators, signals, positions, equity, fees, and funding are all
  computed as batched array ops across the `P` axis. This is the core performance
  design — write the sim as vectorized state updates over the time axis, broadcasting
  across the population axis.
- **Sequential-in-time, parallel-in-population.** The time axis `N` is stepped
  sequentially (a Python/Numba loop over bars is acceptable and correct) because
  position/margin state is path-dependent. The population axis `P` is fully
  vectorized. Do **not** vectorize across time in a way that leaks future bars.
- **CPU fallback must produce identical results** (within floating tolerance) to the
  GPU path. Provide a pytest-free smoke check in `run_demo.py` comparing a tiny run
  on both backends when a GPU is present.

---

## 3. INSTRUMENT & DATA MODEL

### 3.1 Instrument spec (in `config.py`)
Define a frozen dataclass `InstrumentSpec` for SOL-PERP with: `product_id="SOL-PERP"`,
`tick_size`, `lot_size` (contract/step size), `min_notional`, `maker_fee_bps`,
`taker_fee_bps`, `funding_interval_hours=1`, `max_leverage=2.0`,
`maintenance_margin_rate`. Provide sensible documented defaults and mark them
`# VERIFY vs live Coinbase spec` — treat Coinbase Advanced Trade as offering SOL-PERP
with hourly funding as stated; do not block on verification.

### 3.2 Data timeframes
- **Signal timeframe:** configurable, **default `1h`** (`SIGNAL_TIMEFRAME`). Indicators
  and entry decisions use only *fully closed* signal candles.
- **Fill/simulation timeframe:** **`1m`** (`FILL_TIMEFRAME`). Post-only fill detection,
  intrabar stop/TP triggering, funding settlement, and margin/liquidation checks run on
  the fine 1m series. This dual-timeframe design is required for realistic fills.
- The loader must align the two: each 1h signal candle maps to the 60 subsequent 1m
  candles. A signal from the 1h candle closing at `T` may only act on 1m candles with
  timestamp `> T`.

### 3.3 Data loader (`core/data.py`)
- `load_ohlcv(path, timeframe) -> DataFrame` reading **Parquet or CSV** with a strict,
  documented schema: `timestamp` (UTC, ms or ISO), `open, high, low, close, volume`.
  Validate monotonic increasing timestamps, no duplicates, no gaps (or forward-fill
  gaps with an explicit flag and log a warning).
- `load_funding(path) -> DataFrame` with `timestamp, funding_rate` (per-interval rate).
- `fetch_coinbase(...)` optional REST fetcher that pulls historical 1h + 1m candles
  and funding, caches to Parquet on disk. Network optional — never required to run the
  demo.
- `make_synthetic(seed, days, ...)` generates realistic correlated 1m + 1h OHLCV and a
  funding series (GBM with regime shifts + intrabar noise) so the **entire system runs
  with zero external dependencies**. The demo uses this.

---

## 4. STRATEGY TEMPLATE (`core/strategy.py`)

A single concrete, well-defined strategy family that the genome parameterizes. This
removes ambiguity — implement exactly this:

**Regime filter (discrete toggle `regime_mode`):**
- `EMA`: long-regime if `EMA(close, ema_fast) > EMA(close, ema_slow)` and rising;
  short-regime if the inverse.
- `SUPERTREND`: long-regime if Supertrend(period `st_period`, mult `st_mult`) is bullish;
  short-regime if bearish.

**Entry (post-only maker):**
- Only take entries in the direction of the current regime.
- On the close of signal candle `T`, if regime is long and price pulled back by
  `entry_atr_mult * ATR(atr_period)` below the close (mean-reversion-in-trend), place a
  **post-only limit BUY** at `close - entry_offset_atr_mult * ATR` for the `T+1` window.
  Symmetric for shorts. (This is a trend-filter + ATR pullback breakout template — keep
  it internally consistent; document the exact rule you implement.)

**Exits:**
- Take-profit: limit (maker) at `entry_price ± tp_atr_mult * ATR`.
- Stop-loss: **stop-market (taker)** at `entry_price ∓ sl_atr_mult * ATR`.
- Optional time stop: close after `max_hold_bars` signal candles.

**Genome (continuous unless noted):**
`ema_fast, ema_slow, st_period, st_mult, atr_period, entry_offset_atr_mult,
tp_atr_mult, sl_atr_mult, max_hold_bars, target_leverage (<=2.0),
regime_mode (discrete: EMA|SUPERTREND)`. Each param has a documented `(min, max)` range
in `config.py`. Provide `Genome` dataclass + `encode()/decode()` to/from a flat array
for vectorized ops, and `random_genome(rng)` / `clamp(genome)`.

`strategy.compute_signals(features, genome_batch) -> intents` must be fully vectorized
over the population axis and must consume only closed-candle features.

---

## 5. MODULE — BACKTESTER (`core/backtester.py`): STRICT REAL-WORLD EMULATION

Simulate live Coinbase Advanced Trade mechanics exactly. Batched over population `P`,
sequential over 1m fill bars `N`.

- **Temporal causality (enforced structurally):** compute all indicators on closed 1h
  candles, then *shift by one* so bar `T`'s feature vector is only visible for decisions
  acting at `T+1`. Add an assertion/guard helper `assert_no_lookahead()` used in tests
  and the demo.
- **Post-only maker fills:** a resting post-only limit fills **only if the 1m market
  price crosses *through* the limit by at least 1 tick** (i.e., for a buy limit at `L`,
  fill only when `low <= L - tick_size`; not merely `low <= L`). If a post-only order
  would cross the spread at placement, it is **rejected** (post-only would take
  liquidity) — model this rejection.
- **Cost accounting:** deduct **maker fee** on maker fills, **taker fee** on
  stop-market exits, and settle **hourly funding** at each funding timestamp on open
  positions (`funding_payment = position_notional * funding_rate`, sign-correct for
  long/short). Track fees and funding separately in the equity attribution.
- **Margin mechanics:** cap position notional at `target_leverage <= 2.0x` of available
  equity. Enforce a **maintenance-margin/liquidation check every 1m bar**: if equity <
  maintenance margin, force liquidation as a taker market close and record it.
- **Outputs per genome:** equity curve (both 1m and resampled), trade log, counts of
  maker fills / taker exits / liquidations / post-only rejections, and the aggregate
  stats needed by `metrics.py`. Return arrays shaped `(P, ...)`.

---

## 6. MODULE — METRICS & FITNESS (`core/metrics.py`)

Implement, all vectorized over `P`, with explicit, documented conventions:

- **Sharpe:** `mean(returns)/std(returns) * sqrt(periods_per_year)`. Define
  `periods_per_year` from the return sampling frequency. `std==0 -> Sharpe=0`.
- **Sortino:** same but denominator uses downside deviation only; no downside -> use a
  large-but-finite cap, not `inf`.
- **Max Drawdown:** as a fraction in `[0, 1]` from the running peak of the equity curve.
- **Walk-Forward Efficiency (WFE):** `oos_metric / is_metric`, clamped to `[0, 1.5]`
  (define which metric — use annualized return or Sharpe; document it).

**Robust fitness (fixes the naive formula):**
```
sharpe_pos  = max(sharpe, 0.0)
sortino_pos = max(sortino, 0.0)
dd_factor   = clip(1.0 - max_drawdown, 0.0, 1.0)     # never negative
wfe_factor  = clip(walk_forward_efficiency, 0.0, 1.5)
raw         = sqrt(sharpe_pos * sortino_pos) * dd_factor * wfe_factor  # geometric, sign-safe

# smooth trade-count penalty (NOT a hard cliff), MIN_TRADES default 50:
trade_factor = min(1.0, trades / MIN_TRADES) ** TRADE_PENALTY_POWER   # e.g. power=1.5
fitness      = raw * trade_factor
```
- Never produce `NaN`/`inf`: sanitize all inputs; a strategy with negative Sharpe or
  Sortino scores `0`, not a misleadingly positive product. Document why `sqrt` +
  clamping fixes the sign-flip and DD>100% bugs of the original spec.

---

## 7. MODULE — EVOLUTIONARY ENGINE (`core/evolution.py`): GA + WFO

- **Walk-Forward windows:** sliding **60-day in-sample** train + **14-day out-of-sample**
  validate, both configurable. The GA optimizes fitness on IS; genomes are then scored
  on the untouched OOS window.
- **Genetic operators:**
  - **Tournament selection** (configurable tournament size).
  - **Uniform crossover** on the genome vector.
  - **Gaussian mutation** for continuous genes (per-gene sigma as a fraction of the
    gene's range), **bit-flip** for the discrete `regime_mode`.
  - **Elitism:** top **5%** of genomes carried over unchanged each generation.
  - Always `clamp()` offspring back into valid ranges.
- **Deployment gate:** a genome graduates to the **Deployment Pool** only if its OOS
  Sharpe is **>= 70% of its IS Sharpe** (retention threshold, configurable) AND it meets
  the min-trades requirement on OOS. Persist the pool to disk (JSON).
- **Continuous loop:** runs generation after generation, emitting a state snapshot after
  each generation (see §8 payload) via an async queue/callback so the UI can broadcast
  it. Must be cancellable cleanly.
- **Checkpointing:** save/restore full GA state (population, generation, RNG state,
  pool) to disk so a run can resume.

---

## 8. MODULE — WEB UI (`ui/server.py` + `ui/index.html`)

- **Backend:** FastAPI. Serve `index.html` at `/`. WebSocket endpoint **`/ws/evolution`**
  broadcasting a JSON snapshot after each generation:
  ```json
  {
    "generation": int,
    "best_fitness": float,
    "wfo_window": {"is_start": "...", "is_end": "...", "oos_start": "...", "oos_end": "..."},
    "is_equity_curve": [[t, v], ...],
    "oos_equity_curve": [[t, v], ...],
    "top_genomes": [{"rank": int, "params": {...}, "is_sharpe": f, "oos_sharpe": f, "fitness": f, "trades": int}],
    "deployment_pool_size": int,
    "backend": "cupy|numpy"
  }
  ```
- REST: `GET /api/state` (latest snapshot), `GET /api/pool` (deployment pool),
  `POST /api/deploy` (body: genome id) to push a winning genome to the paper executor.
- **Frontend (single `index.html`):** CDN TailwindCSS + Chart.js only. Show: generation
  counter + backend badge, a line chart overlaying **In-Sample vs Out-of-Sample equity
  curves**, a **top-genomes parameter table**, deployment-pool list, and a **"Deploy to
  Paper"** button per pool genome that calls `POST /api/deploy`. Auto-reconnect the
  WebSocket with backoff. No build step, no npm.

---

## 9. MODULE — LIVE ASYNC EXECUTOR (`live/exchange.py` + `live/executor.py`)

- **Adapter seam:** define `ExchangeAdapter` (abstract) with async methods:
  `connect()`, `stream_trades()/stream_candles()`, `place_post_only_limit()`,
  `place_stop_market()`, `cancel_all()`, `market_close_all()`, `get_equity()`,
  `get_positions()`. Provide `CoinbaseCDPAdapter` using the official **`cdp-sdk`**,
  `aiohttp`, and `websockets`, and a `PaperAdapter` that simulates fills against the
  live price stream (used by the "Deploy to Paper" button).
- **Resiliency:** WebSocket wrapped in **exponential-backoff auto-reconnect** with jitter
  and a max cap; resubscribe on reconnect; heartbeat/staleness watchdog.
- **Risk guardrails (hard):**
  - **Circuit breaker:** if unrealized drawdown reaches **6% of total equity within a
    rolling 24h window**, immediately `cancel_all()`, `market_close_all()`, and **halt**
    (set a tripped flag that blocks new orders until manual reset).
  - **Dynamic sizing:** `contracts = floor_to_lot((equity * target_leverage) / price)`
    with `target_leverage <= 2.0`, respecting `min_notional` and `lot_size`.
- The executor consumes a deployed `Genome`, runs the §4 strategy live, routes
  post-only entries + maker TP + stop-market SL, and defaults to **paper mode**. Live
  mode must require an explicit config flag + env-provided API keys; never hardcode keys.

---

## 10. CONFIG (`config.py`)

Single source of truth. Frozen/typed dataclasses (or `pydantic`), grouped:
`InstrumentSpec`, `DataConfig` (paths, timeframes, seeds), `GAConfig` (pop size,
generations, elitism %, mutation sigma, tournament size, MIN_TRADES,
TRADE_PENALTY_POWER, retention threshold), `WFOConfig` (IS/OOS days), `RiskConfig`
(leverage, DD breaker %), `UIConfig` (host/port), `LiveConfig` (paper/live flag, env key
names). Every genome gene's `(min, max)` lives here. Include a `.env.example` reference
for keys; load secrets from env only.

---

## 11. PHASED BUILD PLAN (build & self-verify in this order)

Build incrementally. After each phase, ensure the code imports and the phase's mini-check
passes before moving on. Do not write later modules against imagined APIs — finalize each
interface first.

1. **Phase 1 — Foundations:** `config.py`, `core/xp.py` (GPU/CPU backend + logging),
   `core/data.py` (loader + `make_synthetic`). Check: synthesize data, print shape/head.
2. **Phase 2 — Math:** `core/indicators.py`, `core/strategy.py` (Genome + vectorized
   signals). Check: signals computed on synthetic data, assert no NaNs, assert
   causality shift.
3. **Phase 3 — Simulation:** `core/backtester.py` + `core/metrics.py`. Check: run a small
   population on synthetic data; verify post-only "cross-through-by-1-tick" fill logic,
   fee/funding deduction, and liquidation on a hand-crafted adversarial series. Verify
   GPU vs CPU parity if a GPU is present.
4. **Phase 4 — Evolution:** `core/evolution.py` (GA + WFO + deployment gate +
   checkpointing). Check: short 5-generation run improves best fitness and populates a
   deployment pool.
5. **Phase 5 — UI:** `ui/server.py` + `ui/index.html`. Check: server starts, WebSocket
   streams generation snapshots, charts render, deploy button hits the API.
6. **Phase 6 — Live/Paper:** `live/exchange.py` + `live/executor.py`. Check: PaperAdapter
   runs the strategy against a simulated live stream; circuit breaker trips on a forced
   6% drawdown; sizing respects 2.0x cap.
7. **Phase 7 — Demo & wiring:** `run_demo.py` runs the full pipeline end-to-end on
   synthetic data (GA for a few generations → launches UI → deploy a genome to paper) with
   **no network or GPU required**, GPU auto-used if present. `requirements.txt` finalized.

---

## 12. DELIVERABLES & QUALITY BAR

- Complete, runnable implementations of every file in §1, thoroughly commented, Python
  3.11+, full type hints.
- **`run_demo.py`** must launch the whole system on synthetic data with a single command
  and print/serve visible results (no external API, no GPU strictly required).
- **`requirements.txt`** — lightweight, CUDA-capable stack:
  `numpy, pandas, pyarrow, cupy-cuda12x (optional/extra), numba, cdp-sdk, aiohttp,
  websockets, fastapi, uvicorn, python-dotenv`. Clearly mark `cupy` as an optional GPU
  extra with a NumPy fallback so install never hard-fails without CUDA.
- No lookahead, no NaN/inf fitness, deterministic under seed, graceful GPU/network
  absence. Leave `# ASSUMPTION:` / `# VERIFY` comments wherever you infer a Coinbase or
  SDK detail rather than guessing silently.
- Follow **AGENTS.md** for workflow, style, and verification conventions.
