# AGENTS.md

Operating guide for the AI agent building the **SOL-PERP Evolutionary Trading System**.
Read this fully before writing code. The functional spec lives in
`DEEPSEEK_BUILD_PROMPT.md`; this file governs *how* you work.

---

## 1. Mission

Build a modular, GPU-accelerated, strictly zero-lookahead algorithmic trading system:
backtester, GA + walk-forward optimizer, real-time web UI, and live/paper async
executor for Coinbase Advanced Trade SOL-PERP. Correctness beats cleverness.

## 2. Golden Rules (do not violate)

1. **No lookahead, ever.** A decision at time `T` uses only data `<= T`; resulting
   orders act at `T+1` or later. Indicators use only fully-closed candles. When unsure,
   pick the more conservative (no-leak) interpretation and add `# ASSUMPTION:`.
2. **Determinism.** Every stochastic path takes an explicit `seed`. Same seed + config
   ⇒ identical results.
3. **Graceful degradation.** Code must import and the demo must run with **no GPU and no
   network**. GPU (CuPy/Numba-CUDA) and Coinbase API are optional accelerators/adapters.
4. **No secrets in code.** API keys come from environment variables only. Provide
   `.env.example`, never real keys. Default to **paper mode**.
5. **Money math is exact.** Round to `tick_size`/`lot_size` before any fill comparison.
   Post-only fills require price crossing *through* the limit by ≥ 1 tick. Never compare
   raw floats for fills.
6. **No NaN/inf fitness.** Sanitize/clamp; a bad strategy scores `0`, never a
   misleadingly positive value.
7. **Interfaces before implementations.** Finalize a module's public API before writing
   code that depends on it. Don't invent APIs you haven't defined.

## 3. Tech Constraints

- Python **3.11+**, full type hints, docstrings on public symbols.
- Array backend via `core/xp.py`: **CuPy** if CUDA present else **NumPy**; identical
  results (within float tolerance) on both. Use **Numba** for reductions that don't
  vectorize cleanly.
- Allowed deps only: `numpy, pandas, pyarrow, cupy-cuda12x (optional), numba, cdp-sdk,
  aiohttp, websockets, fastapi, uvicorn, python-dotenv`. Frontend uses CDN
  Tailwind + Chart.js only — **no npm/build step**.
- Async I/O for `live/` and `ui/`; the GA/backtester are CPU/GPU compute, not async.

## 4. Architecture Boundaries

- Sub-systems are decoupled and communicate via typed dataclasses / JSON snapshots, not
  shared globals.
- Backtester is **batched over population `P`, sequential over the 1m fill axis `N`**.
  Vectorize across genomes; step time forward one bar at a time.
- Dual timeframe: **1h signals (default, configurable)** + **1m fills/funding/margin**.
- `config.py` is the single source of truth for all tunables, ranges, and instrument
  spec.

## 5. Workflow (phased — see spec §11)

Build in order and verify each phase before advancing:
Foundations → Math → Simulation → Evolution → UI → Live/Paper → Demo & wiring.
After each phase: ensure imports succeed and run the phase's mini-check. Never mark a
phase done on intent — only after its check actually passes.

## 6. Verification (this project uses a demo, not a formal test suite)

- Provide `run_demo.py` that exercises the **entire** pipeline on synthetic data end to
  end with one command, no network, GPU optional.
- Embed lightweight inline checks (asserts) for the critical invariants:
  - causality shift (`assert_no_lookahead`),
  - post-only cross-through-by-1-tick fill logic,
  - fee + hourly funding deduction signs,
  - maintenance-margin liquidation,
  - fitness never NaN/inf and `0` for negative-Sharpe strategies,
  - GA best-fitness is non-decreasing across a few generations (elitism),
  - GPU vs CPU parity when a GPU is present.
- If a check fails, fix the root cause and re-run — do not weaken the check to pass.

## 7. Coding Standards

- Descriptive names; small, single-purpose functions; module-level docstring explaining
  the file's role and its inputs/outputs.
- Comment the *why* (especially causality shifts, fee/funding signs, margin math), not
  the obvious *what*.
- Mark inferred external details with `# VERIFY vs live Coinbase/CDP spec`; mark chosen
  interpretations of ambiguity with `# ASSUMPTION:`.
- No dead code, no unused deps, no commented-out blocks left behind.
- Keep functions GPU/CPU-agnostic by using `xp` rather than importing numpy/cupy
  directly in kernels.

## 8. Definition of Done

- All files in spec §1 implemented, importable in isolation, and wired together.
- `run_demo.py` runs the full flow on synthetic data and serves the UI.
- No lookahead, deterministic under seed, no NaN/inf fitness, paper mode by default,
  no secrets in code, GPU/network optional.
- `requirements.txt` finalized with `cupy` marked optional and a working NumPy fallback.
- Assumptions and unverified external details are clearly annotated in comments.

## 9. When Blocked

If an instruction is contradictory or an external API detail is unknown: choose the
safest zero-lookahead / paper-mode-default interpretation, implement behind the relevant
interface (`ExchangeAdapter`, data loader), leave a clear `# ASSUMPTION:` / `# VERIFY`
note, and keep going. Do not silently guess; do not stall.

## 10. Git Commit & Push Workflow

Whenever finishing a workflow, task, or user request:
1. **Branching on Major Changes**: If a task involves a major refactor, architectural pivot, experimental exploration, or significant change in direction, create and work on a dedicated feature/experiment branch (e.g., `feat/<name>` or `exp/<name>`). Once verified and passing all checks, merge back into `main` (or the parent branch) as appropriate.
2. Stage all modified and relevant new files (`git add`).
3. Create a clean, descriptive git commit summarizing the changes.
4. Push the commit (and any active branch or merged `main`) to the remote repository (`git push`).

