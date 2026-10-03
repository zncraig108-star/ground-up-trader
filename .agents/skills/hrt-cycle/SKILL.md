---
name: hrt-cycle
description: >-
  Systematic Hypothesis-Refactor-Test (HRT) framework for continuous autonomous improvement.
  Use this skill whenever tasked with optimizing, debugging, improving performance, refactoring
  complex systems, trading algorithms, or models where empirical gains must be proven without
  regressions or curve-fitting.
---

# Hypothesis-Refactor-Test (HRT) Continuous Improvement Framework

The **HRT Cycle** is an empirical, scientific workflow designed for autonomous AI agents to continuously improve codebases, algorithms, and systems. It prevents the two most common failure modes of AI-driven optimization:
1. **Shotgun debugging & parameter churn** (random changes without a causal theory).
2. **Curve-fitting & regression** (improving a benchmark by overfitting or breaking invariants).

---

## The 4-Phase Continuous Improvement Loop

```
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. DIAGNOSE & HYPOTHESIZE                                   │
  │    - Isolate physical/logical root cause                    │
  │    - Define falsifiable hypothesis + invariant constraint   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. MINIMAL REFACTOR (/ponytail ultra)                       │
  │    - Isolate to a single causal variable                    │
  │    - Use language primitives / stdlib; zero bloat           │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. MULTI-TIER TESTING & FALSIFICATION                       │
  │    - Tier 1: Invariant & Causality Checks                   │
  │    - Tier 2: In-Sample Benchmark Metric                     │
  │    - Tier 3: Out-of-Sample / Stress Validation              │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
       [Verdict: VERIFIED]              [Verdict: FALSIFIED]
                 │                               │
                 ▼                               ▼
  ┌──────────────────────────────┐┌──────────────────────────────┐
  │ 4A. COMMIT & LEDGER          ││ 4B. CLEAN REVERT             │
  │     - Commit diff with proof ││     - git restore            │
  │     - Record in ledger       ││     - Log why it failed      │
  └──────────────────────────────┘└──────────────────────────────┘
```

---

## Phase 1: Diagnose & Hypothesize

Before modifying any file, you must state a rigorous, falsifiable hypothesis using this format:

```markdown
### Hypothesis [H-00X]: <Descriptive Title>
- **Empirical Observation**: What specific metric, failure log, or behavior indicates underperformance?
- **Root Cause Mechanism**: What physical, mathematical, or mechanical dynamic is causing this?
- **Proposed Intervention**: What is the single, minimal code/system modification?
- **Target Metric**: What quantifiable metric must improve (and by what threshold)?
- **Invariant Constraints**: What essential properties MUST NOT degrade? (e.g., zero lookahead, API signature, fee budget, test suite).
```

### Golden Rules of Hypotheses:
1. **Single-Variable Rule**: Test ONE causal mechanism per iteration. Never bundle 3 unrelated changes together, or you cannot attribute the cause of the result.
2. **Falsifiability**: The test must be capable of showing that the hypothesis is WRONG.
3. **Occam's Mechanism**: Seek the simplest structural explanation before suspecting obscure bugs.

---

## Phase 2: Minimal Refactor (`/ponytail ultra`)

Implement the hypothesis with the smallest possible footprint:
1. **Standard Library & Primitives First**: Reach for language built-ins, standard libraries, or established project utilities before introducing new libraries, classes, or configuration knobs.
2. **Preserve Public Interfaces**: Do not alter external function signatures, data types, or return shapes unless the interface itself is the explicit subject of the hypothesis.
3. **No Zombie Code**: Never leave commented-out blocks, legacy fallbacks, or unused parameters behind. If it is replaced, delete it.
4. **Annotate Rationale**: Add `# ASSUMPTION:` or `# MECHANISM:` comments explaining *why* the specific math or logic was changed.

---

## Phase 3: Multi-Tier Testing & Falsification

A change is only accepted if it passes all three verification tiers:

### Tier 1: Invariant & Causality Verification
- **Conservation Laws**: Did money, tokens, arrays, or states balance exactly?
- **Zero-Lookahead / Causality**: Does perturbing future input data leave past outputs completely unchanged?
- **Syntactic & Type Parity**: Do static types, linter checks, and base unit tests pass with zero warnings?

### Tier 2: In-Sample Benchmark Verification
- Run the benchmark on the primary training dataset.
- Did the target metric move in the predicted direction?
- If the target metric did NOT move or degraded, the hypothesis is **FALSIFIED**.

### Tier 3: Out-of-Sample (OOS) Holdout & Stress Validation
- Evaluate the exact same code on **unseen holdout data** or adversarial edge cases.
- **The Generalization Test**:
  $$\text{Sharpe}_{\text{OOS}} \approx \text{Sharpe}_{\text{IS}} \quad \text{or} \quad \text{Error}_{\text{OOS}} \le \text{Threshold}$$
- If the change works in-sample but collapses out-of-sample, it is a **curve-fit illusion**. Reject it.

---

## Phase 4: Resolution & Ledgering

### If Verified:
1. Stage modified files (`git add`).
2. Commit with a clean, descriptive message linking the hypothesis and empirical evidence:
   ```bash
   git commit -m "refactor(engine): <summary> [H-00X: +Y% target metric, verified OOS]"
   ```
3. Document the confirmed finding in the project's experiment ledger.

### If Falsified:
1. Revert immediately using `git restore <files>`. Do not leave broken experiments in the working tree.
2. Record the failed hypothesis and the empirical reason it failed in the project's ledger. A documented negative result prevents the agent (and future teammates) from repeating the same dead end.

---

## Practical Application Example (From Quantitative Trading)

- **Observation**: 505 trades in 5 years; median duration was 4 hours; total fees were 13.6% of equity; returns stalled at +26%.
- **Hypothesis**: The strategy state machine was exiting on 4h candle closes, while the 1h backtester was exiting on intrabar wicks, causing the backtester to re-enter the same trade immediately as a "zombie signal."
- **Refactor**: Added `prev_tr == 0` guard in backtester to require a fresh signal transition before opening a position.
- **Test**:
  - Invariant: Zero-lookahead perturbation passed.
  - In-Sample: Trade count dropped from 505 to 96; fees dropped by 90%; returns compounded to 10.5x.
  - Out-of-Sample: 2024–2026 unseen data delivered 9.35x return with Sharpe 1.33 (matching 1.39 in-sample).
- **Resolution**: Verified & Committed.
