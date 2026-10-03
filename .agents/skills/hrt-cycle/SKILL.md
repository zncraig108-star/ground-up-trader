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
  └──────────────────────────────┘└──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │ 4C. STOP, AUDIT & ZOOM OUT   │
                                  │     - Freeze modifications   │
                                  │     - Forensic audit of why  │
                                  │     - Shift macro paradigm   │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │ 4D. THEORIZE & IDEATE        │
                                  │     - First-principles idea  │
                                  │     - Inversion & asymmetry  │
                                  │     - Cross-domain transfer  │
                                  └──────────────┬───────────────┘
                                                 │
                                                 └──────────┐
                                                            ▼
                                               (Loop to Phase 1: Hypothesize)
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

## Phase 4: Resolution, Audit & Ideation Loop

### 4A. If Verified: Commit & Ledger
1. Stage modified files (`git add`).
2. Commit with a clean, descriptive message linking the hypothesis and empirical evidence:
   ```bash
   git commit -m "refactor(engine): <summary> [H-00X: +Y% target metric, verified OOS]"
   ```
3. Document the confirmed finding in the project's experiment ledger.

### 4B. If Falsified: Clean Revert
1. Freeze all code edits immediately. Resist the urge to "just tweak one more line".
2. Run `git restore .` to revert to the last verified baseline. Never construct further changes on top of a compromised or unproven state.
3. Record the failed hypothesis and the empirical reason it failed in the project's ledger.

---

### 4C. The "Stop, Audit & Zoom Out" Protocol

When an experiment fails, a test breaks, or an optimization stalls, **NEVER engage in reactionary micro-patching or shotgun tweaking**. Immediately execute this three-step protocol:

1. **STOP (Halt & Clean Reset)**:
   - Break the thrashing loop: recognize that repetitive minor failures signal a flawed causal model, not a parameter tuning issue.
   - Ensure the working tree is pristine.

2. **AUDIT (Forensic Root Cause Analysis)**:
   - **Audit Assumptions**: What implicit assumptions about data, timing, environment, or execution mechanics were violated?
   - **Audit Mechanics**: Trace the data flow step-by-step. Is there an off-by-one index, hidden fee drag, leakage, or sampling bias?
   - **Audit History**: Review the experiment ledger. Have we attempted a variant of this before? What did we learn?

3. **ZOOM OUT (Macro Architectural Shift)**:
   - **Question the Framing**: Are we trying to squeeze juice from a dry lemon? (e.g., trying to optimize a 5-minute scalper where exchange fees mathematically guarantee negative expectancy).
   - **Step Back to First Principles**: What is the structural driver of the underlying system? What natural asymmetry or constraint exists that we are ignoring?
   - **Reframe at the System Level**: Move from micro-parameter tuning to macro-systemic redesign (e.g., shifting timeframes, expanding asset universe, filtering macro regimes, or rethinking risk allocation).

---

### 4D. Theorize & Ideate (Generating Breakthrough Ideas)

After zooming out, do not fall back into random guessing. Use systematic theoretical ideation frameworks to brainstorm high-conviction, non-obvious hypotheses:

### 1. First-Principles Decomposition
- Strip away convention and break the problem down into fundamental axioms and physical laws:
  - Where does economic value, edge, or information truly originate in this system?
  - What are the irreducible physical constraints (e.g., fees, latency, orderbook depth, entropy, signal-to-noise ratio)?
  - What behavioral or structural forces cause market participants to be forced, programmatic, or systematically biased?

### 2. Inversion (Munger's Inversion Principle)
- Instead of asking *"How do we make 100x?"*, ask: *"What guarantees failure, zero alpha, or capital destruction?"*
  - *Failure mode 1*: Over-trading in chop $\rightarrow$ **Invert**: Require regime/trend filters; hold longer; do nothing when unaligned.
  - *Failure mode 2*: Fee bleeding $\rightarrow$ **Invert**: Widen profit targets; trade higher timeframes; reduce rebalancing frequency.
  - *Failure mode 3*: Fighting secular momentum $\rightarrow$ **Invert**: Asymmetric directional alignment (longs in secular bull regimes, shorts only in macro bears).

### 3. Asymmetry & Convexity Seeking
- Hunt for mechanics with positive skew:
  - **Bounded Downside**: Small, fixed initial risk or dynamic trailing stops that kill false breakouts quickly.
  - **Uncapped Fat-Tailed Upside**: Compounding position sizing, letting macro trends run over weeks/months, and multi-asset diversification that captures monster outlier runs.

### 4. Cross-Domain Analogy & Mechanistic Transfer
- Borrow proven models from other disciplines:
  - **Control Theory**: Damped feedback loops, PID controllers, and Kalman filters for dynamic volatility scaling.
  - **Thermodynamics & Information Theory**: Shannon entropy for regime detection, energy states for breakout accumulation.
  - **Evolutionary Biology & Ecology**: Portfolio diversity, orthogonal niche strategies, and environmental adaptation.

### 5. Synthesize into Phase 1 Hypothesis
- Convert the top theoretical concept into a formal, falsifiable `[H-00X]` hypothesis format. Never leap straight to code; theory must precede implementation.

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
