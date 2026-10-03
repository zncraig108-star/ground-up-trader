# Experiment Ledger Template

Use this template to record every hypothesis tested in the project. Maintain this in `experiments.md` or `docs/experiments.md`.

---

## Experiment [H-00X]: <Short Title>

- **Date**: YYYY-MM-DD
- **Target Subsystem**: `path/to/subsystem`
- **Status**: [ VERIFIED | FALSIFIED | REVERTED ]

### 1. Hypothesis Formulation
- **Empirical Observation**: What metric, behavior, or log triggered this experiment?
- **Causal Mechanism**: Why is this happening (physical, logical, mathematical explanation)?
- **Proposed Intervention**: What is the single minimal refactor or change?
- **Target Metric**: What metric must improve and by how much?
- **Invariants**: What must not break (conservation laws, zero-lookahead, test suite)?

### 2. Implementation (/ponytail ultra)
- **Files Modified**:
  - `path/to/file.py`: (summary of change)
- **Causal Variable**: Exactly one variable modified.

### 3. Empirical Results
| Metric / Test | Baseline | Post-Refactor | Delta / Outcome |
| :--- | :--- | :--- | :--- |
| **Tier 1: Invariant Check** | Pass | Pass | Invariants Preserved |
| **Tier 2: In-Sample Metric** | X.XX | Y.YY | +Z% |
| **Tier 3: Out-of-Sample Metric**| A.AA | B.BB | Generalization Confirmed |

### 4. Conclusion & Action
- **Verdict**: [VERIFIED / FALSIFIED]
- **Action Taken**: 
  - If Verified: Committed in `git commit -m "..."`.
  - If Falsified: Reverted via `git restore .`. Followed Stop, Audit & Zoom Out protocol below.
- **Lesson Learned**: What did this result reveal about the system's underlying dynamics?

### 5. If Falsified: Stop, Audit & Zoom Out Diagnostic
- **STOP**: Code reverted cleanly via `git restore .`? [Yes/No]
- **AUDIT**:
  - Violated assumptions:
  - Data / pipeline / mechanics audit:
  - Ledger check (did we already try this?):
- **ZOOM OUT**:
  - Is the current paradigm or framing flawed?
  - What structural or first-principles macro redesign is indicated?
