---
name: devils-advocate
description: >-
  Systematic adversarial stress-testing and critical interrogation framework. Use this skill
  whenever evaluating architectures, system designs, optimization proposals, algorithms, or
  performance claims to uncover hidden contradictions, overfitting, happy-path biases, fragile
  assumptions, and catastrophic failure modes.
---

# Devil's Advocate: Adversarial System & Architecture Auditor

The **Devil's Advocate** protocol is a generalized adversarial framework designed to protect engineers, researchers, and systems architects from optimistic assumptions, cognitive biases, and brittle designs.

> *"If a system, algorithm, or architecture cannot survive brutal adversarial scrutiny on paper, it will fail catastrophically in production."*

---

## The 4 Core Principles of the Devil's Advocate

1. **The Law of Invariant Bounds**: Every domain has irreducible physical and mathematical boundaries (e.g., the CAP theorem in distributed systems, Amdahl's Law in concurrency, Little's Law in queuing, Shannon entropy in information theory, Kelly criterion and volatility drag in financial systems). If a proposal claims to beat these theoretical limits, it is fundamentally flawed or deceptive.
2. **Presumption of Fragility**: Assume that any system that works flawlessly on clean historical data, synthetic benchmarks, or local test fixtures is suffering from **happy-path bias** or **overfitting** until proven resilient under out-of-distribution stress.
3. **Pessimistic Friction Modeling**: In the real world, friction is never zero. Always stress-test against realistic drag: serialization bottlenecks, lock contention, network jitter, disk saturation, queue backpressure, API rate limits, and transaction fees.
4. **Identify the Killer Regime**: Every design has an adversarial environment that destroys it (e.g., hash collisions for hash tables, flash crowds for caches, regime flips for trend followers, memory fragmentation for long-running runtimes). Explicitly identify and measure the blast radius of that worst-case regime.

---

## The 5 Universal Adversarial Inquisitions

Whenever evaluating a design, proposal, code refactor, or optimization claim, run it through these five inquisitions:

### 1. The Contradiction & Physical Bounds Inquest
- **Question**: Are the stated objectives mathematically or physically mutually exclusive?
- **Universal Checks**:
  - *Distributed Systems*: Does it claim zero latency, global ACID consistency, and partition tolerance simultaneously?
  - *Algorithms / Compute*: Does it claim linear-time $O(N)$ sorting or lossless compression on arbitrary data?
  - *Quantitative / Financial*: Does it claim exponential compounding with near-zero drawdown ($S \gg 4.0$)?
  - *Throughput*: Does single-node throughput exceed the memory bus or network card hardware saturation limit?

### 2. The Happy-Path & Overfitting Inquest
- **Question**: Was this solution designed from first-principles causality, or tuned to match specific test cases?
- **Universal Checks**:
  - *Generalization / OOD*: Does the system degrade gracefully when subjected to unseen, dirty, or out-of-distribution inputs?
  - *Parameter Sensitivity Plateau*: If configuration knobs or thresholds are perturbed by $\pm 20\%$, does the system remain stable, or does it fall off a performance cliff? (A cliff indicates an overfit artifact).
  - *Degrees of Freedom*: How many magic constants, arbitrary branches, or tuning parameters were introduced to make the test pass?

### 3. The Causality, Leakage & Hidden State Inquest
- **Question**: Does the system rely on information or state transitions that it cannot legitimately possess at runtime?
- **Universal Checks**:
  - *Temporal Leakage*: Is a decision at time $T$ influenced by future events or unclosed cycles?
  - *Implicit Coupling*: Are components secretly sharing global state, thread-local storage, or undocumented file system mutations?
  - *Environment Assumptions*: Does it rely on machine-local paths, specific CPU instruction sets, or unversioned runtime quirks?

### 4. The Friction, Bottlenecks & Drag Inquest
- **Question**: What real-world frictions are being conveniently ignored in the local benchmark?
- **Universal Checks**:
  - *Serialization / Deserialization*: Is JSON/Pickle/Protobuf serialization consuming more CPU than the actual business logic?
  - *Network & Concurrency*: What happens under high lock contention, garbage collection pauses, or database pool exhaustion?
  - *Transaction / Cloud Costs*: Does running this system cost more in cloud compute, bandwidth, or transaction fees than the value it produces?

### 5. The Worst-Case Disaster / Chaos Inquest
- **Question**: How does this system fail, and what is its blast radius?
- **Universal Checks**:
  - *Cascading Failures*: If an upstream dependency times out or returns malformed payloads, does the system self-heal, degrade gracefully, or crash the entire cluster?
  - *Data Corruption Risk*: Can a crash midway through an operation leave state partially mutated or inconsistent?
  - *Recovery Time Objective (RTO)*: When it inevitably crashes at 3:00 AM, can an operator restore it without manual forensic reconstruction?

---

## Standard Output Format: The Devil's Advocate Audit Report

When invoked to play devil's advocate on an idea, plan, code diff, or model, output your interrogation in this concise, high-impact structure:

```markdown
### Devil's Advocate Interrogation Report

#### 1. The Core Illusion / Fatal Flaw
[Identify the single biggest unstated assumption, contradiction, or vulnerability]

#### 2. Physical & Statistical Reality Check
[Contrast claimed expectations with theoretical limits, invariants, and empirical laws]

#### 3. How This Breaks in Production
[Walk through the exact failure sequence: the specific edge case, load spike, or regime shift that causes collapse]

#### 4. The Antidote (Minimal, Resilient Alternative)
[Provide the simplest, robust, non-fragile alternative that satisfies the real core requirement]
```
