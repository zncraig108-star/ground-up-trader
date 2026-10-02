"""core/ensemble.py — deterministic, no-lookahead portfolio combination.

Role
----
Combine multiple validated genomes / equity curves into one portfolio. Every public
function is a pure, deterministic function of the inputs it is given: weights are
derived only from the *past* per-period return matrix supplied by the caller, and a
member's equity curve can only influence the combined curve at or after the member's
own timestamps. There is no randomness, no global state, and no shared mutable state.

All outputs are finite: non-finite inputs are sanitized to neutral values and every
fallback is documented, so callers never receive NaN/inf weights or equity.

Backend
-------
Small ``(periods x members)`` matrices and short equity curves are combined here, so
this module uses NumPy directly and never touches the ``xp`` GPU backend.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "EnsembleMember",
    "inverse_vol_weights",
    "risk_parity_weights",
    "combine_equity_curves",
    "weights_from_members",
]

_EPS = 1e-12
_CONVERGENCE_TOL = 1e-12
_RISK_PARITY_TARGET = 1.0  # unnormalized per-member risk-contribution target


@dataclass(frozen=True)
class EnsembleMember:
    """One validated genome participating in an ensemble portfolio."""

    genome_id: str
    genome: tuple[float, ...]
    weight: float = 0.0


def _clean_returns(period_returns: np.ndarray) -> np.ndarray:
    """Coerce ``(M, P)`` returns to a finite float64 array, rejecting other shapes."""
    r = np.asarray(period_returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("period_returns must be 2-D (M periods x P members)")
    # ASSUMPTION: a non-finite return sample (missing/stale curve) contributes 0.0 to
    # vol/risk estimation rather than poisoning the whole allocation.
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def inverse_vol_weights(period_returns: np.ndarray) -> np.ndarray:
    """Inverse-volatility weights for ``(M periods, P members)`` per-period returns.

    Each member's volatility is the population std (``ddof=0``) of its column, so a
    single-period input still yields finite values. A member whose vol is ~0 is
    clipped to the smallest positive vol present (ties share the top weight) so no
    weight diverges; if every member has zero vol the weights are equal. Deterministic,
    non-negative, finite, and summing to 1.
    """
    r = _clean_returns(period_returns)
    p = r.shape[1]
    if p == 0:
        return np.zeros(0, dtype=np.float64)
    if r.shape[0] == 0:
        return np.full(p, 1.0 / p, dtype=np.float64)

    vol = r.std(axis=0, ddof=0)
    positive = vol[vol > _EPS]
    if positive.size == 0:
        return np.full(p, 1.0 / p, dtype=np.float64)

    raw = 1.0 / np.maximum(vol, positive.min())
    weights = raw / raw.sum()
    if not np.all(np.isfinite(weights)):
        # Overflow guard for pathologically large return scales.
        return np.full(p, 1.0 / p, dtype=np.float64)
    return weights


def risk_parity_weights(period_returns: np.ndarray, iters: int = 200) -> np.ndarray:
    """Iterative risk-parity weights for ``(M periods, P members)`` returns.

    Algorithm (cyclical coordinate descent on the risk-budgeting fixed point
    ``w_i * (Cov w)_i == target`` with ``target = 1``; the final result is normalized,
    which preserves equal risk contributions): for each member ``i`` in turn, with
    ``a = Cov_ii`` and ``b = sum_{j != i} Cov_ij w_j``, replace ``w_i`` by the positive
    root of ``a * w_i^2 + b * w_i - target == 0``. Sweeps repeat until the largest
    weight change is below 1e-12 or ``iters`` sweeps have run.

    Documented fallback: if there is no usable covariance information -- fewer than two
    periods, fewer than two members with positive volatility, or a non-finite
    covariance matrix -- return ``inverse_vol_weights(period_returns)``. Members with
    ~zero volatility are excluded from risk budgeting and receive weight 0.0 (a
    degenerate case for risk parity). Output is always finite, non-negative, and sums
    to 1.
    """
    r = _clean_returns(period_returns)
    p = r.shape[1]
    if p == 0:
        return np.zeros(0, dtype=np.float64)
    m = r.shape[0]
    if m < 2:
        return inverse_vol_weights(period_returns)

    vol = r.std(axis=0, ddof=0)
    active = np.flatnonzero(vol > _EPS)
    if active.size < 2:
        return inverse_vol_weights(period_returns)

    cov = np.cov(r[:, active], rowvar=False, ddof=1)
    if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0.0):
        return inverse_vol_weights(period_returns)

    k = active.size
    w = np.full(k, 1.0 / k, dtype=np.float64)
    for _ in range(max(int(iters), 1)):
        max_delta = 0.0
        for i in range(k):
            a = cov[i, i]
            b = float(cov[i] @ w) - a * w[i]
            root = (-b + np.sqrt(b * b + 4.0 * a * _RISK_PARITY_TARGET)) / (2.0 * a)
            max_delta = max(max_delta, abs(root - w[i]))
            w[i] = root
        if max_delta < _CONVERGENCE_TOL:
            break

    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return inverse_vol_weights(period_returns)

    full = np.zeros(p, dtype=np.float64)
    full[active] = w / total
    return full


def combine_equity_curves(curves: list[list[tuple[int, float]]], weights: np.ndarray,
                          initial_equity: float = 10_000.0) -> list[tuple[int, float]]:
    """Timestamp-aligned weighted combination of member equity curves.

    ``curves[i]`` is member ``i``'s ``[(ts_ms, equity), ...]`` history and ``weights``
    are applied per member. The output lives on the sorted union of all member
    timestamps:

    * at a member's own timestamps its recorded equity is used; between them the
      member's equity is forward-filled;
    * before a member's first timestamp the member is still at ``initial_equity`` (it
      has not started trading), so a member's curve can only influence the combined
      curve at or after its own timestamps (no lookahead);
    * duplicate timestamps inside one member keep the last value seen.

    Returns ascending ``(ts_ms, equity)`` pairs with finite equity. Weights are applied
    as given (callers pass simplex weights, e.g. from ``inverse_vol_weights``).
    """
    ws = np.asarray(weights, dtype=np.float64).reshape(-1)
    if ws.shape[0] != len(curves):
        raise ValueError(f"weights has {ws.shape[0]} entries for {len(curves)} curves")
    # ASSUMPTION: a non-finite weight is treated as 0.0 so it cannot poison the sum.
    ws = np.nan_to_num(ws, nan=0.0, posinf=0.0, neginf=0.0)

    cleaned: list[dict[int, float]] = []
    all_ts: set[int] = set()
    for curve in curves:
        points: dict[int, float] = {}
        for ts, equity in curve:
            eq = float(equity)
            if not np.isfinite(eq):
                # ASSUMPTION: a corrupted equity sample falls back to initial_equity.
                eq = float(initial_equity)
            points[int(ts)] = eq
        cleaned.append(points)
        all_ts.update(points)
    if not all_ts:
        return []

    union = np.asarray(sorted(all_ts), dtype=np.int64)
    combined = np.zeros(union.shape[0], dtype=np.float64)
    for points, weight in zip(cleaned, ws):
        if not points:
            combined += weight * float(initial_equity)
            continue
        member_ts = np.asarray(sorted(points), dtype=np.int64)
        member_eq = np.asarray([points[t] for t in member_ts.tolist()], dtype=np.float64)
        pos = np.searchsorted(member_ts, union, side="right") - 1
        aligned = np.where(pos >= 0, member_eq[np.maximum(pos, 0)], float(initial_equity))
        combined += weight * aligned

    combined = np.nan_to_num(combined, nan=float(initial_equity), posinf=0.0, neginf=0.0)
    return [(int(ts), float(eq)) for ts, eq in zip(union.tolist(), combined.tolist())]


def weights_from_members(members: list[EnsembleMember],
                         period_returns: np.ndarray) -> list[EnsembleMember]:
    """Return copies of ``members`` with weights filled by inverse-vol allocation.

    Inverse-vol is the default policy because it depends only on each member's past
    return-stream volatility (no joint optimization, no lookahead). Use
    ``risk_parity_weights`` explicitly when a covariance-aware allocation is wanted.
    """
    weights = inverse_vol_weights(period_returns)
    if weights.shape[0] != len(members):
        raise ValueError(f"weights has {weights.shape[0]} entries for {len(members)} members")
    return [replace(member, weight=float(w)) for member, w in zip(members, weights)]
