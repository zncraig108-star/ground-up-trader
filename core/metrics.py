"""core/metrics.py — Sharpe/Sortino/max-DD/WFE + a robust, NaN-safe fitness function.

All functions are vectorized over the population axis ``P`` (first axis of the inputs)
and operate on the ``xp`` backend, so they work identically on CPU/GPU.

Conventions (documented):
  * ``returns`` is a ``(P, T)`` array of period-over-period fractional returns.
  * ``periods_per_year`` is the number of return samples per year (e.g. 8760 hourly).
  * Sharpe/Sortino use the arithmetic mean/std of returns.
  * ``std == 0`` -> Sharpe 0; no downside -> Sortino capped at ``SORTINO_CAP`` (finite).
  * Max drawdown is a fraction in ``[0, 1]`` from the running peak.
  * WFE = oos_metric / is_metric, clamped to ``[0, 1.5]``.

Why the robust fitness uses ``sqrt`` + clamping (fixes the naive formula's bugs):
  * ``sqrt(sharpe_pos * sortino_pos)`` is geometric, so a *negative* Sharpe or Sortino
    (zeroed via ``max(., 0)``) can no longer flip the sign of the product and produce a
    misleadingly *positive* score for a losing strategy.
  * ``dd_factor = clip(1 - max_dd, 0, 1)`` can never go negative, so a drawdown > 100%
    (equity going negative) can no longer produce a fitness that *increases* with
    larger losses.
  * Every input is sanitized; any NaN/inf makes the whole fitness 0.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.xp import xp

__all__ = [
    "SORTINO_CAP",
    "HOURS_PER_YEAR",
    "MINUTES_PER_YEAR",
    "MetricsResult",
    "sharpe",
    "sortino",
    "max_drawdown",
    "walk_forward_efficiency",
    "robust_fitness",
    "compute_metrics",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "sharpe_shrinkage",
    "downside_penalized_score",
    "multi_window_score",
    "probability_of_backtest_overfitting",
]

SORTINO_CAP = 10.0  # finite cap when there is zero downside deviation
HOURS_PER_YEAR = 365 * 24
MINUTES_PER_YEAR = 365 * 24 * 60

_FLOAT = xp.float64


def _sanitize(x: Any) -> Any:
    """Return ``x`` with NaN/inf replaced by 0 (input may be scalar or array)."""
    return xp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def sharpe(returns: Any, periods_per_year: float) -> Any:
    """Annualized Sharpe ratio, vectorized over ``P``.

    ``std == 0 -> 0``; NaN/inf sanitized to 0.
    """
    r = xp.asarray(returns, dtype=_FLOAT)
    mean = r.mean(axis=1)
    std = r.std(axis=1)
    with xp.errstate(divide="ignore", invalid="ignore"):
        out = xp.where(std > 0.0, mean / std * math.sqrt(periods_per_year), 0.0)
    return _sanitize(out)


def sortino(returns: Any, periods_per_year: float) -> Any:
    """Annualized Sortino ratio (downside deviation only), vectorized over ``P``.

    No downside deviation -> ``SORTINO_CAP`` if mean is positive, else 0 (never inf).
    """
    r = xp.asarray(returns, dtype=_FLOAT)
    mean = r.mean(axis=1)
    downside = xp.minimum(r, 0.0) ** 2
    dd = xp.sqrt(downside.mean(axis=1))
    with xp.errstate(divide="ignore", invalid="ignore"):
        ratio = mean / dd * math.sqrt(periods_per_year)
    out = xp.where(dd > 0.0, ratio, xp.where(mean > 0.0, SORTINO_CAP, 0.0))
    return _sanitize(out)


def max_drawdown(equity: Any) -> Any:
    """Maximum drawdown as a fraction in ``[0, 1]`` from the running peak, per genome."""
    eq = xp.asarray(equity, dtype=_FLOAT)
    peak = xp.maximum.accumulate(eq, axis=1)
    with xp.errstate(divide="ignore", invalid="ignore"):
        dd = 1.0 - eq / peak
    out = dd.max(axis=1)
    # Clamp to [0, 1]: protects against negative/zero equity curves.
    out = xp.where(peak.max(axis=1) <= 0.0, 0.0, out)
    return _sanitize(xp.clip(out, 0.0, 1.0))


def walk_forward_efficiency(oos_metric: Any, is_metric: Any) -> Any:
    """WFE = oos / is, clamped to ``[0, 1.5]`` (nan/inf-safe)."""
    oos = xp.asarray(oos_metric, dtype=_FLOAT)
    is_ = xp.asarray(is_metric, dtype=_FLOAT)
    with xp.errstate(divide="ignore", invalid="ignore"):
        wfe = xp.where(is_ > 0.0, oos / is_, 0.0)
    return _sanitize(xp.clip(wfe, 0.0, 1.5))


def robust_fitness(sharpe_arr: Any, sortino_arr: Any, max_dd: Any, wfe: Any,
                   trades: Any, min_trades: int, trade_penalty_power: float,
                   total_return: Any | None = None) -> Any:
    """Robust fitness per DEEPSEEK_BUILD_PROMPT.md §6 with total return growth scaling.
    Never returns NaN/inf."""
    s = _sanitize(xp.asarray(sharpe_arr, dtype=_FLOAT))
    so = _sanitize(xp.asarray(sortino_arr, dtype=_FLOAT))
    dd = xp.clip(xp.asarray(max_dd, dtype=_FLOAT), 0.0, 1.0)
    w = xp.clip(xp.asarray(wfe, dtype=_FLOAT), 0.0, 1.5)
    tr = xp.asarray(trades, dtype=_FLOAT)

    sharpe_pos = xp.maximum(s, 0.0)
    sortino_pos = xp.maximum(so, 0.0)
    dd_factor = xp.clip(1.0 - dd, 0.0, 1.0)
    wfe_factor = xp.clip(w, 0.0, 1.5)

    raw = xp.sqrt(sharpe_pos * sortino_pos) * dd_factor * wfe_factor

    # Reward massive capital growth when returns are positive
    if total_return is not None:
        ret = _sanitize(xp.asarray(total_return, dtype=_FLOAT))
        return_factor = xp.power(1.0 + xp.maximum(ret, 0.0), 0.6)
        raw = raw * return_factor

    # Smooth trade-count penalty (NOT a hard cliff).
    trade_factor = xp.minimum(1.0, tr / max(min_trades, 1)) ** trade_penalty_power
    out = raw * trade_factor
    return _sanitize(out)


@dataclass
class MetricsResult:
    """Aggregate per-genome metrics (all arrays shaped ``(P,)``)."""

    sharpe: Any
    sortino: Any
    max_drawdown: Any
    total_return: Any  # equity[-1]/equity[0] - 1
    trades: Any
    fitness: Any  # wfe == 1.0 unless an OOS window is provided


def compute_metrics(equity_curve: Any, periods_per_year: float, trades: Any,
                    min_trades: int = 50, trade_penalty_power: float = 1.5,
                    wfe: Any | None = None) -> MetricsResult:
    """Compute the full metric bundle for a population of equity curves.

    ``equity_curve``: ``(P, T)``. ``trades``: ``(P,)`` trade counts. ``wfe``: optional
    ``(P,)`` walk-forward efficiency (defaults to 1.0 for a single-window run).
    """
    eq = xp.asarray(equity_curve, dtype=_FLOAT)
    p = eq.shape[0]
    returns = xp.diff(eq, axis=1) / xp.maximum(eq[:, :-1], 1e-12)

    s = sharpe(returns, periods_per_year)
    so = sortino(returns, periods_per_year)
    dd = max_drawdown(eq)

    total_return = _sanitize(eq[:, -1] / xp.maximum(eq[:, 0], 1e-12) - 1.0)

    if wfe is None:
        wfe_arr = xp.ones(p, dtype=_FLOAT)
    else:
        wfe_arr = xp.asarray(wfe, dtype=_FLOAT)

    fit = robust_fitness(s, so, dd, wfe_arr, trades, min_trades, trade_penalty_power, total_return)

    return MetricsResult(
        sharpe=s,
        sortino=so,
        max_drawdown=dd,
        total_return=total_return,
        trades=xp.asarray(trades, dtype=xp.int64),
        fitness=fit,
    )


# --- Phase 3: statistically rigorous evaluation primitives -------------------
#
# These build on the same conventions as the core metrics above: every input is a
# population-first array, everything runs on ``xp``, and every output is sanitized so a
# bad row scores 0 rather than contaminating the population with NaN/inf.

# Euler-Mascheroni constant, used by the expected-maximum-Sharpe order statistic
# (Bailey & Lopez de Prado, "The Deflated Sharpe Ratio", 2014).
_EULER_MASCHERONI = 0.5772156649015329

# Number of independent observations of information assumed per executed trade.
# Used only for the effective-sample-size shrinkage below.
_PER_TRADE_INFO = 3.0


def _norm_ppf(p: Any) -> Any:
    """Inverse standard-normal CDF ``Z^-1(p)`` (Acklam's rational approximation).

    Vectorized over ``p``, which is clipped to ``(0, 1)``; the result is finite for any
    finite input. Relative error ~1e-9, ample for the expected-max order statistic used
    by :func:`deflated_sharpe_ratio`. Implemented with elementary ``xp`` ops (no
    ``erfinv``) so CuPy/NumPy behave identically.
    """
    a0, a1, a2, a3, a4, a5 = (
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    )
    b0, b1, b2, b3, b4 = (
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    )
    c0, c1, c2, c3, c4, c5 = (
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    )
    d0, d1, d2, d3 = (
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow

    p_ = xp.clip(xp.asarray(p, dtype=_FLOAT), 1e-12, 1.0 - 1e-12)
    q = p_ - 0.5
    r = q * q
    central = (
        (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
        / (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0)
    )
    lower_t = xp.sqrt(-2.0 * xp.log(p_))
    lower = (
        (((((c0 * lower_t + c1) * lower_t + c2) * lower_t + c3) * lower_t + c4)
          * lower_t + c5)
        / ((((d0 * lower_t + d1) * lower_t + d2) * lower_t + d3) * lower_t + 1.0)
    )
    upper_t = xp.sqrt(-2.0 * xp.log1p(-p_))
    upper = -(
        (((((c0 * upper_t + c1) * upper_t + c2) * upper_t + c3) * upper_t + c4)
          * upper_t + c5)
        / ((((d0 * upper_t + d1) * upper_t + d2) * upper_t + d3) * upper_t + 1.0)
    )
    return _sanitize(xp.where(p_ < plow, lower, xp.where(p_ > phigh, upper, central)))


def _norm_erf(x: Any) -> Any:
    """Error function via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7).

    Implemented with elementary ``xp`` ops because NumPy has no ``erf`` and CuPy's may
    differ across builds; this keeps CUDA/CPU parity exact.
    """
    z = xp.asarray(x, dtype=_FLOAT)
    ax = xp.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_abs = 1.0 - poly * xp.exp(-ax * ax)
    # ``sign(0) == 0``, so erf(0) == 0 falls out of the odd extension naturally.
    return _sanitize(xp.sign(z) * erf_abs)


def _norm_cdf(z: Any) -> Any:
    """Standard-normal CDF ``Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))``, in ``[0, 1]``.

    Uses :func:`_norm_erf`; finite for finite ``z``.
    """
    erf = _norm_erf(xp.asarray(z, dtype=_FLOAT) / math.sqrt(2.0))
    return xp.clip(0.5 * (1.0 + erf), 0.0, 1.0)


def probabilistic_sharpe_ratio(returns: Any, sr_benchmark: float = 0.0,
                               periods_per_year: float = HOURS_PER_YEAR) -> Any:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012), per genome.

    For each row of a ``(P, T)`` array of per-period fractional returns::

        SR_hat = mean(returns) / std(returns)          # per-period, NOT annualized
        SR*    = sr_benchmark / sqrt(periods_per_year) # benchmark de-annualized
        PSR    = Phi( (SR_hat - SR*) * sqrt(T - 1)
                      / sqrt(1 - skew*SR_hat + ((kurt - 1)/4)*SR_hat**2) )

    with ``skew`` / ``kurt`` the population skewness and *non-excess* kurtosis of the
    row (normal kurtosis == 3, so ``(kurt - 1)/4 == 0.5`` under normality).

    Returns a probability in ``[0, 1]``; ``T < 3``, zero variance, a non-positive
    variance denominator, or any non-finite return in a row yields exactly ``0.0``.
    """
    r = xp.asarray(returns, dtype=_FLOAT)
    if r.ndim != 2:
        raise ValueError("probabilistic_sharpe_ratio expects a (P, T) returns array")
    t = r.shape[1]
    row_finite = xp.isfinite(r).all(axis=1)
    r0 = _sanitize(r)

    mean = r0.mean(axis=1)
    dev = r0 - mean[:, None]
    m2 = (dev ** 2).mean(axis=1)
    std = xp.sqrt(m2)
    with xp.errstate(divide="ignore", invalid="ignore"):
        sr_hat = xp.where(std > 0.0, mean / std, 0.0)
        skew = xp.where(std > 0.0, (dev ** 3).mean(axis=1) / std ** 3, 0.0)
        kurt = xp.where(std > 0.0, (dev ** 4).mean(axis=1) / std ** 4, 0.0)

    # ASSUMPTION: ``sr_benchmark`` is in annualized units (module convention) and is
    # de-annualized so it shares the per-observation frequency of SR_hat, as required by
    # the variance term of the PSR formula.
    ppy = float(periods_per_year)
    if not math.isfinite(ppy) or ppy <= 0.0:
        ppy = 1.0  # ASSUMPTION: a degenerate frequency means "no annualization".
    sb = float(sr_benchmark)
    sr_star = sb / math.sqrt(ppy) if math.isfinite(sb) else 0.0

    var_denom = 1.0 - skew * sr_hat + 0.25 * (kurt - 1.0) * sr_hat ** 2
    with xp.errstate(divide="ignore", invalid="ignore"):
        z_stat = (sr_hat - sr_star) * math.sqrt(max(t - 1, 0)) / xp.sqrt(var_denom)

    ok = row_finite & (t >= 3) & (std > 0.0) & (var_denom > 0.0)
    psr = xp.clip(_norm_cdf(xp.where(ok, z_stat, 0.0)), 0.0, 1.0)
    return _sanitize(xp.where(ok, psr, 0.0))


def deflated_sharpe_ratio(sharpe_obs: Any, n_trials: int, n_obs: int, skew: Any,
                          kurtosis: Any, sr_variance_across_trials: Any,
                          sr_benchmark: float = 0.0) -> Any:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014), per genome.

    First the expected maximum Sharpe under the null of no skill across ``n_trials``
    independent trials (variance ``sr_variance_across_trials``)::

        E[max SR] = sqrt(var) * ( (1 - gamma) * Z^-1(1 - 1/N)
                                  + gamma * Z^-1(1 - 1/(N*e)) )

    with Euler-Mascheroni ``gamma``. Then the PSR of ``sharpe_obs`` against the hurdle
    ``SR* = sr_benchmark + max(E[max SR], 0)``::

        DSR = Phi( (SR_obs - SR*) * sqrt(n_obs - 1)
                   / sqrt(1 - skew*SR_obs + ((kurt - 1)/4)*SR_obs**2) )

    ``skew`` / ``kurtosis`` are vectors or scalars in the same units as ``sharpe_obs``
    and should describe the return series the Sharpe was computed from (kurtosis is
    *non-excess*; normal == 3). All Sharpe-valued inputs share the frequency of the
    observations (non-annualized is the statistically exact convention;
    ``# ASSUMPTION:`` pass consistently-annualized values if that is what you have --
    the deflation ordering is preserved either way).

    ``n_trials < 2`` uses ``E[max SR] = 0``. Returns a value in ``[0, 1]``; ``n_obs < 3``,
    non-positive variance denominator, negative/zero trial variance, or any non-finite
    input yields exactly ``0.0``.
    """
    sr_raw = xp.asarray(sharpe_obs, dtype=_FLOAT)
    sk_raw = xp.asarray(skew, dtype=_FLOAT)
    ku_raw = xp.asarray(kurtosis, dtype=_FLOAT)
    var_raw = xp.asarray(sr_variance_across_trials, dtype=_FLOAT)
    finite = (xp.isfinite(sr_raw) & xp.isfinite(sk_raw) & xp.isfinite(ku_raw)
              & xp.isfinite(var_raw))
    sr = _sanitize(sr_raw)
    sk = _sanitize(sk_raw)
    ku = _sanitize(ku_raw)
    var = xp.maximum(_sanitize(var_raw), 0.0)

    n = max(int(n_trials), 1)
    if n >= 2:
        z1 = _norm_ppf(1.0 - 1.0 / n)
        z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
        expected_max = xp.sqrt(var) * (
            (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
        )
    else:
        # A single trial has no multiple-testing inflation: E[max SR] = 0.
        expected_max = xp.zeros_like(var)
    sb = float(sr_benchmark)
    hurdle = expected_max + (sb if math.isfinite(sb) else 0.0)

    n_obs_i = max(int(n_obs), 0)
    var_denom = 1.0 - sk * sr + 0.25 * (ku - 1.0) * sr ** 2
    with xp.errstate(divide="ignore", invalid="ignore"):
        z_stat = (sr - hurdle) * math.sqrt(max(n_obs_i - 1, 0)) / xp.sqrt(var_denom)
    ok = finite & (n_obs_i >= 3) & (var_denom > 0.0)
    dsr = xp.clip(_norm_cdf(xp.where(ok, z_stat, 0.0)), 0.0, 1.0)
    return _sanitize(xp.where(ok, dsr, 0.0))


def sharpe_shrinkage(sharpe_obs: Any, n_obs: Any, n_trades: Any,
                     prior_strength: float = 20.0) -> Any:
    """Bayesian-style shrinkage of a Sharpe estimate toward 0, per genome::

        n_eff   = min(n_obs, n_trades * 3.0)
        shrunk  = sharpe_obs * n_eff / (n_eff + prior_strength)

    ``n_obs`` is the number of return observations and ``n_trades`` the executed trade
    count; short samples or thin trade counts are pulled toward 0 (the shrinkage is on
    magnitude, so a negative Sharpe becomes less negative). Inputs are sanitized and
    clamped non-negative, so the output is always finite and never exceeds
    ``abs(sharpe_obs)`` in magnitude. ``prior_strength`` is in effective-observation
    units (default 20).
    """
    sr = _sanitize(xp.asarray(sharpe_obs, dtype=_FLOAT))
    n_o = xp.maximum(_sanitize(xp.asarray(n_obs, dtype=_FLOAT)), 0.0)
    n_t = xp.maximum(_sanitize(xp.asarray(n_trades, dtype=_FLOAT)), 0.0)
    # ASSUMPTION: each executed trade carries ~3 hourly observations of information, so
    # trades are converted to an observation-equivalent sample size before capping by
    # the actual number of observations.
    n_eff = xp.minimum(n_o, n_t * _PER_TRADE_INFO)
    prior = max(float(prior_strength), 0.0)
    with xp.errstate(divide="ignore", invalid="ignore"):
        factor = n_eff / xp.maximum(n_eff + prior, 1e-12)
    return _sanitize(sr * xp.clip(factor, 0.0, 1.0))


def downside_penalized_score(sharpe_arr: Any, max_dd: Any, trades: Any,
                             min_trades: int) -> Any:
    """Cost/robustness-oriented scalar per genome::

        score = max(sharpe, 0) * clip(1 - max_dd, 0, 1) * min(1, trades / max(min_trades, 1))

    Losing strategies score 0; drawdown > 100% cannot reward; thin trade counts shrink
    the score smoothly. All inputs are sanitized (NaN/inf -> 0) and the result is always
    finite and ``>= 0``.
    """
    s = xp.maximum(_sanitize(xp.asarray(sharpe_arr, dtype=_FLOAT)), 0.0)
    dd = xp.clip(_sanitize(xp.asarray(max_dd, dtype=_FLOAT)), 0.0, 1.0)
    tr = xp.maximum(_sanitize(xp.asarray(trades, dtype=_FLOAT)), 0.0)
    trade_factor = xp.minimum(1.0, tr / max(float(min_trades), 1.0))
    return _sanitize(s * (1.0 - dd) * trade_factor)


def multi_window_score(returns_by_window: Any, weights: Any | None = None) -> Any:
    """Robust multi-window Sharpe score per genome, over a ``(P, W, T)`` array.

    Each window's rows are annualized with :func:`sharpe` at ``HOURS_PER_YEAR``
    (``# ASSUMPTION:`` windows are hourly returns, the system's default timeframe);
    zero-variance windows score 0. The genome score is the worst window blended with the
    median window::

        score = 0.5 * median_window_sharpe + 0.5 * min_window_sharpe

    so a strong median cannot hide a bad regime (``score <= median`` always holds).

    ``weights`` is an optional per-window weight vector of shape ``(W,)`` or ``(P, W)``
    (default equal). Weights tilt the *median* via a weighted median; the minimum stays
    unweighted to preserve the worst-window guarantee (``# ASSUMPTION:``). Windows whose
    returns contain non-finite values score 0. All outputs are sanitized to finite.
    """
    r = xp.asarray(returns_by_window, dtype=_FLOAT)
    if r.ndim != 3:
        raise ValueError("multi_window_score expects a (P, W, T) array")
    p, w, t = r.shape
    window_finite = xp.isfinite(r).all(axis=2)

    # Sanitize before the reduction: invalid windows are zeroed by the mask below
    # anyway, and this keeps np.nan/inf out of mean/std (no spurious CPU warnings).
    safe = xp.reshape(_sanitize(r), (p * w, t))
    per_window = xp.reshape(sharpe(safe, HOURS_PER_YEAR), (p, w))
    per_window = xp.where(window_finite, per_window, 0.0)

    if weights is None:
        wt = xp.ones((p, w), dtype=_FLOAT)
    else:
        raw_w = xp.maximum(_sanitize(xp.asarray(weights, dtype=_FLOAT)), 0.0)
        if raw_w.ndim == 1:
            raw_w = xp.ones((p, 1), dtype=_FLOAT) * raw_w[None, :]
        elif raw_w.ndim != 2 or raw_w.shape != (p, w):
            raise ValueError("weights must have shape (W,) or (P, W)")
        # Rows with no usable weight fall back to equal weighting.
        wt = xp.where(raw_w.sum(axis=1, keepdims=True) > 0.0,
                      raw_w, xp.ones((p, w), dtype=_FLOAT))

    order = xp.argsort(per_window, axis=1)
    sorted_sr = xp.take_along_axis(per_window, order, axis=1)
    sorted_w = xp.take_along_axis(wt, order, axis=1)
    cum_w = xp.cumsum(sorted_w, axis=1)
    half_w = 0.5 * sorted_w.sum(axis=1, keepdims=True)
    median_idx = xp.argmax(cum_w >= half_w, axis=1)
    median_sr = xp.take_along_axis(sorted_sr, median_idx[:, None], axis=1)[:, 0]
    worst_sr = per_window.min(axis=1)

    return _sanitize(0.5 * median_sr + 0.5 * worst_sr)


def probability_of_backtest_overfitting(period_returns: Any,
                                        max_combinations: int = 2000) -> float:
    """CSCV estimate of the Probability of Backtest Overfitting (Bailey et al.).

    Input is a ``(P, S)`` matrix of **per-period returns** for ``P`` candidates over ``S``
    aligned validation periods. For each symmetric split of the periods into an
    in-sample and an out-of-sample half, the best in-sample candidate's out-of-sample
    rank is measured; PBO is the fraction of splits where it lands below the median.

    Returns a float in ``[0, 1]`` (higher = more overfit). Returns ``0.5`` (not
    estimable) when there are fewer than 2 candidates or fewer than 4 periods. The
    statistic is computed on the CPU over a small matrix and is deterministic.
    """
    r = np.asarray(period_returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("probability_of_backtest_overfitting expects a (P, S) matrix")
    p, s = r.shape
    if p < 2 or s < 4:
        return 0.5
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    half = s // 2
    combos = list(itertools.combinations(range(s), half))
    if len(combos) > max_combinations:
        stride = len(combos) / max_combinations
        combos = [combos[int(i * stride)] for i in range(max_combinations)]
    below_median = 0
    for is_idx in combos:
        is_set = set(is_idx)
        oos_idx = [j for j in range(s) if j not in is_set]
        is_perf = r[:, list(is_idx)].mean(axis=1)
        oos_perf = r[:, oos_idx].mean(axis=1)
        best = int(np.argmax(is_perf))
        # Mid-rank of the in-sample winner, in [0, 1].
        rank = (np.sum(oos_perf < oos_perf[best])
                + 0.5 * np.sum(oos_perf == oos_perf[best])) / p
        if rank < 0.5:
            below_median += 1
    return float(below_median / len(combos))
