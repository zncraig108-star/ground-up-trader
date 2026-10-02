"""core/indicators.py — GPU/CPU indicator kernels (batched over the population axis).

Every kernel follows the same convention as the backtester: **vectorized over the
population axis ``P``, sequential over the time axis ``N``**. Indicator values are
path-dependent (exponential smoothing / running bands).

Two execution paths give identical numbers:
  * **Numba-JIT** (``@njit(parallel=True)`` over ``P``) when numba is installed and the
    NumPy backend is active — the time loop is compiled, which removes the Python
    interpreter overhead that dominates signal generation.
  * **xp fallback** — a NumPy/CuPy time loop used on the GPU backend or without numba.

Progress callbacks are emitted per time step on the fallback path; the compiled path
emits only start/end (a per-step host callback inside a parallel kernel would dominate).

Warm-up convention (documented ASSUMPTION): smoothing recursions are seeded with the
first bar's value (``EMA_0 = close_0``, ``ATR_0 = TR_0``) rather than a look-back SMA.
This keeps the variable-period batch fully vectorizable and remains deterministic; the
warm-up bias is negligible relative to the thousands of bars the optimizer sees.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.xp import backend, xp

__all__ = [
    "ema_batch",
    "atr_batch",
    "supertrend_batch",
    "adx",
    "rsi",
]

try:  # pragma: no cover - depends on optional dependency
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore
        def _deco(func):
            return func
        return _deco

    prange = range  # type: ignore

# Toggle used by the demo's parity check to force the reference path.
USE_NUMBA = True

_FLOAT = xp.float64


@njit(parallel=True, cache=True, fastmath=False)
def _ema_kernel(values: np.ndarray, periods: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    p = periods.shape[0]
    out = np.empty((p, n), dtype=np.float64)
    for pi in prange(p):
        alpha = 2.0 / (periods[pi] + 1.0)
        prev = values[0]
        out[pi, 0] = prev
        for t in range(1, n):
            prev = prev + alpha * (values[t] - prev)
            out[pi, t] = prev
    return out


@njit(parallel=True, cache=True, fastmath=False)
def _atr_kernel(tr: np.ndarray, periods: np.ndarray) -> np.ndarray:
    n = tr.shape[0]
    p = periods.shape[0]
    out = np.empty((p, n), dtype=np.float64)
    for pi in prange(p):
        alpha = 1.0 / periods[pi]
        prev = tr[0]
        out[pi, 0] = prev
        for t in range(1, n):
            prev = prev + alpha * (tr[t] - prev)
            out[pi, t] = prev
    return out


@njit(parallel=True, cache=True, fastmath=False)
def _st_kernel(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               mults: np.ndarray, atr: np.ndarray) -> np.ndarray:
    p, n = atr.shape
    mid = (high + low) / 2.0
    trend = np.empty((p, n), dtype=np.int8)
    for pi in prange(p):
        m = mults[pi]
        prev_ub = mid[0] + m * atr[pi, 0]
        prev_lb = mid[0] - m * atr[pi, 0]
        prev_trend = 1
        trend[pi, 0] = 1
        for t in range(1, n):
            basic_ub = mid[t] + m * atr[pi, t]
            basic_lb = mid[t] - m * atr[pi, t]
            if basic_ub < prev_ub or close[t - 1] > prev_ub:
                prev_ub = basic_ub
            if basic_lb > prev_lb or close[t - 1] < prev_lb:
                prev_lb = basic_lb
            if prev_trend == 1:
                if close[t] < prev_lb:
                    prev_trend = -1
            else:
                if close[t] > prev_ub:
                    prev_trend = 1
            trend[pi, t] = prev_trend
    return trend


def _vec(a: Any) -> Any:
    """Coerce ``a`` to a 1-D float64 array on the active backend."""
    return xp.asarray(a, dtype=_FLOAT)


def _periods(p: Any) -> Any:
    """Coerce a period array to a 1-D float64 array (scalar -> (1,))."""
    p = xp.asarray(p, dtype=_FLOAT)
    return p.reshape(-1)


def _use_compiled() -> bool:
    return NUMBA_AVAILABLE and USE_NUMBA and backend == "numpy"


def _tick(progress_cb: Any, progress_every: int, t: int, n: int) -> None:
    """Fire an optional ``progress_cb(t, n)`` every ``progress_every`` time steps."""
    if progress_cb is not None and progress_every > 0 and (t % progress_every == 0):
        progress_cb(t, n)


def _true_range(high: Any, low: Any, close: Any) -> Any:
    """Vectorized True Range: ``max(high-low, |high-prev_close|, |low-prev_close|)``."""
    n = high.shape[0]
    hl = high - low
    tr = xp.empty(n, dtype=_FLOAT)
    tr[0] = hl[0]
    if n > 1:
        hc = xp.abs(high[1:] - close[:-1])
        lc = xp.abs(low[1:] - close[:-1])
        tr[1:] = xp.maximum(xp.maximum(hl[1:], hc), lc)
    return tr


def ema_batch(values: Any, period: Any, progress_cb: Any = None,
              progress_every: int = 0) -> Any:
    """Exponential moving average, batched over ``P`` periods.

    Args:
        values: 1-D price series, shape ``(N,)``.
        period: EMA period(s), scalar or shape ``(P,)`` (non-integer allowed).
        progress_cb: optional ``cb(done, total)`` for live progress.
        progress_every: time-step interval for ``progress_cb``.

    Returns:
        ``(P, N)`` float64 array where ``out[p]`` is the EMA of ``values`` with
        ``period[p]`` (``alpha = 2 / (period + 1)``). Seeded at ``values[0]``.
    """
    values = _vec(values)
    period = _periods(period)
    n = values.shape[0]
    p = period.shape[0]

    if n == 0:
        return xp.empty((p, 0), dtype=_FLOAT)

    if _use_compiled():
        v = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        per = np.ascontiguousarray(np.asarray(period, dtype=np.float64))
        out = _ema_kernel(v, per)
        if progress_cb is not None and progress_every > 0:
            progress_cb(n, n)
        return out

    alpha = 2.0 / (period + 1.0)  # (P,)
    out = xp.empty((p, n), dtype=_FLOAT)
    out[:, 0] = values[0]
    for t in range(1, n):
        _tick(progress_cb, progress_every, t, n)
        out[:, t] = out[:, t - 1] + alpha * (values[t] - out[:, t - 1])
    return out


def atr_batch(high: Any, low: Any, close: Any, period: Any, progress_cb: Any = None,
              progress_every: int = 0) -> Any:
    """Wilder Average True Range, batched over ``P`` periods.

    Args:
        high, low, close: 1-D price series, shape ``(N,)``.
        period: ATR period(s), scalar or shape ``(P,)`` (non-integer allowed).
        progress_cb: optional ``cb(done, total)`` for live progress.
        progress_every: time-step interval for ``progress_cb``.

    Returns:
        ``(P, N)`` float64 array. Smoothing uses ``alpha = 1 / period`` seeded at
        ``TR_0``; True Range is ``max(high-low, |high-prev_close|, |low-prev_close|)``.
    """
    high = _vec(high)
    low = _vec(low)
    close = _vec(close)
    period = _periods(period)
    n = high.shape[0]
    p = period.shape[0]

    if n == 0:
        return xp.empty((p, 0), dtype=_FLOAT)

    tr = _true_range(high, low, close)

    if _use_compiled():
        tr_np = np.ascontiguousarray(np.asarray(tr, dtype=np.float64))
        per = np.ascontiguousarray(np.asarray(period, dtype=np.float64))
        out = _atr_kernel(tr_np, per)
        if progress_cb is not None and progress_every > 0:
            progress_cb(n, n)
        return out

    alpha = 1.0 / period  # (P,)  Wilder smoothing
    out = xp.empty((p, n), dtype=_FLOAT)
    out[:, 0] = tr[0]
    for t in range(1, n):
        _tick(progress_cb, progress_every, t, n)
        out[:, t] = out[:, t - 1] + alpha * (tr[t] - out[:, t - 1])
    return out


def supertrend_batch(high: Any, low: Any, close: Any, period: Any, mult: Any,
                     progress_cb: Any = None, progress_every: int = 0) -> Any:
    """Supertrend direction, batched over ``P`` parameter sets.

    Uses the standard basic-band formulation with mid-price ``(high+low)/2`` and ATR
    bands ``mid +/- mult * ATR(period)``. Final bands only widen (never contract), and
    the trend flips when price closes through the opposite band.

    Args:
        high, low, close: 1-D price series, shape ``(N,)``.
        period: ATR period(s), scalar or shape ``(P,)``.
        mult: band multiplier(s), scalar or shape ``(P,)``.
        progress_cb: optional ``cb(done, total)`` for live progress.
        progress_every: time-step interval for ``progress_cb``.

    Returns:
        ``(P, N)`` int8 array of ``+1`` (bullish) / ``-1`` (bearish). The first bar is
        arbitrarily seeded bullish (ASSUMPTION).
    """
    high = _vec(high)
    low = _vec(low)
    close = _vec(close)
    period = _periods(period)
    mult = _periods(mult)
    n = high.shape[0]
    p = period.shape[0]

    if n == 0:
        return xp.zeros((p, 0), dtype=xp.int8)

    tr = _true_range(high, low, close)

    if _use_compiled():
        tr_np = np.ascontiguousarray(np.asarray(tr, dtype=np.float64))
        per = np.ascontiguousarray(np.asarray(period, dtype=np.float64))
        mult_np = np.ascontiguousarray(np.asarray(mult, dtype=np.float64))
        h = np.ascontiguousarray(np.asarray(high, dtype=np.float64))
        lo = np.ascontiguousarray(np.asarray(low, dtype=np.float64))
        c = np.ascontiguousarray(np.asarray(close, dtype=np.float64))
        atr = _atr_kernel(tr_np, per)
        out = _st_kernel(h, lo, c, mult_np, atr)
        if progress_cb is not None and progress_every > 0:
            progress_cb(n, n)
        return out

    atr = atr_batch(high, low, close, period)  # (P, N)
    mid = (high + low) / 2.0

    basic_ub = mid[None, :] + mult[:, None] * atr  # (P, N)
    basic_lb = mid[None, :] - mult[:, None] * atr

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    trend = xp.zeros((p, n), dtype=xp.int8)
    trend[:, 0] = 1  # ASSUMPTION: seed first bar bullish

    for t in range(1, n):
        _tick(progress_cb, progress_every, t, n)
        # Upper band stays (widens) unless the basic band moves higher or price
        # closed above the previous band.
        widen_ub = (basic_ub[:, t] < final_ub[:, t - 1]) | (close[t - 1] > final_ub[:, t - 1])
        final_ub[:, t] = xp.where(widen_ub, basic_ub[:, t], final_ub[:, t - 1])

        widen_lb = (basic_lb[:, t] > final_lb[:, t - 1]) | (close[t - 1] < final_lb[:, t - 1])
        final_lb[:, t] = xp.where(widen_lb, basic_lb[:, t], final_lb[:, t - 1])

        prev = trend[:, t - 1]
        flip_down = (prev == 1) & (close[t] < final_lb[:, t])
        flip_up = (prev == -1) & (close[t] > final_ub[:, t])
        new_trend = xp.where(flip_down, -1, prev)
        trend[:, t] = xp.where(flip_up, 1, new_trend)

    return trend


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's Average Directional Index (ADX), causal, single series.

    Computes +DI/-DI/DX via Wilder smoothing then smooths DX into ADX.
    Returns an ``(N,)`` float64 array; leading ``~2 * period`` bars are near 0
    during warm-up, which naturally suppresses entries until the indicator stabilizes.

    Args:
        high: ``(N,)`` bar highs.
        low: ``(N,)`` bar lows.
        close: ``(N,)`` bar closes (previous close used for True Range).
        period: Wilder smoothing period (default 14).
    """
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if n < period + 2:
        return out
    alpha = 1.0 / period
    tr_s = dmp_s = dmn_s = adx_s = 0.0
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        dm_up = high[i] - high[i - 1]
        dm_dn = low[i - 1] - low[i]
        dmp = dm_up if dm_up > dm_dn and dm_up > 0 else 0.0
        dmn = dm_dn if dm_dn > dm_up and dm_dn > 0 else 0.0
        # Wilder exponential smoothing (equivalent to EMA with alpha = 1/period)
        tr_s += alpha * (tr - tr_s)
        dmp_s += alpha * (dmp - dmp_s)
        dmn_s += alpha * (dmn - dmn_s)
        if i < period:
            continue
        dip = 100.0 * dmp_s / tr_s if tr_s > 1e-12 else 0.0
        din = 100.0 * dmn_s / tr_s if tr_s > 1e-12 else 0.0
        dx = 100.0 * abs(dip - din) / (dip + din) if (dip + din) > 1e-12 else 0.0
        adx_s += alpha * (dx - adx_s)
        out[i] = adx_s
    return out


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI, causal, single series.

    Returns an ``(N,)`` float64 array. Leading ``period`` bars default to 50.0
    (neutral) so they pass the overbought/oversold gate and do not suppress entries
    during warm-up — ADX and ER gates cover the warm-up period adequately.

    Args:
        close: ``(N,)`` bar closes.
        period: Wilder smoothing period (default 14).
    """
    n = len(close)
    out = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return out
    alpha = 1.0 / period
    gain_s = loss_s = 0.0
    for i in range(1, n):
        d = close[i] - close[i - 1]
        gain_s += alpha * (max(d, 0.0) - gain_s)
        loss_s += alpha * (max(-d, 0.0) - loss_s)
        if i < period:
            continue
        rs = gain_s / loss_s if loss_s > 1e-12 else 1e12
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out
