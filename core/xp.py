"""core/xp.py — array-backend abstraction (CuPy if a CUDA GPU is present, else NumPy).

Every numeric kernel imports the ``xp`` namespace from here instead of importing
``numpy``/``cupy`` directly, so the exact same code runs on GPU or CPU. The backend is
detected once at import time and logged; it can be forced via the ``COMPUTE_BACKEND``
environment variable ("auto" | "cpu" | "gpu"). Detection degrades gracefully: if CuPy
is missing or no GPU/driver is present, we fall back to NumPy silently (a warning is
logged only when a GPU backend was explicitly forced but is unavailable).

Public symbols:
    xp            active array namespace (cupy or numpy)
    backend       "cupy" or "numpy"
    is_gpu        True iff backend == "cupy"
    GPU_AVAILABLE True iff a CUDA GPU was detected and selected
    FLOAT_DTYPE   float64 (chosen for cross-backend determinism)
    asnumpy(a)    copy any array to a plain numpy array (for logging/IO)
    to_backend(a) move a numpy array onto the active backend
    seed(n)       seed the active backend's global RNG for deterministic runs
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as _np

__all__ = [
    "xp",
    "backend",
    "is_gpu",
    "GPU_AVAILABLE",
    "CUDA_AVAILABLE",
    "FLOAT_DTYPE",
    "asnumpy",
    "to_backend",
    "seed",
    "get_backend_info",
]

logger = logging.getLogger("gt.core.xp")

# float64 throughout for cross-backend determinism (float32 is faster but less exact).
FLOAT_DTYPE = "float64"

_forced = os.environ.get("COMPUTE_BACKEND", "auto").strip().lower()


def _try_cupy() -> Any:
    """Import CuPy and confirm at least one CUDA device is usable.

    Returns the cupy module, or None if cupy is missing / no GPU / driver absent.
    """
    try:
        import cupy  # type: ignore

        if cupy.cuda.runtime.getDeviceCount() <= 0:
            return None
        return cupy
    except Exception:
        return None


_cp: Any = None
xp: Any = _np
backend: str = "numpy"
GPU_AVAILABLE: bool = False
CUDA_AVAILABLE: bool = False

if _forced == "cpu":
    backend = "numpy"
    xp = _np
    logger.info("compute backend: numpy (forced)")
elif _forced == "gpu":
    _cp = _try_cupy()
    if _cp is not None:
        backend = "cupy"
        xp = _cp
        GPU_AVAILABLE = True
        CUDA_AVAILABLE = True
        logger.info("compute backend: cupy (forced)")
    else:
        logger.warning("COMPUTE_BACKEND=gpu requested but CuPy/CUDA unavailable; using numpy")
        backend = "numpy"
        xp = _np
else:  # auto
    _cp = _try_cupy()
    if _cp is not None:
        backend = "cupy"
        xp = _cp
        GPU_AVAILABLE = True
        CUDA_AVAILABLE = True
        logger.info("compute backend: cupy (CUDA GPU detected)")
    else:
        backend = "numpy"
        xp = _np
        logger.info("compute backend: numpy (no CUDA GPU detected)")

is_gpu: bool = backend == "cupy"


def asnumpy(a: Any) -> _np.ndarray:
    """Return ``a`` as a plain numpy array (copy if on GPU)."""
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return _np.asarray(a)


def to_backend(a: Any, dtype: Any = FLOAT_DTYPE) -> Any:
    """Move a numpy array onto the active backend (returns ``a`` unchanged if already
    there)."""
    arr = _np.asarray(a, dtype=dtype)
    if _cp is not None:
        return _cp.asarray(arr)
    return arr


def seed(n: int) -> None:
    """Seed the active backend's global RNG (deterministic runs)."""
    xp.random.seed(n)


def get_backend_info() -> dict:
    """Small diagnostic dict (used by the demo/UI badge)."""
    info = {"backend": backend, "gpu": GPU_AVAILABLE, "cuda": CUDA_AVAILABLE}
    if _cp is not None and GPU_AVAILABLE:
        try:
            info["device"] = _cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        except Exception:
            info["device"] = "unknown"
    return info
