"""core — numerical + simulation engine for the SOL-PERP Evolutionary Trading System.

Sub-modules (see DEEPSEEK_BUILD_PROMPT.md §1):
  xp           — array-backend abstraction (CuPy if CUDA present, else NumPy)
  data         — data loader + synthetic generation
  indicators   — GPU/CPU indicator kernels
  strategy     — genome -> signals
  backtester   — GPU-vectorized population backtest
  metrics      — Sharpe/Sortino/DD/WFE + robust fitness
  evolution    — GA + Walk-Forward Optimization loop
"""
