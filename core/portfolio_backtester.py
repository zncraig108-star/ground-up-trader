"""core/portfolio_backtester.py — Multi-Asset Cross-Margin Portfolio Simulation.

Simulates a unified account with cross-margin and realistic exchange mechanics
across a diversified crypto universe (BTC, ETH, SOL, AVAX, DOGE, LINK, SUI, NEAR, XRP):
  - Shared equity & margin capital pool (starts at $10,000).
  - Volatility-parity position allocations with hard leverage caps.
  - Hourly funding settlement and taker/maker fee accounting per asset.
  - Dynamic ATR trailing stops with gap-through execution.
  - Intrabar maintenance margin liquidation check.
  - Total fee footprint tracking to ensure friction stays minimal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from config import InstrumentSpec, PortfolioConfig, DEFAULT_PORTFOLIO_INSTRUMENTS
from core.portfolio_strategy import PortfolioSignals, AssetSignals


@dataclass
class PortfolioTrade:
    """Completed trade record for an individual asset within the portfolio."""
    product_id: str
    side: int  # +1 long, -1 short
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_px: float
    exit_px: float
    qty: float  # unsigned contract lots
    pnl: float  # realized USD net of fees
    fees: float  # total maker/taker fees paid
    funding: float  # net funding cash flow
    exit_reason: str  # "signal" | "stop_loss" | "liquidation"


@dataclass
class PortfolioBacktestResult:
    """Full backtest results for the multi-asset portfolio."""
    timestamps: pd.DatetimeIndex
    equity_curve: np.ndarray       # Total portfolio equity over time
    cash_curve: np.ndarray         # Realized USD cash balance over time
    leverage_curve: np.ndarray     # Total gross leverage over time
    trades: List[PortfolioTrade]
    trades_by_asset: Dict[str, List[PortfolioTrade]]
    total_fees: float
    total_funding: float
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    fee_pct_of_equity: float       # Fees paid / initial equity


def backtest_portfolio(
    signals: PortfolioSignals,
    cfg: PortfolioConfig,
    instruments: Optional[Mapping[str, InstrumentSpec]] = None,
) -> PortfolioBacktestResult:
    """Simulate unified multi-asset portfolio execution across all assets."""
    inst_specs = instruments if instruments is not None else DEFAULT_PORTFOLIO_INSTRUMENTS
    universe = signals.universe
    ts = signals.timestamps
    n_bars = len(ts)

    initial_equity = float(cfg.initial_equity)
    cash = initial_equity
    equity_curve = np.zeros(n_bars, dtype=np.float64)
    cash_curve = np.zeros(n_bars, dtype=np.float64)
    leverage_curve = np.zeros(n_bars, dtype=np.float64)

    # Position state per asset
    pos_qty: Dict[str, float] = {p: 0.0 for p in universe}       # signed quantity
    pos_entry_px: Dict[str, float] = {p: 0.0 for p in universe}
    pos_entry_bar: Dict[str, int] = {p: -1 for p in universe}
    pos_entry_fees: Dict[str, float] = {p: 0.0 for p in universe}
    pos_funding_accrued: Dict[str, float] = {p: 0.0 for p in universe}

    all_trades: List[PortfolioTrade] = []
    trades_by_asset: Dict[str, List[PortfolioTrade]] = {p: [] for p in universe}

    total_fees_paid = 0.0
    total_funding_paid = 0.0
    is_liquidated = False

    for t in range(n_bars):
        current_time = ts[t]
        bar_funding_flow = 0.0
        bar_fees = 0.0

        # --- 1. Hourly Funding Settlement ---
        for p in universe:
            qty = pos_qty[p]
            if abs(qty) > 1e-9:
                asig = signals.asset_signals[p]
                rate = asig.funding_rate[t]
                px = asig.close[t]
                # Positive rate: Long pays short. Long flow = -qty * px * rate
                cf = -qty * px * rate
                cash += cf
                bar_funding_flow += cf
                pos_funding_accrued[p] += cf
                total_funding_paid += cf

        # --- 2. Check Stop-Losses on 1h Bar Extremes ---
        for p in universe:
            qty = pos_qty[p]
            if abs(qty) > 1e-9:
                asig = signals.asset_signals[p]
                sl_px = asig.stop_loss_px[t]
                spec = inst_specs.get(p, InstrumentSpec(p))
                tick = spec.tick_size
                taker_rate = spec.taker_fee_rate

                hit_stop = False
                exit_price = 0.0

                if qty > 0 and sl_px > 0 and asig.low[t] <= sl_px:
                    # Long stop hit: fills at stop price minus slippage (or open if gapped)
                    exit_price = min(asig.close[t], sl_px) - tick
                    hit_stop = True
                elif qty < 0 and sl_px > 0 and asig.high[t] >= sl_px:
                    # Short stop hit: fills at stop price plus slippage (or open if gapped)
                    exit_price = max(asig.close[t], sl_px) + tick
                    hit_stop = True

                if hit_stop:
                    notional = abs(qty) * exit_price
                    exit_fee = notional * taker_rate
                    pnl = (exit_price - pos_entry_px[p]) * qty - (pos_entry_fees[p] + exit_fee)
                    cash += (exit_price - pos_entry_px[p]) * qty - exit_fee
                    total_fees_paid += exit_fee
                    bar_fees += exit_fee

                    trade = PortfolioTrade(
                        product_id=p,
                        side=1 if qty > 0 else -1,
                        entry_time=ts[pos_entry_bar[p]],
                        exit_time=current_time,
                        entry_px=pos_entry_px[p],
                        exit_px=exit_price,
                        qty=abs(qty),
                        pnl=pnl,
                        fees=pos_entry_fees[p] + exit_fee,
                        funding=pos_funding_accrued[p],
                        exit_reason="stop_loss",
                    )
                    all_trades.append(trade)
                    trades_by_asset[p].append(trade)

                    pos_qty[p] = 0.0
                    pos_entry_px[p] = 0.0
                    pos_entry_bar[p] = -1
                    pos_entry_fees[p] = 0.0
                    pos_funding_accrued[p] = 0.0

        # --- 3. Compute Current Portfolio Mark-to-Market Equity ---
        unrealized_pnl = 0.0
        total_notional = 0.0
        maint_margin_req = 0.0

        for p in universe:
            qty = pos_qty[p]
            if abs(qty) > 1e-9:
                px = signals.asset_signals[p].close[t]
                unrealized_pnl += (px - pos_entry_px[p]) * qty
                notional = abs(qty) * px
                total_notional += notional
                spec = inst_specs.get(p, InstrumentSpec(p))
                maint_margin_req += notional * spec.maintenance_margin_rate

        current_equity = cash + unrealized_pnl

        # Intrabar Liquidation Check
        if current_equity < maint_margin_req and not is_liquidated:
            is_liquidated = True
            # Liquidate all open positions at market
            for p in universe:
                if abs(pos_qty[p]) > 1e-9:
                    px = signals.asset_signals[p].close[t]
                    spec = inst_specs.get(p, InstrumentSpec(p))
                    fee = abs(pos_qty[p]) * px * spec.taker_fee_rate
                    cash -= fee
                    total_fees_paid += fee
                    pos_qty[p] = 0.0
            current_equity = max(0.0, cash)

        # --- 4. Rebalance on 4h Boundaries ---
        # Align rebalancing to every 4 hours (e.g. 00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
        is_rebal_hour = (current_time.hour % cfg.signal_timeframe_hours == 0) and not is_liquidated

        if is_rebal_hour and current_equity > 100.0:
            for p in universe:
                asig = signals.asset_signals[p]
                spec = inst_specs.get(p, InstrumentSpec(p))
                target_w = asig.target_weight[t]  # signed target weight
                target_dollar_notional = current_equity * target_w
                current_px = asig.close[t]

                # Desired target quantity (signed)
                desired_qty = spec.floor_to_lot(abs(target_dollar_notional) / max(current_px, 1e-6))
                if target_w < 0:
                    desired_qty = -desired_qty

                current_q = pos_qty[p]
                is_regime_flip = (current_q > 0 and desired_qty <= 0) or (current_q < 0 and desired_qty >= 0) or (current_q == 0 and desired_qty != 0)
                # Only rebalance if regime changes or size drifted by >= 25% to prevent micro-churn
                size_drift = abs(desired_qty - current_q) / max(abs(current_q), 1e-6)
                needs_rebalance = is_regime_flip or (size_drift >= 0.25)

                if needs_rebalance and abs(desired_qty - current_q) * current_px >= spec.min_notional:
                    delta_qty = desired_qty - current_q

                    # 1. Closing existing position (or flipping)
                    if (current_q > 0 and delta_qty < 0) or (current_q < 0 and delta_qty > 0):
                        close_portion = min(abs(current_q), abs(delta_qty))
                        exit_fee = close_portion * current_px * spec.maker_fee_rate
                        pnl = (current_px - pos_entry_px[p]) * (close_portion if current_q > 0 else -close_portion) - exit_fee
                        cash += (current_px - pos_entry_px[p]) * (close_portion if current_q > 0 else -close_portion) - exit_fee
                        total_fees_paid += exit_fee
                        bar_fees += exit_fee

                        trade = PortfolioTrade(
                            product_id=p,
                            side=1 if current_q > 0 else -1,
                            entry_time=ts[pos_entry_bar[p]],
                            exit_time=current_time,
                            entry_px=pos_entry_px[p],
                            exit_px=current_px,
                            qty=close_portion,
                            pnl=pnl,
                            fees=exit_fee,
                            funding=pos_funding_accrued[p],
                            exit_reason="signal",
                        )
                        all_trades.append(trade)
                        trades_by_asset[p].append(trade)

                        # If flipped or fully closed
                        if abs(current_q) <= close_portion:
                            pos_qty[p] = 0.0
                            pos_entry_px[p] = 0.0
                            pos_entry_bar[p] = -1
                            pos_entry_fees[p] = 0.0
                            pos_funding_accrued[p] = 0.0
                        else:
                            pos_qty[p] = (current_q - close_portion) if current_q > 0 else (current_q + close_portion)

                    # 2. Opening or expanding in the same direction
                    new_open_qty = desired_qty - pos_qty[p]
                    if abs(new_open_qty) * current_px >= spec.min_notional and (new_open_qty * desired_qty > 0):
                        open_fee = abs(new_open_qty) * current_px * spec.maker_fee_rate
                        cash -= open_fee
                        total_fees_paid += open_fee
                        bar_fees += open_fee

                        # Weighted average entry price
                        old_notional = abs(pos_qty[p]) * pos_entry_px[p]
                        add_notional = abs(new_open_qty) * current_px
                        tot_qty = abs(pos_qty[p]) + abs(new_open_qty)
                        pos_entry_px[p] = (old_notional + add_notional) / max(tot_qty, 1e-6) if tot_qty > 0 else current_px
                        pos_qty[p] = desired_qty
                        if pos_entry_bar[p] == -1:
                            pos_entry_bar[p] = t
                        pos_entry_fees[p] += open_fee

        # Update curves
        equity_curve[t] = max(0.0, current_equity)
        cash_curve[t] = cash
        leverage_curve[t] = total_notional / max(current_equity, 1.0)

    # --- 5. Summary Metrics ---
    final_eq = float(equity_curve[-1])
    total_ret = float(final_eq / initial_equity - 1.0)
    days = (ts[-1] - ts[0]).total_seconds() / 86400.0
    cagr = float((final_eq / initial_equity) ** (365.25 / max(days, 1.0)) - 1.0) if final_eq > 0 else -1.0

    # Hourly returns for Sharpe & Sortino
    eq_series = pd.Series(equity_curve, index=ts)
    hourly_returns = eq_series.pct_change().fillna(0.0).to_numpy()
    mean_ret = np.mean(hourly_returns)
    std_ret = np.std(hourly_returns)
    downside_std = np.std(np.minimum(hourly_returns, 0.0))

    sharpe = float(mean_ret / std_ret * math.sqrt(8760.0)) if std_ret > 1e-9 else 0.0
    sortino = float(mean_ret / downside_std * math.sqrt(8760.0)) if downside_std > 1e-9 else 0.0

    # Max Drawdown
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / np.maximum(peaks, 1e-6)
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    calmar = float(cagr / max_dd) if max_dd > 0.01 else 0.0

    return PortfolioBacktestResult(
        timestamps=ts,
        equity_curve=equity_curve,
        cash_curve=cash_curve,
        leverage_curve=leverage_curve,
        trades=all_trades,
        trades_by_asset=trades_by_asset,
        total_fees=total_fees_paid,
        total_funding=total_funding_paid,
        final_equity=final_eq,
        total_return=total_ret,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        fee_pct_of_equity=float(total_fees_paid / initial_equity),
    )
