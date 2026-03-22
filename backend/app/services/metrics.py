"""Performance metrics for market making simulation."""

import numpy as np

from app.models.simulation import TickRecord


def compute_sharpe_ratio(pnl_series: list[float], risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio from PnL time series.

    Computes returns as differences, then mean/std.
    Returns 0.0 if std is zero (division protection).
    """
    if len(pnl_series) < 2:
        return 0.0

    returns = np.diff(pnl_series)
    excess = returns - risk_free_rate
    std = float(np.std(excess))
    if std == 0.0:
        return 0.0
    return float(np.mean(excess) / std)


def compute_max_drawdown(pnl_series: list[float]) -> float:
    """
    Maximum drawdown as a fraction (0 to 1).

    Handles all-negative PnL series correctly.
    """
    if len(pnl_series) < 2:
        return 0.0

    arr = np.array(pnl_series)
    peak = np.maximum.accumulate(arr)

    # Avoid division by zero when peak is zero or negative
    drawdown = 0.0
    for i in range(len(arr)):
        if peak[i] > 0:
            dd = (peak[i] - arr[i]) / peak[i]
            drawdown = max(drawdown, dd)
        elif peak[i] == 0 and arr[i] < 0:
            # Started from 0 and went negative — treat as 100% or absolute
            drawdown = max(drawdown, 1.0)

    return float(drawdown)


def compute_inventory_stats(inventory_series: list[float]) -> dict:
    """Inventory statistics: mean, std, max_abs, zero_crossings."""
    if not inventory_series:
        return {"mean": 0.0, "std": 0.0, "max_abs": 0.0, "zero_crossings": 0}

    arr = np.array(inventory_series)
    signs = np.sign(arr)
    # Count zero crossings (sign changes)
    crossings = int(np.sum(np.abs(np.diff(signs)) > 0))

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "max_abs": float(np.max(np.abs(arr))),
        "zero_crossings": crossings,
    }


def compute_spread_stats(spread_series: list[float]) -> dict:
    """Spread statistics: mean, std, min, max."""
    if not spread_series:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(spread_series)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def generate_report(tick_records: list[TickRecord]) -> dict:
    """Generate full performance report from tick records."""
    if not tick_records:
        return {}

    pnl_series = [r.mm_pnl for r in tick_records]
    inventory_series = [r.mm_inventory for r in tick_records]
    spread_series = [r.spread for r in tick_records if r.spread > 0]
    total_trades = sum(r.num_trades for r in tick_records)

    return {
        "total_ticks": len(tick_records),
        "total_trades": total_trades,
        "mm_final_pnl": pnl_series[-1],
        "mm_final_inventory": inventory_series[-1],
        "sharpe_ratio": compute_sharpe_ratio(pnl_series),
        "max_drawdown": compute_max_drawdown(pnl_series),
        "inventory_stats": compute_inventory_stats(inventory_series),
        "spread_stats": compute_spread_stats(spread_series),
    }
