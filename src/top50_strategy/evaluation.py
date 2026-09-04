"""Evaluation metrics, walk-forward time splits, and M0-M7 ablation testing."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from top50_strategy.types import DatasetSplit


@dataclass(frozen=True)
class EnableDecision:
    enabled: bool
    reason: str


def make_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_years: int = 2,
    valid_months: int = 6,
    test_months: int = 6,
) -> list[DatasetSplit]:
    """Generates rolling non-overlapping walk-forward splits."""
    dates_sorted = dates.sort_values()
    start_year = dates_sorted.min().year
    end_year = dates_sorted.max().year

    splits: list[DatasetSplit] = []
    curr_train_start = dates_sorted.min()

    # Move forward in steps of test_months
    step_days = int(test_months * 30.4)
    train_days = int(train_years * 365.25)
    valid_days = int(valid_months * 30.4)

    total_days = (dates_sorted.max() - dates_sorted.min()).days
    if total_days < train_days + valid_days + step_days:
        # Fallback to single split proportionally if period is short
        n = len(dates_sorted)
        n_tr = int(n * 0.6)
        n_val = int(n * 0.2)
        tr = dates_sorted[:n_tr]
        val = dates_sorted[n_tr : n_tr + n_val]
        te = dates_sorted[n_tr + n_val :]
        if len(tr) > 0 and len(val) > 0 and len(te) > 0:
            splits.append(DatasetSplit(train=tr, valid=val, test=te))
        return splits

    cursor = curr_train_start
    while True:
        t_tr_end = cursor + pd.Timedelta(days=train_days)
        t_val_end = t_tr_end + pd.Timedelta(days=valid_days)
        t_test_end = t_val_end + pd.Timedelta(days=step_days)

        if t_test_end > dates_sorted.max():
            # If remaining test set is large enough, append final split
            if t_val_end < dates_sorted.max():
                tr = dates_sorted[(dates_sorted >= cursor) & (dates_sorted < t_tr_end)]
                val = dates_sorted[(dates_sorted >= t_tr_end) & (dates_sorted < t_val_end)]
                te = dates_sorted[dates_sorted >= t_val_end]
                if len(tr) > 10 and len(val) > 5 and len(te) > 5:
                    splits.append(DatasetSplit(train=tr, valid=val, test=te))
            break

        tr = dates_sorted[(dates_sorted >= cursor) & (dates_sorted < t_tr_end)]
        val = dates_sorted[(dates_sorted >= t_tr_end) & (dates_sorted < t_val_end)]
        te = dates_sorted[(dates_sorted >= t_val_end) & (dates_sorted <= t_test_end)]

        if len(tr) > 10 and len(val) > 5 and len(te) > 5:
            splits.append(DatasetSplit(train=tr, valid=val, test=te))

        cursor = cursor + pd.Timedelta(days=step_days)

    return splits


def calculate_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> dict[str, float]:
    """Calculates comprehensive quantitative portfolio metrics."""
    s = returns.dropna()
    if len(s) == 0:
        return {
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
        }

    cum = (1.0 + s).cumprod()
    n_days = len(s)
    ann_factor = 252.0 / max(n_days, 1)

    final_nav = float(cum.iloc[-1])
    cagr = (final_nav ** ann_factor) - 1.0 if final_nav > 0 else -1.0

    vol = float(s.std() * np.sqrt(252.0))
    sharpe = float((cagr - risk_free_rate) / (vol + 1e-8))

    downside = s[s < 0]
    downside_vol = float(downside.std() * np.sqrt(252.0)) if len(downside) > 1 else 1e-6
    sortino = float((cagr - risk_free_rate) / (downside_vol + 1e-8))

    cummax = cum.cummax()
    drawdown = (cum - cummax) / (cummax + 1e-8)
    max_dd = float(drawdown.min())
    calmar = float(cagr / (abs(max_dd) + 1e-8))

    win_rate = float(np.mean(s > 0))

    return {
        "annual_return": float(cagr),
        "annual_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "final_nav": final_nav,
    }


def should_enable(
    candidate_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    min_sharpe_delta: float = 0.02,
    max_mdd_degradation: float = 0.05,
) -> EnableDecision:
    """Ablation enablement gate: enforces genuine out-of-sample risk-adjusted value."""
    cand_sr = candidate_metrics.get("sharpe", 0.0)
    base_sr = baseline_metrics.get("sharpe", 0.0)

    cand_mdd = candidate_metrics.get("max_drawdown", 0.0)
    base_mdd = baseline_metrics.get("max_drawdown", 0.0)

    # Check if MaxDD degrades by more than acceptable degradation (e.g. from -0.10 to -0.30)
    if cand_mdd < base_mdd - max_mdd_degradation:
        return EnableDecision(
            enabled=False,
            reason=f"Rejected: Max drawdown degraded excessively ({cand_mdd:.2%} vs base {base_mdd:.2%})",
        )

    if cand_sr < base_sr + min_sharpe_delta:
        return EnableDecision(
            enabled=False,
            reason=f"Rejected: Sharpe improvement {cand_sr - base_sr:.2f} < threshold {min_sharpe_delta:.2f}",
        )

    return EnableDecision(
        enabled=True,
        reason=f"Approved: Sharpe improved from {base_sr:.2f} to {cand_sr:.2f} with controlled drawdown ({cand_mdd:.2%})",
    )