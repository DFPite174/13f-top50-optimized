import numpy as np
import pandas as pd
import pytest

from top50_strategy.evaluation import (
    make_walk_forward_splits,
    calculate_metrics,
    should_enable,
    EnableDecision,
)
from top50_strategy.types import DatasetSplit


def test_walk_forward_splits_never_overlap_or_reverse_time():
    dates = pd.bdate_range("2018-01-01", "2024-12-31", tz="UTC")
    splits = make_walk_forward_splits(dates, train_years=2, valid_months=6, test_months=6)
    assert len(splits) >= 2
    for split in splits:
        assert split.train.max() < split.valid.min()
        assert split.valid.max() < split.test.min()


def test_metrics_include_required_fields():
    # 252 days of normal returns
    rets = pd.Series(np.random.RandomState(42).normal(0.0008, 0.01, 252))
    m = calculate_metrics(rets)
    required = {"annual_return", "annual_volatility", "sharpe", "max_drawdown", "calmar", "win_rate"}
    assert required.issubset(m.keys())
    assert m["max_drawdown"] <= 0.0


def test_candidate_with_excessive_drawdown_is_not_enabled():
    base_m = {"sharpe": 1.2, "annual_return": 0.15, "max_drawdown": -0.10, "turnover": 0.10}
    bad_candidate = {"sharpe": 1.25, "annual_return": 0.16, "max_drawdown": -0.30, "turnover": 0.50}

    decision = should_enable(bad_candidate, base_m, max_mdd_degradation=0.05)
    assert decision.enabled is False