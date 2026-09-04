import numpy as np
import pandas as pd
import pytest
import torch

from top50_strategy.features import FeatureBuilder, make_execution_schedule
from top50_strategy.experts import (
    DualExpertPolicy,
    HardRegimeGate,
    SmoothRegimeGate,
    SingleExpert,
)
from top50_strategy.types import ExpertAction, PortfolioState


def dates(start="2020-01-02", periods=5):
    return pd.bdate_range(start=start, periods=periods, tz="UTC")


def test_future_price_never_fills_past_feature():
    idx = dates("2020-01-02", 3)
    # Day 0 is NaN, Day 1 is 100.0, Day 2 is 105.0
    close = pd.DataFrame({"AAPL": [np.nan, 100.0, 105.0]}, index=idx)
    macro = pd.DataFrame({"GSPC": [3000.0, 3010.0, 3020.0], "TNX": [2.0, 2.05, 2.1], "VIX": [15.0, 14.8, 14.5]}, index=idx)

    builder = FeatureBuilder(universe=("AAPL",), price_ratio_window=2, momentum_window=1)
    batch = builder.fit_transform(close, macro)

    # AAPL on day 0 must be invalid / NaN and masked out
    assert not batch.valid_mask[0, 0]


def test_close_feature_executes_next_session():
    idx = dates("2020-01-02", 3)
    sched = make_execution_schedule(idx)
    assert sched.loc[idx[0]] == idx[1]
    assert sched.loc[idx[1]] == idx[2]


def test_scaler_ignores_validation_values():
    train_idx = dates("2020-01-02", 5)
    valid_idx = dates("2020-01-09", 5)
    train_close = pd.DataFrame({"AAPL": [10.0, 10.0, 10.0, 10.0, 10.0]}, index=train_idx)
    valid_close = pd.DataFrame({"AAPL": [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]}, index=valid_idx)
    macro_t = pd.DataFrame({"GSPC": [3000.0]*5, "TNX": [2.0]*5, "VIX": [15.0]*5}, index=train_idx)
    macro_v = pd.DataFrame({"GSPC": [4000.0]*5, "TNX": [3.0]*5, "VIX": [35.0]*5}, index=valid_idx)

    builder = FeatureBuilder(universe=("AAPL",), price_ratio_window=2, momentum_window=1)
    builder.fit(train_close, macro_t)
    mean_before = builder.mean_.copy()

    builder.transform(valid_close, macro_v)
    np.testing.assert_array_equal(builder.mean_, mean_before)


def test_dual_expert_blend_is_normalized():
    univ = ("AAPL", "MSFT")
    e_fisher = SingleExpert("fisher", {"AAPL": 0.6, "MSFT": 0.4})
    e_bw = SingleExpert("bridgewater", {"AAPL": 0.2, "MSFT": 0.8})

    gate = HardRegimeGate()
    policy = DualExpertPolicy(e_fisher, e_bw, gate)
    macro = {"gspc_mom": 0.05, "vix_level": 14.0, "vix_mom": -0.05}

    action = policy.act(pd.Timestamp("2020-01-02", tz="UTC"), univ, macro)
    assert action is not None
    assert action.weights.sum() == pytest.approx(1.0)
    assert np.all(action.weights >= 0)
    assert action.confidence == pytest.approx(1.0)


def test_single_available_expert_has_half_confidence():
    univ = ("AAPL", "MSFT")
    e_fisher = SingleExpert("fisher", {"AAPL": 0.6, "MSFT": 0.4})
    e_missing = SingleExpert("bridgewater", None)  # Not available

    policy = DualExpertPolicy(e_fisher, e_missing, HardRegimeGate())
    macro = {"gspc_mom": 0.0, "vix_level": 20.0, "vix_mom": 0.0}

    action = policy.act(pd.Timestamp("2020-01-02", tz="UTC"), univ, macro)
    assert action is not None
    assert action.confidence == pytest.approx(0.5)
    assert action.weights.sum() == pytest.approx(1.0)


def test_no_available_expert_returns_no_label():
    univ = ("AAPL", "MSFT")
    policy = DualExpertPolicy(SingleExpert("fisher", None), SingleExpert("bw", None), HardRegimeGate())
    action = policy.act(pd.Timestamp("2020-01-02", tz="UTC"), univ, {})
    assert action is None