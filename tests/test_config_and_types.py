from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from top50_strategy.config import RunConfig
from top50_strategy.types import (
    BacktestResult,
    CostBreakdown,
    DatasetSplit,
    ExpertAction,
    PortfolioState,
)


def test_run_config_rejects_invalid_split_order():
    with pytest.raises(ValueError, match="Date boundaries must strictly satisfy"):
        RunConfig(train_start="2020-01-01", train_end="2021-01-01", valid_end="2020-06-01", test_end="2022-01-01")


@pytest.mark.parametrize(
    "override,match",
    [
        ({"universe_size": 0}, "universe_size must be positive"),
        ({"lookback_quarters": 0}, "lookback_quarters must be positive"),
        ({"top_k": 0}, "top_k must be positive"),
        ({"bottom_m": -1}, "bottom_m must be nonnegative"),
        ({"commission_rate": -0.001}, "costs must be nonnegative"),
        ({"slippage_rate": -0.001}, "costs must be nonnegative"),
        ({"hidden_dims": ()}, "hidden_dims must contain positive values"),
        ({"hidden_dims": (128, -64)}, "hidden_dims must contain positive values"),
        ({"top_k": 40, "bottom_m": 20, "universe_size": 50}, "top_k \\+ bottom_m <= universe_size"),
    ],
)
def test_run_config_rejects_invalid_constraints(override, match):
    with pytest.raises(ValueError, match=match):
        RunConfig(**override)


def test_run_config_loads_baseline_toml():
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.toml"
    cfg = RunConfig.from_toml(cfg_path)
    assert cfg.universe_size == 50
    assert cfg.top_k == 15
    assert cfg.bottom_m == 8
    assert cfg.commission_rate == 0.00025


@pytest.mark.parametrize(
    "weights",
    [
        np.array([0.8, -0.1]),
        np.array([0.5, 0.4]),
        np.array([]),
    ],
)
def test_expert_action_requires_normalized_nonnegative_weights(weights):
    with pytest.raises(ValueError):
        ExpertAction(
            date=pd.Timestamp("2020-01-02", tz="UTC"),
            weights=weights,
            confidence=1.0,
            gross_exposure=1.0,
        )


@pytest.mark.parametrize(
    "date,confidence,gross_exposure,err_type,match",
    [
        ("2020-01-02", 0.5, 1.0, TypeError, "date must be a timestamp"),
        (pd.Timestamp("2020-01-02"), -0.1, 1.0, ValueError, "confidence must be in \\[0, 1\\]"),
        (pd.Timestamp("2020-01-02"), 1.1, 1.0, ValueError, "confidence must be in \\[0, 1\\]"),
        (pd.Timestamp("2020-01-02"), np.nan, 1.0, ValueError, "confidence must be in \\[0, 1\\]"),
        (pd.Timestamp("2020-01-02"), 0.5, -0.1, ValueError, "gross exposure must be nonnegative"),
        (pd.Timestamp("2020-01-02"), 0.5, np.inf, ValueError, "gross exposure must be nonnegative"),
    ],
)
def test_expert_action_validates_metadata(date, confidence, gross_exposure, err_type, match):
    with pytest.raises(err_type, match=match):
        ExpertAction(
            date=date,
            weights=np.array([1.0]),
            confidence=confidence,
            gross_exposure=gross_exposure,
        )


def test_portfolio_state_preserves_accounting_identity():
    state = PortfolioState.from_values(
        date=pd.Timestamp("2020-01-02", tz="UTC"),
        long_value=120.0,
        short_value=-20.0,
        cash=10.0,
        debt=10.0,
    )
    assert state.equity == pytest.approx(100.0)
    assert state.gross_exposure == pytest.approx(1.4)
    assert state.net_exposure == pytest.approx(1.0)


def test_portfolio_state_rejects_positive_short_value():
    with pytest.raises(ValueError, match="short_value must be nonpositive"):
        PortfolioState(
            date=pd.Timestamp("2020-01-02", tz="UTC"),
            long_value=100.0,
            short_value=10.0,
            cash=0.0,
            debt=0.0,
            equity=110.0,
            net_exposure=1.0,
            gross_exposure=1.0,
        )


def test_dataset_split_is_immutable():
    d1 = pd.bdate_range("2020-01-01", "2020-01-10", tz="UTC")
    d2 = pd.bdate_range("2020-01-11", "2020-01-20", tz="UTC")
    d3 = pd.bdate_range("2020-01-21", "2020-01-31", tz="UTC")
    split = DatasetSplit(train=d1, valid=d2, test=d3)
    with pytest.raises(AttributeError):
        split.train = d2


def test_backtest_result_is_immutable_and_carries_required_series():
    idx = pd.bdate_range("2020-01-01", periods=3, tz="UTC")
    s = pd.Series([0.01, 0.02, -0.01], index=idx)
    res = BacktestResult(
        returns=s,
        cum_returns=(1 + s).cumprod(),
        turnover=s.abs(),
        drawdown=s,
        equity_curve=(1 + s).cumprod(),
        metrics={"sharpe": 1.5, "cagr": 0.2},
    )
    assert "sharpe" in res.metrics
    with pytest.raises(AttributeError):
        res.metrics = {}
