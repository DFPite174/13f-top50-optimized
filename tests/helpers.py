"""Helper fixtures and deterministic synthetic data generators for tests."""

from pathlib import Path
import numpy as np
import pandas as pd

from top50_strategy.config import RunConfig
from top50_strategy.types import PortfolioState, ExpertAction, DatasetSplit, BacktestResult


def dates(start: str = "2020-01-02", periods: int = 5) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=periods, tz="UTC")


def sample_timestamp(date_str: str = "2020-01-02") -> pd.Timestamp:
    return pd.Timestamp(date_str, tz="UTC")


def flat_state(assets: int = 2) -> PortfolioState:
    return PortfolioState.from_values(
        date=pd.Timestamp("2020-01-02", tz="UTC"),
        long_value=100.0,
        short_value=0.0,
        cash=0.0,
        debt=0.0,
        long_weights=np.ones(assets) / assets if assets > 0 else np.zeros(0),
        short_weights=np.zeros(assets),
    )


def sample_action(assets: int = 2, confidence: float = 1.0, gross_exposure: float = 1.0) -> ExpertAction:
    weights = np.ones(assets) / assets
    return ExpertAction(
        date=pd.Timestamp("2020-01-02", tz="UTC"),
        weights=weights,
        confidence=confidence,
        gross_exposure=gross_exposure,
    )


def tiny_config(output_dir: Path | None = None) -> RunConfig:
    return RunConfig(
        train_start="2020-01-01",
        train_end="2020-06-30",
        valid_end="2020-09-30",
        test_end="2020-12-31",
        universe_size=10,
        lookback_quarters=4,
        top_k=5,
        bottom_m=2,
        seed=42,
    )
