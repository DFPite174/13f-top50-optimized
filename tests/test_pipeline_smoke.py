from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from top50_strategy.config import RunConfig
from top50_strategy.pipeline import run_research, SyntheticAdapters


def test_pipeline_runs_on_injected_synthetic_adapters(tmp_path):
    dates = pd.bdate_range("2020-01-01", "2021-12-31", tz="UTC")
    tickers = ("AAPL", "MSFT", "GOOG", "AMZN", "NVDA")

    adapters = SyntheticAdapters(dates, tickers)
    cfg = RunConfig(
        train_start="2020-01-01",
        train_end="2020-12-31",
        valid_end="2021-06-30",
        test_end="2021-12-31",
        universe_size=5,
        lookback_quarters=4,
        top_k=3,
        bottom_m=1,
        epochs=3,
        seed=42,
    )

    report = run_research(cfg, adapters.filing_adapter, adapters.market_adapter, output_dir=tmp_path)
    assert report.data_audit.future_access_count == 0
    assert set(report.ablation_table["Model"].values) == {"M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"}
    assert len(report.latest_weights) > 0
    # Artifacts saved
    assert (tmp_path / "ablation_metrics.csv").exists()
    assert (tmp_path / "data_quality.json").exists()