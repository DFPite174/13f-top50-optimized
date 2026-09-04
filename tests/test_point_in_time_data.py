from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from top50_strategy.config import RunConfig
from top50_strategy.data import (
    FilingRecord,
    DataIssue,
    DataQualityError,
    filings_available_at,
    build_universe,
    parse_filings,
    build_point_in_time_dataset,
)


def make_filing(ticker, released, period_end="2019-12-31", value=100.0, expert="fisher", source_id="acc-1"):
    t_rel = pd.Timestamp(released, tz="UTC")
    t_end = pd.Timestamp(period_end, tz="UTC")
    return FilingRecord(
        period_end=t_end,
        filed_at=t_rel,
        available_at=t_rel,
        ticker=ticker,
        value=value,
        source_id=source_id,
        expert=expert,
    )


def test_filings_available_at_excludes_future_release():
    records = [
        make_filing("AAPL", released="2020-02-14 16:30:00"),
        make_filing("MSFT", released="2020-05-15 16:30:00"),
    ]
    available = filings_available_at(records, pd.Timestamp("2020-03-01", tz="UTC"))
    assert {r.ticker for r in available} == {"AAPL"}


def test_universe_uses_only_last_eight_available_quarters():
    records = []
    for i in range(1, 9):
        records.append(make_filing("AAPL", released=f"2020-0{i}-15" if i < 10 else f"2020-{i}-15", period_end=f"2020-0{i}-01" if i < 10 else f"2020-{i}-01", value=100.0))
        if i >= 3:
            records.append(make_filing("MSFT", released=f"2020-0{i}-15" if i < 10 else f"2020-{i}-15", period_end=f"2020-0{i}-01" if i < 10 else f"2020-{i}-01", value=80.0))
    # GOOG released in 2023
    records.append(make_filing("GOOG", released="2023-01-15", period_end="2022-12-31", value=500.0))

    universe = build_universe(records, pd.Timestamp("2021-01-01", tz="UTC"), quarters=8, size=2)
    assert universe == ("AAPL", "MSFT")
    assert "GOOG" not in universe


def test_universe_raises_data_quality_error_when_insufficient():
    records = [make_filing("AAPL", released="2020-02-14")]
    with pytest.raises(DataQualityError, match="Insufficient eligible assets"):
        build_universe(records, pd.Timestamp("2020-03-01", tz="UTC"), quarters=8, size=5)


def test_bad_filing_is_reported_not_silently_dropped():
    raw_filings = [
        {"period_end": "2020-12-31", "filed_at": "2021-02-14", "ticker": "AAPL", "value": 100.0, "expert": "fisher", "source_id": "good-1"},
        {"period_end": "bad-date", "filed_at": "2021-02-14", "ticker": "MSFT", "value": -10.0, "expert": "fisher", "source_id": "bad-1"},
    ]
    res = parse_filings(raw_filings)
    assert len(res.records) == 1
    assert res.records[0].ticker == "AAPL"
    assert len(res.issues) == 1
    assert res.issues[0].code == "FILING_PARSE_FAILED"
    assert res.issues[0].source_id == "bad-1"


def test_no_label_before_first_available_filing():
    class DummyFilingAdapter:
        def load(self, config):
            return [
                {"period_end": "2021-12-31", "filed_at": "2022-02-14", "ticker": "AAPL", "value": 100.0, "expert": "fisher", "source_id": "f1"}
            ]

    class DummyMarketAdapter:
        def load(self, config):
            dates = pd.bdate_range("2020-01-01", "2020-06-30", tz="UTC")
            return pd.DataFrame({"AAPL": 100.0}, index=dates)

    cfg = RunConfig(
        train_start="2020-01-01",
        train_end="2020-06-30",
        valid_end="2020-09-30",
        test_end="2020-12-31",
        universe_size=1,
        top_k=1,
        bottom_m=0,
    )
    dataset = build_point_in_time_dataset(cfg, DummyFilingAdapter(), DummyMarketAdapter())
    assert dataset.expert_actions.empty