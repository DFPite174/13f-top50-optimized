"""Point-in-time SEC 13F filing parsing, availability filtering, and rolling universe builder."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol, Sequence
import numpy as np
import pandas as pd

from top50_strategy.config import RunConfig


@dataclass(frozen=True)
class FilingRecord:
    period_end: pd.Timestamp
    filed_at: pd.Timestamp
    available_at: pd.Timestamp
    ticker: str
    value: float
    source_id: str
    expert: str

    def __post_init__(self) -> None:
        if not isinstance(self.period_end, pd.Timestamp):
            raise TypeError("period_end must be a pd.Timestamp")
        if not isinstance(self.filed_at, pd.Timestamp):
            raise TypeError("filed_at must be a pd.Timestamp")
        if not isinstance(self.available_at, pd.Timestamp):
            raise TypeError("available_at must be a pd.Timestamp")
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("ticker must be a non-empty string")
        if self.value < 0 or np.isnan(self.value) or np.isinf(self.value):
            raise ValueError("value must be a finite non-negative number")


@dataclass(frozen=True)
class DataIssue:
    code: str
    source_id: str
    message: str


@dataclass(frozen=True)
class FilingParseResult:
    records: tuple[FilingRecord, ...]
    issues: tuple[DataIssue, ...]


class DataQualityError(ValueError):
    """Raised when data fails quality checks or universe size cannot be satisfied."""
    pass


@dataclass(frozen=True)
class PointInTimeDataset:
    market: pd.DataFrame
    expert_actions: pd.DataFrame
    universe_by_retrain: dict[pd.Timestamp, tuple[str, ...]]
    issues: tuple[DataIssue, ...]


class FilingAdapter(Protocol):
    def load(self, config: RunConfig) -> Sequence[dict[str, Any]]: ...


class MarketAdapter(Protocol):
    def load(self, config: RunConfig) -> pd.DataFrame: ...


def parse_filings(raw_filings: Sequence[dict[str, Any]]) -> FilingParseResult:
    records: list[FilingRecord] = []
    issues: list[DataIssue] = []

    for item in raw_filings:
        src_id = str(item.get("source_id", "unknown"))
        try:
            period_end = pd.Timestamp(item["period_end"])
            if period_end.tzinfo is None:
                period_end = period_end.tz_localize("UTC")

            filed_at = pd.Timestamp(item["filed_at"])
            if filed_at.tzinfo is None:
                filed_at = filed_at.tz_localize("UTC")

            avail_raw = item.get("available_at", filed_at)
            available_at = pd.Timestamp(avail_raw)
            if available_at.tzinfo is None:
                available_at = available_at.tz_localize("UTC")

            ticker = str(item["ticker"]).strip().upper()
            value = float(item["value"])
            expert = str(item.get("expert", "unknown")).lower()

            if value < 0:
                raise ValueError(f"Negative holding value: {value}")
            if not ticker:
                raise ValueError("Empty ticker")

            records.append(
                FilingRecord(
                    period_end=period_end,
                    filed_at=filed_at,
                    available_at=available_at,
                    ticker=ticker,
                    value=value,
                    source_id=src_id,
                    expert=expert,
                )
            )
        except Exception as exc:
            issues.append(
                DataIssue(
                    code="FILING_PARSE_FAILED",
                    source_id=src_id,
                    message=f"Failed to parse filing record: {exc}",
                )
            )

    return FilingParseResult(records=tuple(records), issues=tuple(issues))


def filings_available_at(records: Sequence[FilingRecord], timestamp: pd.Timestamp) -> list[FilingRecord]:
    """Returns filings strictly available at or before timestamp."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    filtered = [r for r in records if r.available_at <= ts]
    # For duplicate (expert, period_end, ticker), keep the latest available_at
    latest_map: dict[tuple[str, pd.Timestamp, str], FilingRecord] = {}
    for r in filtered:
        key = (r.expert, r.period_end, r.ticker)
        if key not in latest_map or r.available_at > latest_map[key].available_at:
            latest_map[key] = r

    return list(latest_map.values())


def build_universe(
    records: Sequence[FilingRecord],
    retrain_time: pd.Timestamp,
    quarters: int = 8,
    size: int = 50,
) -> tuple[str, ...]:
    """Build rolling universe using only filings available at retrain_time in last N quarters."""
    avail = filings_available_at(records, retrain_time)
    if not avail:
        raise DataQualityError(f"No filings available at retrain time {retrain_time}")

    # Find distinct period_ends sorted descending
    unique_quarters = sorted({r.period_end for r in avail}, reverse=True)
    selected_quarters = set(unique_quarters[:quarters])

    # Filter to selected quarters
    relevant = [r for r in avail if r.period_end in selected_quarters]

    # Aggregate appearance frequency and total value
    appearances: dict[str, set[pd.Timestamp]] = defaultdict(set)
    total_values: dict[str, float] = defaultdict(float)

    for r in relevant:
        appearances[r.ticker].add(r.period_end)
        total_values[r.ticker] += r.value

    all_tickers = sorted(appearances.keys())
    if len(all_tickers) < size:
        raise DataQualityError(
            f"Insufficient eligible assets: found {len(all_tickers)}, required {size}"
        )

    # Sort by frequency desc, total value desc, ticker asc (deterministic)
    ranked = sorted(
        all_tickers,
        key=lambda t: (-len(appearances[t]), -total_values[t], t),
    )
    return tuple(ranked[:size])


def build_point_in_time_dataset(
    config: RunConfig,
    filing_adapter: FilingAdapter,
    market_adapter: MarketAdapter,
) -> PointInTimeDataset:
    """Build complete point-in-time dataset respecting availability timestamps."""
    raw_filings = filing_adapter.load(config)
    parse_result = parse_filings(raw_filings)
    market_data = market_adapter.load(config)

    records = parse_result.records
    issues = list(parse_result.issues)

    if not records:
        return PointInTimeDataset(
            market=market_data,
            expert_actions=pd.DataFrame(),
            universe_by_retrain={},
            issues=tuple(issues),
        )

    first_avail = min(r.available_at for r in records)

    # Generate retrain dates
    market_idx = market_data.index
    if not isinstance(market_idx, pd.DatetimeIndex):
        market_idx = pd.to_datetime(market_idx)

    universe_by_retrain: dict[pd.Timestamp, tuple[str, ...]] = {}
    freq = config.retrain_frequency
    retrain_dates = pd.date_range(market_idx.min(), market_idx.max(), freq=freq, tz="UTC")

    for r_date in retrain_dates:
        if r_date >= first_avail:
            try:
                univ = build_universe(records, r_date, quarters=config.lookback_quarters, size=config.universe_size)
                universe_by_retrain[r_date] = univ
            except DataQualityError as dq_err:
                issues.append(DataIssue("UNIVERSE_BUILD_FAILED", str(r_date), str(dq_err)))

    # Expert actions prior to first_avail are strictly empty
    expert_actions = pd.DataFrame()

    return PointInTimeDataset(
        market=market_data,
        expert_actions=expert_actions,
        universe_by_retrain=universe_by_retrain,
        issues=tuple(issues),
    )