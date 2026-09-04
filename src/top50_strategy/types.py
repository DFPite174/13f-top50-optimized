"""Cross-module data structures, contracts, and invariant checks."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpertAction:
    date: pd.Timestamp
    weights: np.ndarray
    confidence: float
    gross_exposure: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.date, pd.Timestamp):
            raise TypeError(f"date must be a timestamp, got {type(self.date)}")

        arr = np.asarray(self.weights, dtype=float)
        if arr.ndim != 1 or len(arr) == 0:
            raise ValueError("weights must be a 1D non-empty array")
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("weights must be finite numbers")
        if np.any(arr < -1e-8):
            raise ValueError("weights must be nonnegative and sum to one")
        total_w = float(np.sum(arr))
        if abs(total_w - 1.0) > 1e-4 and total_w > 0:
            raise ValueError(f"weights must be nonnegative and sum to one, got sum={total_w}")

        if not (0.0 <= self.confidence <= 1.0) or np.isnan(self.confidence):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

        if self.gross_exposure < 0.0 or np.isinf(self.gross_exposure) or np.isnan(self.gross_exposure):
            raise ValueError(f"gross exposure must be nonnegative, got {self.gross_exposure}")


@dataclass(frozen=True)
class PortfolioState:
    date: pd.Timestamp
    long_value: float
    short_value: float
    cash: float
    debt: float
    equity: float
    net_exposure: float
    gross_exposure: float
    long_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    short_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        if self.short_value > 1e-7:
            raise ValueError(f"short_value must be nonpositive (<= 0), got {self.short_value}")
        if self.long_value < -1e-7:
            raise ValueError(f"long_value must be nonnegative (>= 0), got {self.long_value}")
        if self.debt < -1e-7:
            raise ValueError(f"debt must be nonnegative (>= 0), got {self.debt}")

        expected_equity = self.long_value + self.short_value + self.cash - self.debt
        if abs(self.equity - expected_equity) > 1e-4:
            raise ValueError(
                f"Accounting identity violated: equity={self.equity} != "
                f"long({self.long_value}) + short({self.short_value}) + cash({self.cash}) - debt({self.debt}) = {expected_equity}"
            )

    @classmethod
    def from_values(
        cls,
        date: pd.Timestamp,
        long_value: float,
        short_value: float,
        cash: float,
        debt: float,
        long_weights: np.ndarray | None = None,
        short_weights: np.ndarray | None = None,
    ) -> "PortfolioState":
        equity = long_value + short_value + cash - debt
        eq_denom = equity if abs(equity) > 1e-8 else 1.0
        net_exposure = (long_value + short_value) / eq_denom
        gross_exposure = (abs(long_value) + abs(short_value)) / eq_denom

        return cls(
            date=pd.Timestamp(date),
            long_value=float(long_value),
            short_value=float(short_value),
            cash=float(cash),
            debt=float(debt),
            equity=float(equity),
            net_exposure=float(net_exposure),
            gross_exposure=float(gross_exposure),
            long_weights=np.zeros(0) if long_weights is None else np.asarray(long_weights, dtype=float),
            short_weights=np.zeros(0) if short_weights is None else np.asarray(short_weights, dtype=float),
        )


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DatetimeIndex
    valid: pd.DatetimeIndex
    test: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if len(self.train) > 0 and len(self.valid) > 0:
            if self.train.max() >= self.valid.min():
                raise ValueError("train index must strictly precede valid index")
        if len(self.valid) > 0 and len(self.test) > 0:
            if self.valid.max() >= self.test.min():
                raise ValueError("valid index must strictly precede test index")


@dataclass(frozen=True)
class CostBreakdown:
    commission: float = 0.0
    slippage: float = 0.0
    borrow_cost: float = 0.0
    margin_interest: float = 0.0

    @property
    def total(self) -> float:
        return self.commission + self.slippage + self.borrow_cost + self.margin_interest


@dataclass(frozen=True)
class BacktestResult:
    returns: pd.Series
    cum_returns: pd.Series
    turnover: pd.Series
    drawdown: pd.Series
    equity_curve: pd.Series
    metrics: dict[str, float]
    cost_history: pd.DataFrame | None = None
    weights_history: pd.DataFrame | None = None
