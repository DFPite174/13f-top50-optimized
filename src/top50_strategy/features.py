"""Historical feature engineering, point-in-time scaler, and execution scheduling."""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBatch:
    features: np.ndarray
    valid_mask: np.ndarray
    dates: pd.DatetimeIndex
    feature_names: tuple[str, ...]


class FeatureBuilder:
    def __init__(
        self,
        universe: Sequence[str],
        price_ratio_window: int = 20,
        momentum_window: int = 5,
    ) -> None:
        self.universe = tuple(universe)
        self.price_ratio_window = price_ratio_window
        self.momentum_window = momentum_window
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.feature_names_: tuple[str, ...] = ()

    def _compute_raw(
        self,
        close: pd.DataFrame,
        macro: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        aligned_idx = close.index
        num_rows = len(aligned_idx)
        num_assets = len(self.universe)

        feat_list: list[np.ndarray] = []
        name_list: list[str] = []

        # 1. Price ratios (close / rolling_mean)
        for sym in self.universe:
            if sym in close.columns:
                s = close[sym].astype(float)
                ma = s.rolling(self.price_ratio_window, min_periods=self.price_ratio_window).mean()
                ratio = s / (ma + 1e-8) - 1.0
                feat_list.append(ratio.values)
            else:
                feat_list.append(np.full(num_rows, np.nan))
            name_list.append(f"{sym}_pr_ratio")

        # 2. Short-term momentum (close / shift - 1.0)
        for sym in self.universe:
            if sym in close.columns:
                s = close[sym].astype(float)
                mom = s / (s.shift(self.momentum_window) + 1e-8) - 1.0
                feat_list.append(mom.values)
            else:
                feat_list.append(np.full(num_rows, np.nan))
            name_list.append(f"{sym}_mom")

        # 3. Macro variables: GSPC mom, TNX mom, VIX level & mom
        macro_cols = ["GSPC", "TNX", "VIX"]
        for m_col in macro_cols:
            if m_col in macro.columns:
                ms = macro[m_col].astype(float).reindex(aligned_idx).ffill()
                if m_col == "VIX":
                    v_mom = ms / (ms.shift(self.price_ratio_window) + 1e-8) - 1.0
                    feat_list.append(v_mom.values)
                    name_list.append("VIX_mom")
                else:
                    m_mom = ms / (ms.shift(self.price_ratio_window) + 1e-8) - 1.0
                    feat_list.append(m_mom.values)
                    name_list.append(f"{m_col}_mom")
            else:
                feat_list.append(np.zeros(num_rows))
                name_list.append(f"{m_col}_dummy")

        raw_matrix = np.column_stack(feat_list)
        valid_mask = ~np.isnan(raw_matrix)
        # Never bfill! Safe fill for neural network input
        clean_matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return clean_matrix, valid_mask, tuple(name_list)

    def fit(self, close: pd.DataFrame, macro: pd.DataFrame) -> "FeatureBuilder":
        clean_matrix, valid_mask, names = self._compute_raw(close, macro)
        self.feature_names_ = names

        # Compute mean and std only over valid elements
        means = np.zeros(clean_matrix.shape[1])
        stds = np.ones(clean_matrix.shape[1])

        for col_idx in range(clean_matrix.shape[1]):
            valid_vals = clean_matrix[valid_mask[:, col_idx], col_idx]
            if len(valid_vals) > 1:
                means[col_idx] = float(np.mean(valid_vals))
                s = float(np.std(valid_vals))
                stds[col_idx] = s if s > 1e-6 else 1.0

        self.mean_ = means
        self.scale_ = stds
        return self

    def transform(self, close: pd.DataFrame, macro: pd.DataFrame) -> FeatureBatch:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureBuilder must be fit before transform")

        clean_matrix, valid_mask, names = self._compute_raw(close, macro)
        scaled = (clean_matrix - self.mean_) / (self.scale_ + 1e-8)
        # Apply valid mask zeroing
        scaled = np.where(valid_mask, scaled, 0.0)

        return FeatureBatch(
            features=scaled,
            valid_mask=valid_mask,
            dates=close.index,
            feature_names=self.feature_names_,
        )

    def fit_transform(self, close: pd.DataFrame, macro: pd.DataFrame) -> FeatureBatch:
        return self.fit(close, macro).transform(close, macro)


def make_execution_schedule(feature_times: pd.DatetimeIndex) -> pd.Series:
    """T close signal executes at T+1 open/close session."""
    if len(feature_times) < 2:
        return pd.Series(index=feature_times, dtype="datetime64[ns]")
    exec_dates = feature_times[1:].to_list()
    # Map each T to T+1, with the last session unmapped or dropped
    mapping = {feature_times[i]: feature_times[i + 1] for i in range(len(feature_times) - 1)}
    return pd.Series(mapping)