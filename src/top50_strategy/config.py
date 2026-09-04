"""Configuration dataclasses and validation for the Top50 strategy."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import tomllib
import pandas as pd


@dataclass(frozen=True)
class RunConfig:
    train_start: str = "2018-01-01"
    train_end: str = "2021-12-31"
    valid_end: str = "2022-12-31"
    test_end: str = "2024-12-31"
    universe_size: int = 50
    lookback_quarters: int = 8
    retrain_frequency: str = "180D"
    edgar_identity: str = "Quant Researcher research@antigravity.internal"

    commission_rate: float = 0.00025
    slippage_rate: float = 0.0004
    short_borrow_rate: float = 0.02
    margin_rate: float = 0.04

    top_k: int = 15
    bottom_m: int = 8
    bull_leverage: float = 1.25
    base_leverage: float = 1.0
    panic_scale: float = 0.50
    min_weight: float = 0.01
    max_single_weight: float = 0.20

    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    learning_rate: float = 0.001
    l1_turnover_weight: float = 0.05
    entropy_weight: float = 0.01
    batch_size: int = 32
    epochs: int = 20
    seed: int = 42

    price_ratio_window: int = 20
    momentum_window: int = 5

    dagger_rounds: int = 3
    disagreement_threshold: float = 0.08
    turnover_threshold: float = 0.30

    reward_enabled: bool = False
    reward_hidden_dim: int = 64
    max_kl: float = 0.02
    max_turnover: float = 0.25
    reward_epochs: int = 15
    reward_min_advantage: float = 0.05

    def __post_init__(self) -> None:
        t_start = pd.Timestamp(self.train_start)
        t_train_end = pd.Timestamp(self.train_end)
        t_valid_end = pd.Timestamp(self.valid_end)
        t_test_end = pd.Timestamp(self.test_end)

        if not (t_start < t_train_end < t_valid_end < t_test_end):
            raise ValueError(
                f"Date boundaries must strictly satisfy train_start < train_end < valid_end < test_end: "
                f"{self.train_start} < {self.train_end} < {self.valid_end} < {self.test_end}"
            )

        if self.universe_size <= 0:
            raise ValueError(f"universe_size must be positive, got {self.universe_size}")
        if self.lookback_quarters <= 0:
            raise ValueError(f"lookback_quarters must be positive, got {self.lookback_quarters}")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if self.bottom_m < 0:
            raise ValueError(f"bottom_m must be nonnegative, got {self.bottom_m}")
        if self.top_k + self.bottom_m > self.universe_size:
            raise ValueError(
                f"top_k + bottom_m <= universe_size violated: {self.top_k} + {self.bottom_m} > {self.universe_size}"
            )

        for cost_name, cost_val in [
            ("commission_rate", self.commission_rate),
            ("slippage_rate", self.slippage_rate),
            ("short_borrow_rate", self.short_borrow_rate),
            ("margin_rate", self.margin_rate),
        ]:
            if cost_val < 0:
                raise ValueError(f"costs must be nonnegative, but {cost_name} = {cost_val}")

        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError(f"hidden_dims must contain positive values, got {self.hidden_dims}")

        if self.base_leverage <= 0 or self.bull_leverage < self.base_leverage:
            raise ValueError(
                f"Leverage invalid: bull_leverage ({self.bull_leverage}) must be >= base_leverage ({self.base_leverage}) > 0"
            )
        if not (0 < self.panic_scale <= 1.0):
            raise ValueError(f"panic_scale must be in (0, 1], got {self.panic_scale}")

    @classmethod
    def from_toml(cls, path: str | Path) -> "RunConfig":
        p = Path(path)
        with open(p, "rb") as f:
            data = tomllib.load(f)

        data_sec = data.get("data", {})
        costs_sec = data.get("costs", {})
        port_sec = data.get("portfolio", {})
        model_sec = data.get("model", {})
        feat_sec = data.get("features", {})
        dag_sec = data.get("dagger", {})
        rew_sec = data.get("reward", {})

        raw_hidden = model_sec.get("hidden_dims", (128, 64))
        hidden_dims = tuple(raw_hidden) if isinstance(raw_hidden, (list, tuple)) else (128, 64)

        return cls(
            train_start=data_sec.get("train_start", "2018-01-01"),
            train_end=data_sec.get("train_end", "2021-12-31"),
            valid_end=data_sec.get("valid_end", "2022-12-31"),
            test_end=data_sec.get("test_end", "2024-12-31"),
            universe_size=data_sec.get("universe_size", 50),
            lookback_quarters=data_sec.get("lookback_quarters", 8),
            retrain_frequency=data_sec.get("retrain_frequency", "180D"),
            edgar_identity=data_sec.get("edgar_identity", "Quant Researcher research@antigravity.internal"),

            commission_rate=costs_sec.get("commission_rate", 0.00025),
            slippage_rate=costs_sec.get("slippage_rate", 0.0004),
            short_borrow_rate=costs_sec.get("short_borrow_rate", 0.02),
            margin_rate=costs_sec.get("margin_rate", 0.04),

            top_k=port_sec.get("top_k", 15),
            bottom_m=port_sec.get("bottom_m", 8),
            bull_leverage=port_sec.get("bull_leverage", 1.25),
            base_leverage=port_sec.get("base_leverage", 1.0),
            panic_scale=port_sec.get("panic_scale", 0.50),
            min_weight=port_sec.get("min_weight", 0.01),
            max_single_weight=port_sec.get("max_single_weight", 0.20),

            hidden_dims=hidden_dims,
            dropout=model_sec.get("dropout", 0.1),
            learning_rate=model_sec.get("learning_rate", 0.001),
            l1_turnover_weight=model_sec.get("l1_turnover_weight", 0.05),
            entropy_weight=model_sec.get("entropy_weight", 0.01),
            batch_size=model_sec.get("batch_size", 32),
            epochs=model_sec.get("epochs", 20),
            seed=model_sec.get("seed", 42),

            price_ratio_window=feat_sec.get("price_ratio_window", 20),
            momentum_window=feat_sec.get("momentum_window", 5),

            dagger_rounds=dag_sec.get("rounds", 3),
            disagreement_threshold=dag_sec.get("disagreement_threshold", 0.08),
            turnover_threshold=dag_sec.get("turnover_threshold", 0.30),

            reward_enabled=rew_sec.get("enabled", False),
            reward_hidden_dim=rew_sec.get("reward_hidden_dim", 64),
            max_kl=rew_sec.get("max_kl", 0.02),
            max_turnover=rew_sec.get("max_turnover", 0.25),
            reward_epochs=rew_sec.get("epochs", 15),
            reward_min_advantage=rew_sec.get("min_advantage", 0.05),
        )
