"""Dual-expert synthesis (Fisher + Bridgewater), macro regime gates, and target actions."""

from typing import Any, Mapping, Protocol, Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from top50_strategy.types import ExpertAction


class RegimeGate(Protocol):
    def alpha(self, macro: Mapping[str, float]) -> float: ...


class HardRegimeGate:
    """Fixed-threshold macro regime gate for baseline explainability."""

    def __init__(self, bull_alpha: float = 0.70, bear_alpha: float = 0.30, neutral_alpha: float = 0.50):
        self.bull_alpha = bull_alpha
        self.bear_alpha = bear_alpha
        self.neutral_alpha = neutral_alpha

    def alpha(self, macro: Mapping[str, float]) -> float:
        gspc_mom = macro.get("gspc_mom", 0.0)
        vix_level = macro.get("vix_level", 20.0)
        vix_mom = macro.get("vix_mom", 0.0)

        is_bull = (gspc_mom > 0.02) and (vix_level < 18.0)
        if is_bull:
            return self.bull_alpha

        is_panic = (gspc_mom < -0.03) or (vix_level > 28.0) or (vix_mom > 0.20)
        if is_panic:
            return self.bear_alpha

        return self.neutral_alpha


class SmoothRegimeGate(nn.Module):
    """Small parametric gate mapping macro features to [0, 1] alpha via Sigmoid."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.sigmoid = nn.Sigmoid()
        # Initialize to output ~0.5
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))

    def alpha(self, macro: Mapping[str, float]) -> float:
        vec = torch.tensor(
            [macro.get("gspc_mom", 0.0), macro.get("vix_level", 20.0) / 20.0 - 1.0, macro.get("vix_mom", 0.0)],
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            val = float(self.forward(vec).item())
        return min(max(val, 0.0), 1.0)


class SingleExpert:
    def __init__(self, name: str, weights_map: Mapping[str, float] | None) -> None:
        self.name = name
        self.weights_map = dict(weights_map) if weights_map is not None else None

    def get_weights(self, date: pd.Timestamp, universe: Sequence[str]) -> np.ndarray | None:
        if self.weights_map is None:
            return None
        w = np.array([self.weights_map.get(sym, 0.0) for sym in universe], dtype=float)
        tot = np.sum(w)
        if tot > 1e-8:
            return w / tot
        return np.ones(len(universe)) / len(universe) if len(universe) > 0 else np.zeros(0)


class DualExpertPolicy:
    def __init__(
        self,
        fisher: SingleExpert,
        bridgewater: SingleExpert,
        gate: RegimeGate,
    ) -> None:
        self.fisher = fisher
        self.bridgewater = bridgewater
        self.gate = gate

    def act(
        self,
        date: pd.Timestamp,
        universe: Sequence[str],
        macro: Mapping[str, float],
    ) -> ExpertAction | None:
        w_f = self.fisher.get_weights(date, universe)
        w_bw = self.bridgewater.get_weights(date, universe)

        if w_f is None and w_bw is None:
            return None

        if w_f is not None and w_bw is None:
            weights = w_f
            confidence = 0.5
        elif w_f is None and w_bw is not None:
            weights = w_bw
            confidence = 0.5
        else:
            alpha = self.gate.alpha(macro)
            weights = alpha * w_f + (1.0 - alpha) * w_bw
            confidence = 1.0

        tot = float(np.sum(weights))
        if tot > 1e-8:
            weights = weights / tot
        else:
            weights = np.ones(len(universe)) / max(len(universe), 1)

        return ExpertAction(
            date=date,
            weights=weights,
            confidence=confidence,
            gross_exposure=1.0,
        )