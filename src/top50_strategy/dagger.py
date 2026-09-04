"""Interactive DAgger (Dataset Aggregation) with closed-loop simulation and query-on-visited states."""

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import torch

from top50_strategy.config import RunConfig
from top50_strategy.overlay import ActionProjector
from top50_strategy.policy import PortfolioPolicyMLP, train_behavior_clone
from top50_strategy.simulator import MarketSimulator


@dataclass(frozen=True)
class StateActionPair:
    state: np.ndarray
    policy_weights: np.ndarray
    expert_weights: np.ndarray
    confidence: float


@dataclass(frozen=True)
class DaggerRoundDiagnostic:
    round_index: int
    new_pairs: int
    total_pairs: int
    mean_disagreement: float


@dataclass(frozen=True)
class DaggerResult:
    policy: PortfolioPolicyMLP
    aggregated_pairs: list[StateActionPair]
    diagnostics: list[DaggerRoundDiagnostic]


def aggregate_pairs(
    pairs: Sequence[StateActionPair],
    distance_threshold: float = 1e-4,
) -> list[StateActionPair]:
    """Deduplicate pairs using Euclidean distance on states."""
    aggregated: list[StateActionPair] = []
    for candidate in pairs:
        is_dup = False
        for existing in aggregated:
            s_dist = float(np.linalg.norm(candidate.state - existing.state))
            a_dist = float(np.linalg.norm(candidate.expert_weights - existing.expert_weights))
            if s_dist < distance_threshold and a_dist < distance_threshold:
                is_dup = True
                break
        if not is_dup:
            aggregated.append(candidate)
    return aggregated


def collect_dagger_rollout(
    simulator: MarketSimulator,
    policy: Any,
    expert: Any,
    projector: ActionProjector,
    tickers: Sequence[str],
    disagreement_threshold: float = 0.05,
) -> list[StateActionPair]:
    """Roll out policy in simulator, query expert on visited states, and collect divergent pairs."""
    simulator.reset()
    collected: list[StateActionPair] = []
    num_assets = len(tickers)

    while simulator.current_step < len(simulator.dates):
        current_date = simulator.dates[simulator.current_step]
        curr_state = simulator.current_state

        # State: portfolio weights (num_assets) + cash (1) + exposure (1)
        holdings = curr_state.long_weights if len(curr_state.long_weights) == num_assets else np.zeros(num_assets)
        cash_ratio = curr_state.cash / (curr_state.equity + 1e-8)
        exp_ratio = curr_state.gross_exposure
        base_state = np.concatenate([holdings, [cash_ratio, exp_ratio]])

        # 1. Policy generates action on currently visited state (pad with zeros if policy expects market features)
        if hasattr(policy, "trunk") and len(policy.trunk) > 0 and hasattr(policy.trunk[0], "in_features"):
            in_dim = policy.trunk[0].in_features
            if len(base_state) < in_dim:
                pad = np.zeros(in_dim - len(base_state))
                state_vec = np.concatenate([pad, base_state])
            else:
                state_vec = base_state
        else:
            state_vec = base_state

        t_state = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p_out = policy(t_state)
            p_weights = p_out.asset_weights[0].cpu().numpy()
            p_exp = float(p_out.gross_exposure[0].item())

        # 2. Query expert on this same state
        macro = {"gspc_mom": 0.0, "vix_level": 20.0, "vix_mom": 0.0}
        exp_action = expert.act(current_date, tickers, macro)

        if exp_action is not None:
            e_weights = exp_action.weights
            disagreement = float(np.linalg.norm(p_weights - e_weights))
            if disagreement >= disagreement_threshold or len(collected) == 0:
                collected.append(
                    StateActionPair(
                        state=state_vec,
                        policy_weights=p_weights,
                        expert_weights=e_weights,
                        confidence=exp_action.confidence,
                    )
                )

        # 3. Policy action executed in simulator to alter next state
        target_action = projector.project(
            policy_weights=p_weights,
            exposure=p_exp,
            macro=macro,
            stock_moms=np.zeros(num_assets),
        )
        simulator.step(target_action.long_weights, target_action.short_weights)

    return collected