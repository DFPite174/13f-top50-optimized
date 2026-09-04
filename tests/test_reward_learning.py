import numpy as np
import pytest
import torch

from top50_strategy.reward_learning import (
    PortfolioRewardNet,
    train_reward_model,
    constrained_policy_update,
    TrajectorySample,
)
from top50_strategy.policy import PortfolioPolicyMLP, PolicyOutput


def test_reward_model_scores_expert_above_bad_actions():
    # 2 samples: expert action [1, 0], negative action [0, 1]
    expert_states = np.array([[1.0, 0.0], [0.5, 0.5]])
    expert_actions = np.array([[1.0, 0.0], [1.0, 0.0]])
    negative_actions = np.array([[0.0, 1.0], [0.0, 1.0]])

    dataset = TrajectorySample(
        states=expert_states,
        expert_actions=expert_actions,
        negative_actions=negative_actions,
    )

    result = train_reward_model(dataset, state_dim=2, action_dim=2, epochs=40, seed=42)
    assert result.enabled is True
    assert result.validation_expert_reward > result.validation_negative_reward


def test_failed_reward_validation_disables_module():
    # Inseparable data: expert and negative have identical actions
    states = np.array([[1.0, 0.0], [0.5, 0.5]])
    actions = np.array([[0.5, 0.5], [0.5, 0.5]])

    dataset = TrajectorySample(
        states=states,
        expert_actions=actions,
        negative_actions=actions,
    )

    result = train_reward_model(dataset, state_dim=2, action_dim=2, epochs=5, seed=42, min_advantage=0.1)
    # Advantage will be ~0, so it must be disabled and trigger fallback
    assert result.enabled is False