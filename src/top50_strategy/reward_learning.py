"""Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) and constrained policy optimization."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from top50_strategy.policy import PortfolioPolicyMLP


@dataclass(frozen=True)
class TrajectorySample:
    states: np.ndarray
    expert_actions: np.ndarray
    negative_actions: np.ndarray


@dataclass(frozen=True)
class RewardModelResult:
    reward_net: nn.Module | None
    enabled: bool
    validation_expert_reward: float
    validation_negative_reward: float
    advantage: float
    message: str


class PortfolioRewardNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


def train_reward_model(
    dataset: TrajectorySample,
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 64,
    epochs: int = 25,
    lr: float = 0.005,
    min_advantage: float = 0.05,
    seed: int = 42,
) -> RewardModelResult:
    """Train reward network to score expert actions higher than divergent actions."""
    torch.manual_seed(seed)
    reward_net = PortfolioRewardNet(state_dim, action_dim, hidden_dim)
    optimizer = optim.Adam(reward_net.parameters(), lr=lr)

    s = torch.tensor(dataset.states, dtype=torch.float32)
    a_pos = torch.tensor(dataset.expert_actions, dtype=torch.float32)
    a_neg = torch.tensor(dataset.negative_actions, dtype=torch.float32)

    reward_net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        r_pos = reward_net(s, a_pos)
        r_neg = reward_net(s, a_neg)
        # Margin ranking loss: softplus(r_neg - r_pos) = log(1 + exp(r_neg - r_pos))
        loss = torch.mean(torch.log1p(torch.exp(r_neg - r_pos)))
        loss.backward()
        optimizer.step()

    reward_net.eval()
    with torch.no_grad():
        r_pos_val = float(torch.mean(reward_net(s, a_pos)).item())
        r_neg_val = float(torch.mean(reward_net(s, a_neg)).item())
        advantage = r_pos_val - r_neg_val

    if advantage >= min_advantage:
        return RewardModelResult(
            reward_net=reward_net,
            enabled=True,
            validation_expert_reward=r_pos_val,
            validation_negative_reward=r_neg_val,
            advantage=advantage,
            message="Reward model passed validation; enabled for constrained fine-tuning.",
        )
    else:
        return RewardModelResult(
            reward_net=None,
            enabled=False,
            validation_expert_reward=r_pos_val,
            validation_negative_reward=r_neg_val,
            advantage=advantage,
            message="Reward model advantage below threshold; safely falling back to DAgger policy.",
        )


def constrained_policy_update(
    policy: PortfolioPolicyMLP,
    reward_net: PortfolioRewardNet,
    states: torch.Tensor,
    max_kl: float = 0.02,
    max_turnover: float = 0.25,
    lr: float = 0.0005,
    steps: int = 5,
) -> PortfolioPolicyMLP:
    """Fine-tune policy with small steps subject to KL divergence and turnover bounds."""
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    with torch.no_grad():
        base_out = policy(states)
        base_w = base_out.asset_weights.detach()

    for _ in range(steps):
        optimizer.zero_grad()
        out = policy(states)
        pred_w = out.asset_weights

        # Reward maximization
        r = reward_net(states, pred_w)
        obj = -torch.mean(r)

        # KL constraint penalty
        kl = torch.mean(torch.sum(pred_w * torch.log((pred_w + 1e-8) / (base_w + 1e-8)), dim=-1))
        # Turnover penalty
        turnover = torch.mean(torch.sum(torch.abs(pred_w - base_w), dim=-1))

        loss = obj + 10.0 * torch.relu(kl - max_kl) + 5.0 * torch.relu(turnover - max_turnover)
        loss.backward()
        optimizer.step()

    return policy