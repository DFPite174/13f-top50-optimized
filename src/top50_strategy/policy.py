"""Dual-headed Policy Network (asset weights + gross exposure) with L1 turnover penalty."""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from top50_strategy.config import RunConfig


@dataclass(frozen=True)
class PolicyOutput:
    asset_weights: torch.Tensor  # [B, assets]
    gross_exposure: torch.Tensor  # [B, 1]


@dataclass(frozen=True)
class LossComponents:
    total_loss: torch.Tensor
    bc_loss: torch.Tensor
    turnover_loss: torch.Tensor
    entropy_loss: torch.Tensor


class PortfolioPolicyMLP(nn.Module):
    def __init__(
        self,
        market_dim: int = 103,
        assets: int = 50,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        min_exposure: float = 0.50,
        max_exposure: float = 1.25,
    ) -> None:
        super().__init__()
        self.market_dim = market_dim
        self.assets = assets
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure

        # Total state input: market_dim + assets (current holdings) + 1 (cash) + 1 (exposure)
        total_in_dim = market_dim + assets + 2
        h0 = hidden_dims[0] if len(hidden_dims) > 0 else 128
        h1 = hidden_dims[1] if len(hidden_dims) > 1 else 64

        self.trunk = nn.Sequential(
            nn.Linear(total_in_dim, h0),
            nn.LayerNorm(h0),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h0, h1),
            nn.ReLU(),
        )

        self.asset_head = nn.Sequential(
            nn.Linear(h1, assets),
            nn.Softmax(dim=-1),
        )

        self.exposure_head = nn.Sequential(
            nn.Linear(h1, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> PolicyOutput:
        h = self.trunk(state)
        w = self.asset_head(h)
        raw_exp = self.exposure_head(h)
        bounded_exp = raw_exp * (self.max_exposure - self.min_exposure) + self.min_exposure
        return PolicyOutput(asset_weights=w, gross_exposure=bounded_exp)


def behavior_cloning_loss(
    predicted: PolicyOutput,
    target_weights: torch.Tensor,
    confidence: torch.Tensor,
    current_weights: torch.Tensor,
    l1_turnover_weight: float = 0.05,
    entropy_weight: float = 0.01,
) -> LossComponents:
    # 1. Weighted MSE for behavior cloning
    sq_err = torch.mean((predicted.asset_weights - target_weights) ** 2, dim=-1)  # [B]
    conf_sum = torch.sum(confidence)
    if conf_sum > 1e-6:
        bc_loss = torch.sum(sq_err * confidence) / conf_sum
    else:
        bc_loss = torch.tensor(0.0, device=predicted.asset_weights.device)

    # 2. L1 turnover from actual current portfolio to target
    turnover_loss = torch.mean(torch.sum(torch.abs(predicted.asset_weights - current_weights), dim=-1))

    # 3. Concentration penalty (sum of squared weights)
    entropy_loss = torch.mean(torch.sum(predicted.asset_weights ** 2, dim=-1))

    total = bc_loss + l1_turnover_weight * turnover_loss + entropy_weight * entropy_loss
    return LossComponents(
        total_loss=total,
        bc_loss=bc_loss,
        turnover_loss=turnover_loss,
        entropy_loss=entropy_loss,
    )


def train_behavior_clone(
    model: PortfolioPolicyMLP,
    states: np.ndarray,
    targets: np.ndarray,
    confidences: np.ndarray,
    current_portfolios: np.ndarray,
    config: RunConfig,
) -> PortfolioPolicyMLP:
    """Train BC policy sequentially over time."""
    torch.manual_seed(config.seed)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    t_states = torch.tensor(states, dtype=torch.float32)
    t_targets = torch.tensor(targets, dtype=torch.float32)
    t_confs = torch.tensor(confidences, dtype=torch.float32)
    t_curr = torch.tensor(current_portfolios, dtype=torch.float32)

    n_samples = len(states)
    batch_size = config.batch_size

    model.train()
    for epoch in range(config.epochs):
        # Sequential temporal chunks (no random shuffle)
        for i in range(0, n_samples, batch_size):
            s_b = t_states[i : i + batch_size]
            tgt_b = t_targets[i : i + batch_size]
            c_b = t_confs[i : i + batch_size]
            curr_b = t_curr[i : i + batch_size]

            optimizer.zero_grad()
            out = model(s_b)
            loss_c = behavior_cloning_loss(
                out,
                tgt_b,
                confidence=c_b,
                current_weights=curr_b,
                l1_turnover_weight=config.l1_turnover_weight,
                entropy_weight=config.entropy_weight,
            )
            loss_c.total_loss.backward()
            optimizer.step()

    model.eval()
    return model