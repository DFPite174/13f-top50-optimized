import numpy as np
import pytest
import torch

from top50_strategy.policy import (
    PortfolioPolicyMLP,
    PolicyOutput,
    behavior_cloning_loss,
)


def test_policy_outputs_distribution_and_bounded_exposure():
    model = PortfolioPolicyMLP(market_dim=103, assets=50, min_exposure=0.5, max_exposure=1.25)
    dummy_input = torch.zeros(4, 155)  # 103 market + 50 hold + 1 cash + 1 exp
    out = model(dummy_input)

    assert isinstance(out, PolicyOutput)
    # asset weights must sum to 1 across assets
    torch.testing.assert_close(out.asset_weights.sum(dim=-1), torch.ones(4))
    assert torch.all((out.gross_exposure >= 0.5 - 1e-5) & (out.gross_exposure <= 1.25 + 1e-5))


def test_zero_confidence_label_contributes_no_bc_loss():
    pred_weights = torch.tensor([[0.7, 0.3], [0.8, 0.2]], requires_grad=True)
    pred_exp = torch.tensor([[1.0], [1.0]], requires_grad=True)
    pred = PolicyOutput(pred_weights, pred_exp)

    target_weights = torch.tensor([[0.5, 0.5], [0.1, 0.9]])
    current_weights = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    confidence = torch.tensor([1.0, 0.0])

    loss_obj = behavior_cloning_loss(
        pred,
        target_weights,
        confidence=confidence,
        current_weights=current_weights,
        l1_turnover_weight=0.0,
        entropy_weight=0.0,
    )

    # Only sample 0 should contribute: ((0.7-0.5)^2 + (0.3-0.5)^2) / 2 = (0.04 + 0.04) / 2 = 0.04
    expected_bc = 0.04
    assert loss_obj.bc_loss.item() == pytest.approx(expected_bc, abs=1e-4)