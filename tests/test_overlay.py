import numpy as np
import pytest

from top50_strategy.overlay import ActionProjector, TargetAction
from top50_strategy.policy import PolicyOutput


def test_top_k_keeps_only_highest_weights_and_preserves_exposure():
    # 4 assets, weights = [0.4, 0.3, 0.2, 0.1]
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    tickers = ("A", "B", "C", "D")
    projector = ActionProjector(tickers=tickers, top_k=2, bottom_m=0, max_single_weight=1.0)

    # Bull regime -> long scale = 1.0 (base)
    macro = {"gspc_mom": 0.0, "vix_level": 19.0, "vix_mom": 0.0}
    action = projector.project(
        policy_weights=weights,
        exposure=0.8,
        macro=macro,
        stock_moms=np.array([0.01, 0.02, 0.03, 0.04]),
    )

    # Top 2 are A (0.4) and B (0.3). Normalized: 4/7 and 3/7, scaled by 0.8
    assert action.long_weights[2] == 0.0
    assert action.long_weights[3] == 0.0
    assert action.long_weights[0] == pytest.approx(0.8 * 4 / 7)
    assert action.long_weights[1] == pytest.approx(0.8 * 3 / 7)
    assert np.all(action.short_weights == 0.0)


def test_low_softmax_probability_alone_does_not_create_short():
    weights = np.array([0.9, 0.1])
    tickers = ("A", "B")
    projector = ActionProjector(tickers=tickers, top_k=1, bottom_m=1)

    macro = {"gspc_mom": 0.05, "vix_level": 15.0, "vix_mom": -0.05}
    action = projector.project(
        policy_weights=weights,
        exposure=1.0,
        macro=macro,
        stock_moms=np.array([0.05, -0.01]),
    )
    assert np.all(action.short_weights == 0.0)


def test_short_requires_all_eligibility_conditions():
    weights = np.array([0.7, 0.3])
    tickers = ("A", "B")
    projector = ActionProjector(tickers=tickers, top_k=1, bottom_m=1)

    macro = {"gspc_mom": -0.05, "vix_level": 30.0, "vix_mom": 0.25}
    stock_moms = np.array([0.02, -0.08])

    action = projector.project(
        policy_weights=weights,
        exposure=0.5,
        macro=macro,
        stock_moms=stock_moms,
    )
    assert action.short_weights[1] < 0.0
    assert action.short_weights[0] == 0.0