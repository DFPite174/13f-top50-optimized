import numpy as np
import pandas as pd
import pytest
import torch

from top50_strategy.dagger import (
    collect_dagger_rollout,
    aggregate_pairs,
    StateActionPair,
)
from top50_strategy.overlay import ActionProjector
from top50_strategy.policy import PortfolioPolicyMLP
from top50_strategy.simulator import MarketSimulator, CostModel
from top50_strategy.experts import DualExpertPolicy, SingleExpert, HardRegimeGate


class MockExpert:
    def __init__(self, tickers):
        self.tickers = tickers
        self.received_states = []

    def act(self, date, universe, macro):
        w = np.ones(len(universe)) / len(universe)
        return type("Action", (), {"weights": w, "confidence": 1.0, "gross_exposure": 1.0})()


def test_dagger_queries_expert_on_policy_visited_portfolio():
    dates = pd.bdate_range("2020-01-02", periods=3, tz="UTC")
    df_open = pd.DataFrame({"AAPL": [100.0, 102.0, 101.0], "MSFT": [50.0, 51.0, 50.5]}, index=dates)
    df_close = pd.DataFrame({"AAPL": [101.0, 101.5, 102.0], "MSFT": [50.5, 50.8, 51.0]}, index=dates)

    sim = MarketSimulator(df_open, df_close, CostModel(0, 0, 0, 0))
    tickers = ("AAPL", "MSFT")
    expert = MockExpert(tickers)
    projector = ActionProjector(tickers=tickers, top_k=2, bottom_m=0, max_single_weight=1.0)

    class BiasedPolicy:
        def __call__(self, state):
            w = torch.tensor([[1.0, 0.0]])
            exp = torch.tensor([[1.0]])
            return type("Out", (), {"asset_weights": w, "gross_exposure": exp})()

    pairs = collect_dagger_rollout(sim, BiasedPolicy(), expert, projector, tickers)
    assert len(pairs) > 0
    assert np.any(pairs[0].policy_weights > 0.5)


def test_dagger_does_not_duplicate_identical_state_label_pairs():
    s = np.array([1.0, 2.0])
    a = np.array([0.5, 0.5])
    pair1 = StateActionPair(s, a, a, 1.0)
    pair2 = StateActionPair(s, a, a, 1.0)
    pair3 = StateActionPair(np.array([2.0, 3.0]), a, a, 1.0)

    aggregated = aggregate_pairs([pair1, pair2, pair3], distance_threshold=1e-4)
    assert len(aggregated) == 2