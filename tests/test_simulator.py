import numpy as np
import pandas as pd
import pytest

from top50_strategy.simulator import MarketSimulator, CostModel
from top50_strategy.types import PortfolioState


def test_t_close_signal_cannot_earn_t_intraday_return():
    dates = pd.bdate_range("2020-01-02", periods=2, tz="UTC")
    df_open = pd.DataFrame({"AAPL": [100.0, 100.0]}, index=dates)
    df_close = pd.DataFrame({"AAPL": [100.0, 110.0]}, index=dates)

    sim = MarketSimulator(df_open, df_close, CostModel(commission_rate=0, slippage_rate=0, short_borrow_rate=0, margin_rate=0))
    sim.reset(initial_equity=100.0)

    transition = sim.step(target_long_weights=np.array([1.0]), target_short_weights=np.array([0.0]))
    assert transition.return_before_cost == pytest.approx(0.10)
    assert transition.next_state.equity == pytest.approx(110.0)


def test_half_exposure_survives_overnight():
    dates = pd.bdate_range("2020-01-02", periods=2, tz="UTC")
    df_open = pd.DataFrame({"AAPL": [100.0, 100.0]}, index=dates)
    df_close = pd.DataFrame({"AAPL": [100.0, 100.0]}, index=dates)

    sim = MarketSimulator(df_open, df_close, CostModel(0, 0, 0, 0))
    sim.reset(initial_equity=100.0)

    transition = sim.step(target_long_weights=np.array([0.5]), target_short_weights=np.array([0.0]))
    assert transition.next_state.gross_exposure == pytest.approx(0.5)
    assert transition.next_state.cash == pytest.approx(50.0)
    assert transition.next_state.equity == pytest.approx(100.0)


def test_costs_reduce_cash_and_equity_once():
    dates = pd.bdate_range("2020-01-02", periods=2, tz="UTC")
    df_open = pd.DataFrame({"AAPL": [100.0, 100.0]}, index=dates)
    df_close = pd.DataFrame({"AAPL": [100.0, 100.0]}, index=dates)

    cost_model = CostModel(commission_rate=0.001, slippage_rate=0.0, short_borrow_rate=0.0, margin_rate=0.0)
    sim = MarketSimulator(df_open, df_close, cost_model)
    sim.reset(initial_equity=100.0)

    transition = sim.step(target_long_weights=np.array([1.0]), target_short_weights=np.array([0.0]))
    # 100 traded * 0.001 = 0.10 cost
    assert transition.costs.total == pytest.approx(0.10)
    assert transition.next_state.equity == pytest.approx(99.90)


def test_different_actions_create_different_next_portfolios():
    dates = pd.bdate_range("2020-01-02", periods=2, tz="UTC")
    df_open = pd.DataFrame({"AAPL": [100.0, 100.0], "MSFT": [50.0, 50.0]}, index=dates)
    df_close = pd.DataFrame({"AAPL": [100.0, 100.0], "MSFT": [50.0, 50.0]}, index=dates)

    sim1 = MarketSimulator(df_open, df_close, CostModel(0, 0, 0, 0))
    sim1.reset(100.0)
    trans1 = sim1.step(np.array([1.0, 0.0]), np.zeros(2))

    sim2 = MarketSimulator(df_open, df_close, CostModel(0, 0, 0, 0))
    sim2.reset(100.0)
    trans2 = sim2.step(np.array([0.0, 1.0]), np.zeros(2))

    assert not np.array_equal(trans1.next_state.long_weights, trans2.next_state.long_weights)