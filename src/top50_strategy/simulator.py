"""Execution simulator with explicit cash, debt, borrow fees, and turnover accounting."""

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd

from top50_strategy.types import CostBreakdown, PortfolioState


@dataclass(frozen=True)
class CostModel:
    commission_rate: float = 0.00025
    slippage_rate: float = 0.0004
    short_borrow_rate: float = 0.02
    margin_rate: float = 0.04

    def calculate_trade_costs(self, traded_volume: float) -> tuple[float, float]:
        comm = traded_volume * self.commission_rate
        slip = traded_volume * self.slippage_rate
        return float(comm), float(slip)

    def calculate_holding_costs(self, short_notional: float, debt: float) -> tuple[float, float]:
        borrow = abs(short_notional) * (self.short_borrow_rate / 252.0)
        margin = debt * (self.margin_rate / 252.0)
        return float(borrow), float(margin)


@dataclass(frozen=True)
class Transition:
    date: pd.Timestamp
    prev_state: PortfolioState
    target_long_weights: np.ndarray
    target_short_weights: np.ndarray
    costs: CostBreakdown
    return_before_cost: float
    return_after_cost: float
    next_state: PortfolioState


class MarketSimulator:
    def __init__(
        self,
        prices_open: pd.DataFrame,
        prices_close: pd.DataFrame,
        cost_model: CostModel,
        initial_equity: float = 100.0,
    ) -> None:
        self.df_open = prices_open
        self.df_close = prices_close
        self.cost_model = cost_model
        self.initial_equity = float(initial_equity)
        self.dates = prices_close.index
        self.current_step = 1
        self.current_state = PortfolioState.from_values(
            date=self.dates[0],
            long_value=0.0,
            short_value=0.0,
            cash=self.initial_equity,
            debt=0.0,
            long_weights=np.zeros(len(self.df_close.columns)),
            short_weights=np.zeros(len(self.df_close.columns)),
        )

    def reset(self, initial_equity: float | None = None) -> PortfolioState:
        if initial_equity is not None:
            self.initial_equity = float(initial_equity)
        self.current_step = 1
        num_assets = len(self.df_close.columns)
        self.current_state = PortfolioState.from_values(
            date=self.dates[0],
            long_value=0.0,
            short_value=0.0,
            cash=self.initial_equity,
            debt=0.0,
            long_weights=np.zeros(num_assets),
            short_weights=np.zeros(num_assets),
        )
        return self.current_state

    def step(
        self,
        target_long_weights: np.ndarray,
        target_short_weights: np.ndarray,
    ) -> Transition:
        if self.current_step >= len(self.dates):
            raise IndexError("Simulator reached end of market dates")

        t_date = self.dates[self.current_step]
        prev_date = self.dates[self.current_step - 1]
        prev_s = self.current_state

        p_open = self.df_open.loc[t_date].values.astype(float)
        p_close = self.df_close.loc[t_date].values.astype(float)
        p_prev_close = self.df_close.loc[prev_date].values.astype(float)

        num_assets = len(p_open)
        t_long = np.asarray(target_long_weights, dtype=float)
        t_short = np.asarray(target_short_weights, dtype=float)

        # 1. Overnight return on previous holdings
        r_night = np.where(p_prev_close > 0, p_open / p_prev_close - 1.0, 0.0)
        r_night = np.nan_to_num(r_night, nan=0.0)

        # Intraday return from open to close
        r_day = np.where(p_open > 0, p_close / p_open - 1.0, 0.0)
        r_day = np.nan_to_num(r_day, nan=0.0)

        # Value of holdings at morning open before rebalance
        if len(prev_s.long_weights) == num_assets and np.sum(np.abs(prev_s.long_weights)) > 0:
            val_long_at_open = prev_s.long_weights * prev_s.equity * (1.0 + r_night)
        else:
            val_long_at_open = np.zeros(num_assets)

        if len(prev_s.short_weights) == num_assets and np.sum(np.abs(prev_s.short_weights)) > 0:
            # Short return: if price rises, short loses
            val_short_at_open = prev_s.short_weights * prev_s.equity * (1.0 + r_night)
        else:
            val_short_at_open = np.zeros(num_assets)

        equity_at_open = (
            float(np.sum(val_long_at_open))
            + float(np.sum(val_short_at_open))
            + prev_s.cash
            - prev_s.debt
        )
        if equity_at_open <= 0:
            equity_at_open = 1e-4

        # 2. Rebalance to target weights at open
        target_long_val = t_long * equity_at_open
        target_short_val = t_short * equity_at_open  # negative

        trade_long = np.abs(target_long_val - val_long_at_open)
        trade_short = np.abs(target_short_val - val_short_at_open)
        traded_volume = float(np.sum(trade_long) + np.sum(trade_short))

        comm, slip = self.cost_model.calculate_trade_costs(traded_volume)

        # 3. Cash balance adjustment after opening trades
        # Long purchase requires cash, short sale provides cash buffer
        cash_after_trades = prev_s.cash - float(np.sum(target_long_val - val_long_at_open)) - (comm + slip)
        debt_after_trades = prev_s.debt

        if cash_after_trades < 0:
            debt_after_trades += abs(cash_after_trades)
            cash_after_trades = 0.0

        # 4. Holding costs across day
        short_notional = float(np.sum(np.abs(target_short_val)))
        borrow_cost, margin_int = self.cost_model.calculate_holding_costs(short_notional, debt_after_trades)
        total_costs = CostBreakdown(comm, slip, borrow_cost, margin_int)

        cash_final = cash_after_trades - (borrow_cost + margin_int)
        debt_final = debt_after_trades
        if cash_final < 0:
            debt_final += abs(cash_final)
            cash_final = 0.0

        # 5. Mark to market at close
        final_long_val_vec = target_long_val * (1.0 + r_day)
        final_short_val_vec = target_short_val * (1.0 + r_day)

        final_long_val = float(np.sum(final_long_val_vec))
        final_short_val = float(np.sum(final_short_val_vec))
        final_equity = final_long_val + final_short_val + cash_final - debt_final

        # Returns
        eq_base = prev_s.equity if prev_s.equity > 1e-8 else 1.0
        ret_after = (final_equity - prev_s.equity) / eq_base
        ret_before = ret_after + (total_costs.total / eq_base)

        new_long_weights = final_long_val_vec / (final_equity + 1e-8) if final_equity > 0 else np.zeros(num_assets)
        new_short_weights = final_short_val_vec / (final_equity + 1e-8) if final_equity > 0 else np.zeros(num_assets)

        next_state = PortfolioState.from_values(
            date=t_date,
            long_value=final_long_val,
            short_value=final_short_val,
            cash=cash_final,
            debt=debt_final,
            long_weights=new_long_weights,
            short_weights=new_short_weights,
        )

        transition = Transition(
            date=t_date,
            prev_state=prev_s,
            target_long_weights=t_long,
            target_short_weights=t_short,
            costs=total_costs,
            return_before_cost=float(ret_before),
            return_after_cost=float(ret_after),
            next_state=next_state,
        )

        self.current_state = next_state
        self.current_step += 1
        return transition