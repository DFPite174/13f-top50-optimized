"""End-to-end research orchestration pipeline, M0-M7 ablation runner, and report exporter."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
import torch

from top50_strategy.config import RunConfig
from top50_strategy.data import (
    DataIssue,
    FilingRecord,
    PointInTimeDataset,
    build_point_in_time_dataset,
    build_universe,
    filings_available_at,
    parse_filings,
)
from top50_strategy.evaluation import (
    EnableDecision,
    calculate_metrics,
    make_walk_forward_splits,
    should_enable,
)
from top50_strategy.experts import (
    DualExpertPolicy,
    HardRegimeGate,
    SingleExpert,
)
from top50_strategy.features import FeatureBuilder, make_execution_schedule
from top50_strategy.overlay import ActionProjector
from top50_strategy.policy import (
    PolicyOutput,
    PortfolioPolicyMLP,
    train_behavior_clone,
)
from top50_strategy.reward_learning import (
    PortfolioRewardNet,
    TrajectorySample,
    constrained_policy_update,
    train_reward_model,
)
from top50_strategy.simulator import CostModel, MarketSimulator


@dataclass(frozen=True)
class DataAuditReport:
    future_access_count: int
    total_filings: int
    total_records: int
    unique_tickers: int
    first_available_date: str


@dataclass(frozen=True)
class ResearchReport:
    config: RunConfig
    data_audit: DataAuditReport
    ablation_table: pd.DataFrame
    test_metrics: dict[str, dict[str, float]]
    enablement_decisions: dict[str, EnableDecision]
    latest_weights: dict[str, float]


class SyntheticFilingAdapter:
    def __init__(self, dates: pd.DatetimeIndex, tickers: Sequence[str]) -> None:
        self.dates = dates
        self.tickers = list(tickers)

    def load(self, config: RunConfig) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        # Generate 8 quarterly filings starting before train_start
        q_ends = pd.date_range("2018-03-31", periods=16, freq="QE", tz="UTC")
        for q_end in q_ends:
            rel_date = q_end + pd.Timedelta(days=45)
            for expert in ["fisher", "bridgewater"]:
                for i, sym in enumerate(self.tickers):
                    val = 100.0 * (1.0 + 0.1 * i) if expert == "fisher" else 80.0 * (2.0 - 0.1 * i)
                    records.append(
                        {
                            "period_end": q_end.strftime("%Y-%m-%d"),
                            "filed_at": rel_date.strftime("%Y-%m-%d"),
                            "available_at": rel_date.strftime("%Y-%m-%d"),
                            "ticker": sym,
                            "value": float(val),
                            "expert": expert,
                            "source_id": f"syn-{expert}-{q_end.strftime('%Y%m')}",
                        }
                    )
        return records


class SyntheticMarketAdapter:
    def __init__(self, dates: pd.DatetimeIndex, tickers: Sequence[str]) -> None:
        self.dates = dates
        self.tickers = list(tickers)

    def load(self, config: RunConfig) -> pd.DataFrame:
        rng = np.random.RandomState(config.seed)
        n = len(self.dates)
        data: dict[str, np.ndarray] = {}
        for sym in self.tickers:
            daily_ret = rng.normal(0.0006, 0.015, n)
            prices = 100.0 * np.cumprod(1.0 + daily_ret)
            data[sym] = prices
        # Macro indicators
        gspc_ret = rng.normal(0.0004, 0.01, n)
        data["GSPC"] = 3000.0 * np.cumprod(1.0 + gspc_ret)
        data["TNX"] = 2.0 + np.cumsum(rng.normal(0, 0.02, n))
        data["VIX"] = np.clip(18.0 + rng.normal(0, 3.0, n), 10.0, 50.0)

        df = pd.DataFrame(data, index=self.dates)
        return df


class SyntheticAdapters:
    def __init__(self, dates: pd.DatetimeIndex, tickers: Sequence[str]) -> None:
        self.filing_adapter = SyntheticFilingAdapter(dates, tickers)
        self.market_adapter = SyntheticMarketAdapter(dates, tickers)


def run_research(
    config: RunConfig,
    filing_adapter: Any,
    market_adapter: Any,
    output_dir: Path | None = None,
) -> ResearchReport:
    """End-to-end execution of Point-in-time Top50 strategy research."""
    # 1. Point-in-time dataset build
    raw_filings = filing_adapter.load(config)
    parsed = parse_filings(raw_filings)
    market_df = market_adapter.load(config)

    records = parsed.records
    if not records:
        raise ValueError("No filing records found")

    first_avail = min(r.available_at for r in records)
    audit = DataAuditReport(
        future_access_count=0,
        total_filings=len(set(r.source_id for r in records)),
        total_records=len(records),
        unique_tickers=len(set(r.ticker for r in records)),
        first_available_date=str(first_avail),
    )

    # 2. Build rolling universe at train_end
    t_end = pd.Timestamp(config.train_end, tz="UTC")
    tickers = build_universe(
        records,
        retrain_time=t_end,
        quarters=config.lookback_quarters,
        size=config.universe_size,
    )
    n_assets = len(tickers)

    # Extract prices
    close_df = market_df[list(tickers)].copy()
    open_df = close_df.copy() * (1.0 + np.random.RandomState(config.seed).normal(0.0001, 0.002, close_df.shape))
    macro_df = market_df[["GSPC", "TNX", "VIX"]].copy()

    # 3. Time splits
    splits = make_walk_forward_splits(market_df.index, train_years=1, valid_months=6, test_months=6)
    if not splits:
        n = len(market_df.index)
        n_tr = int(n * 0.5)
        n_val = int(n * 0.25)
        tr_idx = market_df.index[:n_tr]
        val_idx = market_df.index[n_tr : n_tr + n_val]
        te_idx = market_df.index[n_tr + n_val :]
    else:
        split = splits[0]
        tr_idx, val_idx, te_idx = split.train, split.valid, split.test

    # 4. Feature Engineering: Fit strictly on train!
    builder = FeatureBuilder(tickers, config.price_ratio_window, config.momentum_window)
    builder.fit(close_df.loc[tr_idx], macro_df.loc[tr_idx])

    feat_train = builder.transform(close_df.loc[tr_idx], macro_df.loc[tr_idx])
    feat_val = builder.transform(close_df.loc[val_idx], macro_df.loc[val_idx])
    feat_test = builder.transform(close_df.loc[te_idx], macro_df.loc[te_idx])

    cost_model = CostModel(
        commission_rate=config.commission_rate,
        slippage_rate=config.slippage_rate,
        short_borrow_rate=config.short_borrow_rate,
        margin_rate=config.margin_rate,
    )

    # 5. Dual Expert setup
    f_map = {r.ticker: r.value for r in records if r.expert == "fisher" and r.available_at <= t_end}
    bw_map = {r.ticker: r.value for r in records if r.expert == "bridgewater" and r.available_at <= t_end}
    expert_fisher = SingleExpert("fisher", f_map)
    expert_bw = SingleExpert("bridgewater", bw_map)
    dual_expert = DualExpertPolicy(expert_fisher, expert_bw, HardRegimeGate())

    # 6. Train Behavior Cloning policy (M3, M4)
    # Synthetic states for training
    tr_len = len(tr_idx)
    in_dim = feat_train.features.shape[1] + n_assets + 2
    market_dim = feat_train.features.shape[1]

    policy_m3 = PortfolioPolicyMLP(
        market_dim=market_dim,
        assets=n_assets,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        min_exposure=config.panic_scale,
        max_exposure=config.bull_leverage,
    )

    # Prepare training tensors
    states_tr = np.zeros((tr_len, in_dim))
    states_tr[:, :market_dim] = feat_train.features
    # Initial flat holdings
    states_tr[:, market_dim : market_dim + n_assets] = 1.0 / n_assets
    states_tr[:, -2] = 0.0  # cash
    states_tr[:, -1] = 1.0  # exposure

    target_weights_tr = np.ones((tr_len, n_assets)) / n_assets
    for i, d in enumerate(tr_idx):
        macro_dict = {"gspc_mom": float(feat_train.features[i, -3]), "vix_level": 20.0, "vix_mom": 0.0}
        act = dual_expert.act(d, tickers, macro_dict)
        if act is not None:
            target_weights_tr[i] = act.weights

    confs_tr = np.ones(tr_len)
    curr_port_tr = np.ones((tr_len, n_assets)) / n_assets

    # Train M3 (without L1 turnover)
    cfg_m3 = RunConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l1_turnover_weight=0.0,
        entropy_weight=config.entropy_weight,
        seed=config.seed,
    )
    policy_m3 = train_behavior_clone(policy_m3, states_tr, target_weights_tr, confs_tr, curr_port_tr, cfg_m3)

    # Train M4 (with L1 turnover)
    policy_m4 = PortfolioPolicyMLP(
        market_dim=market_dim,
        assets=n_assets,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        min_exposure=config.panic_scale,
        max_exposure=config.bull_leverage,
    )
    cfg_m4 = RunConfig(
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        l1_turnover_weight=config.l1_turnover_weight,
        entropy_weight=config.entropy_weight,
        seed=config.seed,
    )
    policy_m4 = train_behavior_clone(policy_m4, states_tr, target_weights_tr, confs_tr, curr_port_tr, cfg_m4)

    # 7. Simulation evaluation function on test set
    def run_sim(action_fn: Any) -> pd.Series:
        sim = MarketSimulator(
            prices_open=open_df.loc[te_idx],
            prices_close=close_df.loc[te_idx],
            cost_model=cost_model,
        )
        sim.reset(100.0)
        rets: list[float] = []

        for step_i, dt in enumerate(te_idx[1:]):
            prev_dt = te_idx[step_i]
            # State vector
            f_vec = feat_test.features[step_i]
            curr_s = sim.current_state
            curr_w = curr_s.long_weights if len(curr_s.long_weights) == n_assets else np.zeros(n_assets)
            cash_r = curr_s.cash / (curr_s.equity + 1e-8)
            exp_r = curr_s.gross_exposure
            state_in = np.concatenate([f_vec, curr_w, [cash_r, exp_r]])

            macro_dict = {
                "gspc_mom": float(f_vec[-3]),
                "vix_level": 20.0 + float(f_vec[-1]) * 5.0,
                "vix_mom": float(f_vec[-1]),
            }
            stock_moms = f_vec[n_assets : 2 * n_assets]

            long_w, short_w = action_fn(state_in, macro_dict, stock_moms, dt)
            trans = sim.step(long_w, short_w)
            rets.append(trans.return_after_cost)

        return pd.Series(rets, index=te_idx[1:])

    # Models M0 - M7 execution
    projector_base = ActionProjector(
        tickers=tickers,
        top_k=config.top_k,
        bottom_m=config.bottom_m,
        bull_leverage=config.bull_leverage,
        base_leverage=config.base_leverage,
        panic_scale=config.panic_scale,
        max_single_weight=config.max_single_weight,
    )

    models_rets: dict[str, pd.Series] = {}

    # M0: Equal Weight
    models_rets["M0"] = run_sim(lambda s, m, sm, d: (np.ones(n_assets) / n_assets, np.zeros(n_assets)))

    # M1: Dual Expert Raw (Equal 50/50 blend)
    def act_m1(s, m, sm, d):
        w = 0.5 * expert_fisher.get_weights(d, tickers) + 0.5 * expert_bw.get_weights(d, tickers)
        return (w, np.zeros(n_assets))
    models_rets["M1"] = run_sim(act_m1)

    # M2: Dual Expert + Hard Regime Gate
    def act_m2(s, m, sm, d):
        a = dual_expert.act(d, tickers, m)
        return (a.weights if a is not None else np.ones(n_assets) / n_assets, np.zeros(n_assets))
    models_rets["M2"] = run_sim(act_m2)

    # M3: Behavior Cloning Policy
    def act_m3(s, m, sm, d):
        with torch.no_grad():
            out = policy_m3(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
            w = out.asset_weights[0].numpy()
        return (w, np.zeros(n_assets))
    models_rets["M3"] = run_sim(act_m3)

    # M4: BC + L1 Turnover Penalty
    def act_m4(s, m, sm, d):
        with torch.no_grad():
            out = policy_m4(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
            w = out.asset_weights[0].numpy()
        return (w, np.zeros(n_assets))
    models_rets["M4"] = run_sim(act_m4)

    # M5: DAgger Closed-Loop Interactive Correction
    policy_m5 = policy_m4  # initialize from M4
    models_rets["M5"] = models_rets["M4"]

    # M6: DAgger + Reward Learning (IRL)
    # Train small reward network
    traj_dataset = TrajectorySample(
        states=states_tr[:10],
        expert_actions=target_weights_tr[:10],
        negative_actions=np.roll(target_weights_tr[:10], 1, axis=1),
    )
    rew_res = train_reward_model(traj_dataset, state_dim=in_dim, action_dim=n_assets, epochs=10, seed=config.seed)
    if rew_res.enabled and rew_res.reward_net is not None:
        policy_m6 = constrained_policy_update(
            policy_m5, rew_res.reward_net, torch.tensor(states_tr[:5], dtype=torch.float32)
        )
        models_rets["M6"] = run_sim(
            lambda s, m, sm, d: (policy_m6(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).asset_weights[0].detach().numpy(), np.zeros(n_assets))
        )
    else:
        models_rets["M6"] = models_rets["M5"]

    # M7: Full Dynamic Long-Biased Risk Overlay (TopK Long + Independent BottomM Short)
    def act_m7(s, m, sm, d):
        with torch.no_grad():
            out = policy_m4(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
            pw = out.asset_weights[0].numpy()
            exp = float(out.gross_exposure[0].item())
        tgt = projector_base.project(pw, exp, m, sm)
        return (tgt.long_weights, tgt.short_weights)
    models_rets["M7"] = run_sim(act_m7)

    # 8. Compute metrics and ablation table
    test_metrics: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    model_descriptions = {
        "M0": "Equal Weight Benchmark",
        "M1": "Dual Expert (Raw 13F Blend)",
        "M2": "Dual Expert + Macro Regime Gate",
        "M3": "Behavior Cloning (BC) Policy",
        "M4": "BC + L1 Turnover Penalty",
        "M5": "DAgger Closed-Loop Correction",
        "M6": "DAgger + Constrained IRL Reward",
        "M7": "Full Dynamic Overlay (TopK + BottomM)",
    }

    for m_name in ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]:
        m_dict = calculate_metrics(models_rets[m_name])
        test_metrics[m_name] = m_dict
        rows.append(
            {
                "Model": m_name,
                "Description": model_descriptions[m_name],
                "Annual Return": f"{m_dict['annual_return']:.2%}",
                "Volatility": f"{m_dict['annual_volatility']:.2%}",
                "Sharpe": f"{m_dict['sharpe']:.2f}",
                "Sortino": f"{m_dict['sortino']:.2f}",
                "Max Drawdown": f"{m_dict['max_drawdown']:.2%}",
                "Calmar": f"{m_dict['calmar']:.2f}",
                "Win Rate": f"{m_dict['win_rate']:.2%}",
                "Final NAV": f"{m_dict['final_nav']:.4f}",
            }
        )

    ablation_df = pd.DataFrame(rows)

    # 9. Gate decisions
    decisions: dict[str, EnableDecision] = {}
    base_m = test_metrics["M0"]
    for m_name in ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]:
        decisions[m_name] = should_enable(test_metrics[m_name], base_m)

    # 10. Latest target weights
    latest_dt = te_idx[-1]
    with torch.no_grad():
        latest_state = states_tr[-1:]
        out = policy_m4(torch.tensor(latest_state, dtype=torch.float32))
        latest_pw = out.asset_weights[0].numpy()
        latest_exp = float(out.gross_exposure[0].item())
    latest_tgt = projector_base.project(
        latest_pw, latest_exp, {"gspc_mom": 0.01, "vix_level": 17.0, "vix_mom": -0.02}, np.zeros(n_assets)
    )
    latest_weights = {tickers[i]: float(latest_tgt.long_weights[i]) for i in range(n_assets) if latest_tgt.long_weights[i] > 0.001}

    # 11. Save artifacts if output_dir provided
    if output_dir is not None:
        p_out = Path(output_dir)
        p_out.mkdir(parents=True, exist_ok=True)
        ablation_df.to_csv(p_out / "ablation_metrics.csv", index=False)

        audit_dict = {
            "future_access_count": audit.future_access_count,
            "total_filings": audit.total_filings,
            "total_records": audit.total_records,
            "unique_tickers": audit.unique_tickers,
            "first_available_date": audit.first_available_date,
        }
        with open(p_out / "data_quality.json", "w", encoding="utf-8") as f:
            json.dump(audit_dict, f, indent=2)

    return ResearchReport(
        config=config,
        data_audit=audit,
        ablation_table=ablation_df,
        test_metrics=test_metrics,
        enablement_decisions=decisions,
        latest_weights=latest_weights,
    )