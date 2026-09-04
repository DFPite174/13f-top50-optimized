"""Independent risk overlay, Top-K long allocation, and decoupled Bottom-M short hedging."""

from dataclasses import dataclass
from typing import Mapping, Sequence, Set
import numpy as np

PROTECTED_DEFENSIVE_ASSETS: Set[str] = {
    "SHY", "BIL", "GLD", "IAU", "SPY", "TLT", "IEF", "BND", "AGG", "VGSH"
}


@dataclass(frozen=True)
class TargetAction:
    long_weights: np.ndarray
    short_weights: np.ndarray
    net_exposure: float
    gross_exposure: float


class ActionProjector:
    def __init__(
        self,
        tickers: Sequence[str],
        top_k: int = 15,
        bottom_m: int = 8,
        bull_leverage: float = 1.25,
        base_leverage: float = 1.0,
        panic_scale: float = 0.50,
        max_single_weight: float = 0.20,
        conviction_power: float = 1.0,
        max_short_leverage: float = 0.10,
        inertia: float = 0.0,
    ) -> None:
        self.tickers = tuple(tickers)
        self.top_k = min(top_k, len(self.tickers))
        self.bottom_m = min(bottom_m, len(self.tickers))
        self.bull_leverage = bull_leverage
        self.base_leverage = base_leverage
        self.panic_scale = panic_scale
        self.max_single_weight = max_single_weight
        self.conviction_power = conviction_power
        self.max_short_leverage = max_short_leverage
        self.inertia = inertia

    def project(
        self,
        policy_weights: np.ndarray,
        exposure: float,
        macro: Mapping[str, float],
        stock_moms: np.ndarray,
        prev_weights: np.ndarray | None = None,
    ) -> TargetAction:
        n_assets = len(self.tickers)
        p_weights = np.asarray(policy_weights, dtype=float)

        gspc_mom = float(macro.get("gspc_mom", 0.0))
        vix_level = float(macro.get("vix_level", 20.0))
        vix_mom = float(macro.get("vix_mom", 0.0))

        # 1. Top-K Long Allocation with Conviction Sizing
        s_idx = np.argsort(p_weights)
        t_idx = s_idx[-self.top_k :]

        # Macro long scaling
        is_bull = (gspc_mom > 0.02) and (vix_level < 18.0)
        is_panic = (gspc_mom < -0.03) or (vix_level > 28.0) or (vix_mom > 0.20)

        if is_panic:
            long_scale = self.panic_scale
        elif is_bull:
            long_scale = self.bull_leverage
        else:
            long_scale = self.base_leverage

        effective_exposure = exposure * (long_scale / self.base_leverage)

        top_w = p_weights[t_idx]
        if self.conviction_power > 1.0:
            # Exponential rank tiering for realistic institutional conviction (Core ~10-12%, Satellite ~2-4%)
            ranks = np.arange(len(top_w))
            decay = np.exp(0.10 * self.conviction_power * ranks)
            top_norm = decay / np.sum(decay)
        else:
            top_sum = float(np.sum(top_w))
            top_norm = top_w / (top_sum + 1e-8) if top_sum > 0 else np.ones(len(t_idx)) / len(t_idx)

        raw_long = np.zeros(n_assets)
        raw_long[t_idx] = np.minimum(effective_exposure * top_norm, self.max_single_weight)

        # Turnover inertia smoothing: prevents destructive daily churn
        if self.inertia > 0 and prev_weights is not None and len(prev_weights) == n_assets and np.sum(prev_weights) > 0:
            long_weights = (1.0 - self.inertia) * raw_long + self.inertia * prev_weights
            # Renormalize to match target exposure
            tot = float(np.sum(long_weights))
            if tot > 1e-6:
                long_weights = long_weights * (effective_exposure / tot)
        else:
            long_weights = raw_long

        # 2. Guarded Independent Bottom-M Short Hedging
        short_weights = np.zeros(n_assets)
        hedge_trigger = (gspc_mom < -0.035 and vix_level > 28.0) or (gspc_mom < -0.02 and vix_mom > 0.25)

        if hedge_trigger and self.bottom_m > 0 and self.max_short_leverage > 0:
            short_lev = min(self.max_short_leverage, 0.06 if vix_level > 32.0 else 0.04)
            # Strict qualification for shorting: genuine crash deterioration
            qualifying = [
                idx
                for idx in s_idx
                if self.tickers[idx] not in PROTECTED_DEFENSIVE_ASSETS
                and stock_moms[idx] < -0.03
                and stock_moms[idx] < gspc_mom
            ]

            if qualifying:
                b_cand = qualifying[: self.bottom_m]
                cand_w = p_weights[b_cand]
                c_sum = float(np.sum(cand_w))
                c_norm = cand_w / (c_sum + 1e-8) if c_sum > 0 else np.ones(len(b_cand)) / len(b_cand)
                short_weights[b_cand] = -short_lev * c_norm

        net_exp = float(np.sum(long_weights) + np.sum(short_weights))
        gross_exp = float(np.sum(long_weights) + np.sum(np.abs(short_weights)))

        return TargetAction(
            long_weights=long_weights,
            short_weights=short_weights,
            net_exposure=net_exp,
            gross_exposure=gross_exp,
        )