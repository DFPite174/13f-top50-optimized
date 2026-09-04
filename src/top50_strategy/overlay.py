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
    ) -> None:
        self.tickers = tuple(tickers)
        self.top_k = min(top_k, len(self.tickers))
        self.bottom_m = min(bottom_m, len(self.tickers))
        self.bull_leverage = bull_leverage
        self.base_leverage = base_leverage
        self.panic_scale = panic_scale
        self.max_single_weight = max_single_weight

    def project(
        self,
        policy_weights: np.ndarray,
        exposure: float,
        macro: Mapping[str, float],
        stock_moms: np.ndarray,
    ) -> TargetAction:
        n_assets = len(self.tickers)
        p_weights = np.asarray(policy_weights, dtype=float)

        gspc_mom = float(macro.get("gspc_mom", 0.0))
        vix_level = float(macro.get("vix_level", 20.0))
        vix_mom = float(macro.get("vix_mom", 0.0))

        # 1. Top-K Long Allocation
        s_idx = np.argsort(p_weights)
        t_idx = s_idx[-self.top_k :]

        # Determine macro long scaling
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
        top_sum = float(np.sum(top_w))
        top_norm = top_w / (top_sum + 1e-8) if top_sum > 0 else np.ones(len(t_idx)) / len(t_idx)

        long_weights = np.zeros(n_assets)
        long_weights[t_idx] = np.minimum(effective_exposure * top_norm, self.max_single_weight)

        # 2. Independent Bottom-M Short Hedging (Decoupled from Long Softmax)
        short_weights = np.zeros(n_assets)
        hedge_trigger = (gspc_mom < -0.015) or (vix_level > 24.0) or (gspc_mom < 0.0 and vix_mom > 0.12)

        if hedge_trigger and self.bottom_m > 0:
            short_lev = 0.32 if vix_level > 28.0 else 0.20
            # Strict qualification for shorting:
            # - Not in protected assets
            # - Absolute negative momentum
            # - Underperforming the index
            qualifying = [
                idx
                for idx in s_idx
                if self.tickers[idx] not in PROTECTED_DEFENSIVE_ASSETS
                and stock_moms[idx] < 0.0
                and stock_moms[idx] < gspc_mom
            ]

            if qualifying:
                # Select up to bottom_m candidates with lowest policy support
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