# DeficitAwareOptimizer: allocate more PRBs to tiers/users with higher required PRBs.
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from channel.channel_model import Channel
from .base import BaseOptimizer


class DeficitAwareOptimizer(BaseOptimizer):
    def __init__(
        self,
        num_prbs: int,
        prb_bw: float,
        ru_x: float,
        ru_y: float,
        channel: Channel,
        tiers_cfg: Dict[str, dict],
    ):
        super().__init__(num_prbs)
        self.prb_bw = prb_bw
        self.ru_x = ru_x
        self.ru_y = ru_y
        self.channel = channel
        self.tiers_cfg = tiers_cfg

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        num_ues = len(user_tiers)
        if num_ues == 0:
            return []

        positions = pred_positions[:, 0, :]

        # Estimate per-user PRBs needed to meet target rate.
        user_need = np.zeros(num_ues, dtype=float)
        for u in range(num_ues):
            tier = user_tiers[u]
            target = float(self.tiers_cfg[tier]["target_rate_mbps"]) * 1e6
            x, y = positions[u, 0], positions[u, 1]
            d_m = math.hypot(float(x) - self.ru_x, float(y) - self.ru_y) * 1000.0
            sinr = self.channel.compute_sinr_linear(distance_m=d_m)
            rate_per_prb = self.prb_bw * math.log2(1.0 + sinr)
            if rate_per_prb <= 0.0:
                user_need[u] = float(self.num_prbs)
            else:
                user_need[u] = min(float(self.num_prbs), target / rate_per_prb)

        # Group users by tier for tier-level quotas.
        tier_to_indices: Dict[str, List[int]] = {}
        for idx, tier in enumerate(user_tiers):
            tier_to_indices.setdefault(tier, []).append(idx)

        tier_need: Dict[str, float] = {}
        for tier, indices in tier_to_indices.items():
            tier_need[tier] = float(user_need[indices].sum())

        total_need = float(sum(tier_need.values()))
        if total_need <= 0.0:
            return self._even_split(num_ues, self.num_prbs).tolist()

        # Allocate tier quotas proportional to tier need.
        tier_quota: Dict[str, int] = {}
        tier_frac: Dict[str, float] = {}
        for tier, need in tier_need.items():
            exact = need / total_need * self.num_prbs
            quota = int(np.floor(exact))
            tier_quota[tier] = quota
            tier_frac[tier] = exact - quota

        remainder = self.num_prbs - int(sum(tier_quota.values()))
        if remainder > 0:
            for tier in sorted(tier_frac, key=tier_frac.get, reverse=True)[:remainder]:
                tier_quota[tier] += 1

        # Allocate each tier's quota across its users proportional to need.
        alloc = np.zeros(num_ues, dtype=int)
        for tier, indices in tier_to_indices.items():
            quota = int(tier_quota.get(tier, 0))
            if quota <= 0:
                continue

            needs = user_need[indices]
            needs_sum = float(needs.sum())
            if needs_sum <= 0.0:
                per_user = quota // len(indices)
                leftover = quota - per_user * len(indices)
                for idx in indices:
                    alloc[idx] = per_user
                for idx in indices[:leftover]:
                    alloc[idx] += 1
                continue

            base = np.floor(needs / needs_sum * quota).astype(int)
            remainder = quota - int(base.sum())
            alloc[indices] = base
            if remainder > 0:
                order = np.argsort(-needs)[:remainder]
                for i in order:
                    alloc[indices[i]] += 1

        return alloc.tolist()
