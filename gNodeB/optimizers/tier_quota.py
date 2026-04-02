# TierQuotaOptimizer: split PRBs by tier quotas, then equally within each tier.
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import BaseOptimizer


class TierQuotaOptimizer(BaseOptimizer):
    def __init__(self, num_prbs: int, tiers_cfg: Dict[str, dict]):
        super().__init__(num_prbs)
        self.tiers_cfg = tiers_cfg

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        num_ues = len(user_tiers)
        if num_ues == 0:
            return []

        # Group users by tier
        tier_to_indices: Dict[str, List[int]] = {}
        for idx, tier in enumerate(user_tiers):
            tier_to_indices.setdefault(tier, []).append(idx)

        # Compute tier mass = weight * count, use it to assign PRB quotas per tier
        tier_mass: Dict[str, float] = {}
        for tier, indices in tier_to_indices.items():
            weight = float(self.tiers_cfg[tier]["weight"])
            tier_mass[tier] = weight * len(indices)

        total_mass = float(sum(tier_mass.values()))
        if total_mass <= 0.0:
            return self._even_split(num_ues, self.num_prbs).tolist()

        # Initial quota by floor of proportional split
        tier_quota: Dict[str, int] = {}
        tier_frac: Dict[str, float] = {}
        for tier, mass in tier_mass.items():
            exact = mass / total_mass * self.num_prbs
            quota = int(np.floor(exact))
            tier_quota[tier] = quota
            tier_frac[tier] = exact - quota

        remainder = self.num_prbs - int(sum(tier_quota.values()))
        if remainder > 0:
            # Distribute remaining PRBs to tiers with largest fractional parts
            for tier in sorted(tier_frac, key=tier_frac.get, reverse=True)[:remainder]:
                tier_quota[tier] += 1

        # Allocate each tier's quota equally among its users
        alloc = np.zeros(num_ues, dtype=int)
        for tier, indices in tier_to_indices.items():
            quota = int(tier_quota.get(tier, 0))
            if quota <= 0:
                continue
            per_user = quota // len(indices)
            leftover = quota - per_user * len(indices)
            for idx in indices:
                alloc[idx] = per_user
            if leftover > 0:
                for idx in indices[:leftover]:
                    alloc[idx] += 1

        return alloc.tolist()
