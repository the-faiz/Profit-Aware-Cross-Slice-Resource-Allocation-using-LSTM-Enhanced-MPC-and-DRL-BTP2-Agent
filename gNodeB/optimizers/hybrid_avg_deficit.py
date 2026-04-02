# HybridAverageDeficitOptimizer: start from average allocation, shift a small budget to deficit tiers.
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from channel.channel_model import Channel
from .base import BaseOptimizer


class HybridAverageDeficitOptimizer(BaseOptimizer):
    def __init__(
        self,
        num_prbs: int,
        prb_bw: float,
        ru_x: float,
        ru_y: float,
        channel: Channel,
        tiers_cfg: Dict[str, dict],
        shift_frac: float = 0.2,
    ):
        super().__init__(num_prbs)
        self.prb_bw = prb_bw
        self.ru_x = ru_x
        self.ru_y = ru_y
        self.channel = channel
        self.tiers_cfg = tiers_cfg
        self.shift_frac = max(0.0, min(1.0, float(shift_frac)))

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        num_ues = len(user_tiers)
        if num_ues == 0:
            return []

        # Average allocation by tier weights.
        weights = np.array([float(self.tiers_cfg[t]["weight"]) for t in user_tiers], dtype=float)
        if weights.sum() <= 0:
            alloc = self._even_split(num_ues, self.num_prbs).astype(int)
        else:
            alloc = np.floor(weights / weights.sum() * self.num_prbs).astype(int)
            remainder = self.num_prbs - int(alloc.sum())
            if remainder > 0:
                idx = np.argsort(-weights)[:remainder]
                for i in idx:
                    alloc[i] += 1

        # Compute satisfaction shortfall per tier (using h=0 positions).
        positions = pred_positions[:, 0, :]
        tier_to_indices: Dict[str, List[int]] = {}
        for idx, tier in enumerate(user_tiers):
            tier_to_indices.setdefault(tier, []).append(idx)

        tier_deficit: Dict[str, float] = {}
        for tier, indices in tier_to_indices.items():
            required = float(self.tiers_cfg[tier]["min_percent_of_user_to_satisfy"]) * len(indices)
            sat_sum = 0.0
            for u in indices:
                target = float(self.tiers_cfg[tier]["target_rate_mbps"]) * 1e6
                x, y = positions[u, 0], positions[u, 1]
                d_m = math.hypot(float(x) - self.ru_x, float(y) - self.ru_y) * 1000.0
                sinr = self.channel.compute_sinr_linear(distance_m=d_m)
                rate_per_prb = self.prb_bw * math.log2(1.0 + sinr)
                data_rate = float(alloc[u]) * rate_per_prb
                sat_sum += min(1.0, data_rate / target)
            tier_deficit[tier] = max(0.0, required - sat_sum)

        total_deficit = float(sum(tier_deficit.values()))
        if total_deficit <= 0.0 or self.shift_frac <= 0.0:
            return alloc.tolist()

        shift_budget = int(round(self.num_prbs * self.shift_frac))
        if shift_budget <= 0:
            return alloc.tolist()

        # Pull PRBs from users in tiers with lowest deficit.
        donors = []
        for tier, indices in tier_to_indices.items():
            deficit = tier_deficit.get(tier, 0.0)
            for u in indices:
                donors.append((deficit, u))
        donors.sort(key=lambda x: x[0])  # low deficit first

        # Give PRBs to users in tiers with highest deficit.
        recipients = []
        for tier, indices in tier_to_indices.items():
            deficit = tier_deficit.get(tier, 0.0)
            for u in indices:
                recipients.append((deficit, u))
        recipients.sort(key=lambda x: x[0], reverse=True)  # high deficit first

        di = 0
        ri = 0
        moved = 0
        while moved < shift_budget and di < len(donors) and ri < len(recipients):
            _, du = donors[di]
            _, ru = recipients[ri]
            if alloc[du] > 0:
                alloc[du] -= 1
                alloc[ru] += 1
                moved += 1
            di += 1
            if di >= len(donors):
                di = 0
            ri += 1
            if ri >= len(recipients):
                ri = 0

        return alloc.tolist()
