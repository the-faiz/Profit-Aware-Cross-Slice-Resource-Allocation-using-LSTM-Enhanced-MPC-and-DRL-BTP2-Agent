# TargetRateProportionalOptimizer: allocate PRBs proportional to required PRBs per user.
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from channel.channel_model import Channel
from .base import BaseOptimizer


class TargetRateProportionalOptimizer(BaseOptimizer):
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
        need = np.zeros(num_ues, dtype=float)
        for u in range(num_ues):
            tier = user_tiers[u]
            target = float(self.tiers_cfg[tier]["target_rate_mbps"]) * 1e6
            x, y = positions[u, 0], positions[u, 1]
            d_m = math.hypot(float(x) - self.ru_x, float(y) - self.ru_y) * 1000.0
            sinr = self.channel.compute_sinr_linear(distance_m=d_m)
            rate_per_prb = self.prb_bw * math.log2(1.0 + sinr)
            if rate_per_prb <= 0.0:
                need[u] = float(self.num_prbs)
            else:
                need[u] = min(float(self.num_prbs), target / rate_per_prb)

        total_need = float(need.sum())
        if total_need <= 0.0:
            return self._even_split(num_ues, self.num_prbs).tolist()

        alloc = np.floor(need / total_need * self.num_prbs).astype(int)
        remainder = self.num_prbs - int(alloc.sum())
        if remainder > 0:
            order = np.argsort(-need)[:remainder]
            for idx in order:
                alloc[idx] += 1
        return alloc.tolist()
