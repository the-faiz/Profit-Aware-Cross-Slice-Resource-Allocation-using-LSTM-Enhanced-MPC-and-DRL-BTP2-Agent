# TopKPriorityOptimizer: allocate most PRBs to users with worst channel (lowest SINR).
from __future__ import annotations

import math
from typing import List

import numpy as np

from channel.channel_model import Channel
from .base import BaseOptimizer


class TopKPriorityOptimizer(BaseOptimizer):
    def __init__(self, num_prbs: int, ru_x: float, ru_y: float, channel: Channel):
        super().__init__(num_prbs)
        self.ru_x = ru_x
        self.ru_y = ru_y
        self.channel = channel

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        num_ues = len(user_tiers)
        if num_ues == 0:
            return []

        positions = pred_positions[:, 0, :]
        sinrs = np.zeros(num_ues, dtype=float)
        for u in range(num_ues):
            x, y = positions[u, 0], positions[u, 1]
            d_m = math.hypot(float(x) - self.ru_x, float(y) - self.ru_y) * 1000.0
            sinrs[u] = self.channel.compute_sinr_linear(distance_m=d_m)

        # Lower SINR is worse. Rank ascending and allocate proportionally with a bias to worst users.
        order = np.argsort(sinrs)
        weights = np.arange(num_ues, 0, -1, dtype=float)
        weights = weights / weights.sum()
        alloc = np.zeros(num_ues, dtype=int)
        base = np.floor(weights * self.num_prbs).astype(int)
        alloc[order] = base
        remainder = self.num_prbs - int(alloc.sum())
        if remainder > 0:
            for idx in order[:remainder]:
                alloc[idx] += 1

        return alloc.tolist()
