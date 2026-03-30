# AverageOptimizer: allocates PRBs proportional to tier weights (weighted fair share baseline).
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseOptimizer


class AverageOptimizer(BaseOptimizer):
    def __init__(self, num_prbs: int, tiers_cfg):
        super().__init__(num_prbs)
        self.tiers_cfg = tiers_cfg

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        num_ues = len(user_tiers)
        if num_ues == 0:
            return []

        weights = np.array([float(self.tiers_cfg[t]["weight"]) for t in user_tiers], dtype=float)
        if weights.sum() <= 0:
            return self._even_split(num_ues, self.num_prbs).tolist()

        alloc = np.floor(weights / weights.sum() * self.num_prbs).astype(int)
        remainder = self.num_prbs - int(alloc.sum())
        if remainder > 0:
            idx = np.argsort(-weights)[:remainder]
            for i in idx:
                alloc[i] += 1
        return alloc.tolist()
