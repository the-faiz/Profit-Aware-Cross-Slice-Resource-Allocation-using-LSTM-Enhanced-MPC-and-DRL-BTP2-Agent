# RandomOptimizer: random PRB split baseline (normalized random weights).
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseOptimizer


class RandomOptimizer(BaseOptimizer):
    def __init__(self, num_prbs: int):
        super().__init__(num_prbs)
        self._rng = np.random.default_rng()

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        num_ues = len(user_tiers)
        if num_ues == 0:
            return []

        weights = self._rng.random(num_ues)
        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            return self._even_split(num_ues, self.num_prbs).tolist()

        alloc = np.floor(weights / weights_sum * self.num_prbs).astype(int)
        remainder = self.num_prbs - int(alloc.sum())
        if remainder > 0:
            idx = self._rng.choice(num_ues, size=remainder, replace=True)
            for i in idx:
                alloc[i] += 1
        return alloc.tolist()
