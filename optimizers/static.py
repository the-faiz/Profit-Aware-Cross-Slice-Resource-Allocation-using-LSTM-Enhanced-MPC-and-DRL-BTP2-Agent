# StaticOptimizer: equal PRB split across users (fixed baseline).
from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseOptimizer


class StaticOptimizer(BaseOptimizer):
    def __init__(self, num_prbs: int):
        super().__init__(num_prbs)

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        return self._even_split(len(user_tiers), self.num_prbs).tolist()
