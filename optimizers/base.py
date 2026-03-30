from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class BaseOptimizer(ABC):
    def __init__(self, num_prbs: int) -> None:
        self.num_prbs = num_prbs

    @abstractmethod
    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        raise NotImplementedError

    @staticmethod
    def _repair_to_budget(alloc: np.ndarray, budget: int) -> np.ndarray:
        alloc = alloc.astype(int, copy=True)
        total = int(alloc.sum())
        if total <= budget:
            return alloc
        excess = total - budget
        while excess > 0:
            candidates = np.flatnonzero(alloc > 0)
            if candidates.size == 0:
                break
            idx = int(np.random.choice(candidates))
            alloc[idx] -= 1
            excess -= 1
        return alloc

    @staticmethod
    def _even_split(num_ues: int, budget: int) -> np.ndarray:
        if num_ues <= 0:
            return np.zeros(0, dtype=int)
        base = budget // num_ues
        alloc = np.full(num_ues, base, dtype=int)
        remainder = budget - base * num_ues
        if remainder > 0:
            alloc[:remainder] += 1
        return alloc
