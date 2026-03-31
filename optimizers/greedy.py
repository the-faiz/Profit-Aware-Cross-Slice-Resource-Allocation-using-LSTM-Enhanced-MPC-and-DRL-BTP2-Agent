# GreedyOptimizer: iteratively assigns each PRB to the user with max marginal reward gain.
from __future__ import annotations

from typing import List

import numpy as np

from utilities.reward import compute_reward_scalar
from .base import BaseOptimizer


class GreedyOptimizer(BaseOptimizer):
    def __init__(
        self,
        num_prbs,
        prb_cost,
        prb_bw,
        gamma,
        horizon,
        ru_x,
        ru_y,
        channel,
        tiers_cfg,
        penalty_lambda,
        profit_sensitivity,
        satisfaction_sensitivity,
    ):
        super().__init__(num_prbs)
        self.prb_cost = prb_cost
        self.prb_bw = prb_bw
        self.gamma = gamma
        self.horizon = horizon
        self.ru_x = ru_x
        self.ru_y = ru_y
        self.channel = channel
        self.tiers_cfg = tiers_cfg
        self.penalty_lambda = penalty_lambda
        self.profit_sensitivity = profit_sensitivity
        self.satisfaction_sensitivity = satisfaction_sensitivity

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        num_ues = len(user_tiers)
        if num_ues == 0:
            return []

        alloc = np.zeros(num_ues, dtype=int)
        positions = pred_positions[:, 0, :]

        for _ in range(self.num_prbs):
            best_u = 0
            best_gain = -float("inf")
            current_reward = compute_reward_scalar(
                positions,
                alloc,
                user_tiers,
                self.tiers_cfg,
                self.ru_x,
                self.ru_y,
                self.prb_bw,
                self.prb_cost,
                self.penalty_lambda,
                self.profit_sensitivity,
                self.satisfaction_sensitivity,
                self.channel,
            )
            for u in range(num_ues):
                alloc[u] += 1
                new_reward = compute_reward_scalar(
                    positions,
                    alloc,
                    user_tiers,
                    self.tiers_cfg,
                    self.ru_x,
                    self.ru_y,
                    self.prb_bw,
                    self.prb_cost,
                    self.penalty_lambda,
                    self.profit_sensitivity,
                    self.satisfaction_sensitivity,
                    self.channel,
                )
                gain = new_reward - current_reward
                alloc[u] -= 1
                if gain > best_gain:
                    best_gain = gain
                    best_u = u

            alloc[best_u] += 1

        return alloc.tolist()
