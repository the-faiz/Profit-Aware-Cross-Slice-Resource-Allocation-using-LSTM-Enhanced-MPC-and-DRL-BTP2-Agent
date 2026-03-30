# RLOptimizer: uses a policy model if provided; otherwise falls back to greedy.
from __future__ import annotations

from typing import List

import numpy as np

from utilities.reward import compute_reward_scalar
from .base import BaseOptimizer

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


class RLOptimizer(BaseOptimizer):
    """
    Simple RL wrapper.

    If a policy model is provided, it is expected to take a tensor of shape (U, 2)
    and return a score per user. If no model is provided, falls back to greedy.
    """

    def __init__(
        self,
        num_prbs,
        prb_cost,
        prb_bw,
        ru_x,
        ru_y,
        channel,
        tiers_cfg,
        penalty_lambda,
        profit_sensitivity,
        satisfaction_sensitivity,
        rl_config=None,
    ):
        super().__init__(num_prbs)
        self.prb_cost = prb_cost
        self.prb_bw = prb_bw
        self.ru_x = ru_x
        self.ru_y = ru_y
        self.channel = channel
        self.tiers_cfg = tiers_cfg
        self.penalty_lambda = penalty_lambda
        self.profit_sensitivity = profit_sensitivity
        self.satisfaction_sensitivity = satisfaction_sensitivity
        self.rl_config = rl_config or {}

        self.policy = None
        policy_path = self.rl_config.get("policy_path")
        if policy_path and torch is not None:
            self.policy = torch.load(policy_path, map_location="cpu")
            if hasattr(self.policy, "eval"):
                self.policy.eval()

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        num_ues = len(user_tiers)
        if num_ues == 0:
            return []

        positions = pred_positions[:, 0, :]

        if self.policy is None:
            # Fallback to greedy behavior if no model.
            alloc = np.zeros(num_ues, dtype=int)
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

        if torch is None:
            return self._even_split(num_ues, self.num_prbs).tolist()

        with torch.no_grad():
            x = torch.from_numpy(positions).float()
            scores = self.policy(x).detach().cpu().numpy()

        scores = np.asarray(scores).reshape(-1)
        if scores.size != num_ues:
            return self._even_split(num_ues, self.num_prbs).tolist()

        scores = np.maximum(scores, 0.0)
        if scores.sum() <= 0:
            return self._even_split(num_ues, self.num_prbs).tolist()

        alloc = np.floor(scores / scores.sum() * self.num_prbs).astype(int)
        remainder = self.num_prbs - int(alloc.sum())
        if remainder > 0:
            idx = np.argsort(-scores)[:remainder]
            for i in idx:
                alloc[i] += 1
        return alloc.tolist()
