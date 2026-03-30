# PSOOptimizer: particle swarm optimization over allocation weights per horizon step.
from __future__ import annotations

from typing import Dict, List

import numpy as np

from utilities.reward import compute_reward_scalar
from .base import BaseOptimizer


class PSOOptimizer(BaseOptimizer):
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
        pso_config: Dict[str, float] | None = None,
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

        cfg = pso_config or {}
        self.pso_particles = int(cfg.get("particles", 30))
        self.pso_iters = int(cfg.get("iterations", 40))
        self.pso_inertia = float(cfg.get("inertia", 0.7))
        self.pso_c1 = float(cfg.get("c1", 1.4))
        self.pso_c2 = float(cfg.get("c2", 1.4))

        self._rng = np.random.default_rng()

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        num_ues = len(user_tiers)
        if num_ues == 0 or self.horizon == 0:
            return []

        dim = num_ues * self.horizon
        particles = self._rng.random((self.pso_particles, dim))
        velocities = self._rng.normal(scale=0.1, size=(self.pso_particles, dim))

        pbest = particles.copy()
        pbest_scores = np.full(self.pso_particles, -float("inf"), dtype=float)
        gbest = None
        gbest_score = -float("inf")

        for _ in range(self.pso_iters):
            for i in range(self.pso_particles):
                alloc = self._decode_allocation(particles[i], num_ues)
                score = self._fitness(alloc, pred_positions, user_tiers)
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = particles[i].copy()
                if score > gbest_score:
                    gbest_score = score
                    gbest = particles[i].copy()

            if gbest is None:
                break

            r1 = self._rng.random((self.pso_particles, dim))
            r2 = self._rng.random((self.pso_particles, dim))
            velocities = (
                self.pso_inertia * velocities
                + self.pso_c1 * r1 * (pbest - particles)
                + self.pso_c2 * r2 * (gbest - particles)
            )
            particles = particles + velocities
            particles = np.clip(particles, 0.0, 1.0)

        if gbest is None:
            return [0] * num_ues

        best_alloc = self._decode_allocation(gbest, num_ues)
        return best_alloc[:, 0].tolist()

    def _decode_allocation(self, vec: np.ndarray, num_ues: int) -> np.ndarray:
        alloc = np.zeros((num_ues, self.horizon), dtype=int)
        for h in range(self.horizon):
            start = h * num_ues
            end = start + num_ues
            weights = vec[start:end]
            weights_sum = float(weights.sum())
            if weights_sum <= 0:
                alloc[:, h] = 0
                continue
            alloc[:, h] = np.floor(weights / weights_sum * self.num_prbs).astype(int)
            remainder = self.num_prbs - int(alloc[:, h].sum())
            if remainder > 0:
                idx = np.argsort(-weights)[:remainder]
                for i in idx:
                    alloc[i, h] += 1
        return alloc

    def _fitness(
        self,
        alloc: np.ndarray,
        pred_positions: np.ndarray,
        user_tiers: List[str],
    ) -> float:
        total = 0.0
        for h in range(self.horizon):
            disc = self.gamma**h
            n_h = alloc[:, h]
            reward_h = compute_reward_scalar(
                pred_positions[:, h, :],
                n_h,
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
            total += disc * reward_h
        return float(total)
