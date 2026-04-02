# GAOptimizer: genetic algorithm over PRB allocations across the prediction horizon.
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from channel.channel_model import Channel
from utilities.reward import compute_reward_scalar
from .heuristics import warm_start_allocations
from .base import BaseOptimizer


class GAOptimizer(BaseOptimizer):
    def __init__(
        self,
        num_prbs,
        prb_cost,
        prb_bw,
        gamma,
        horizon,
        ru_x,
        ru_y,
        channel: Channel,
        tiers_cfg,
        penalty_lambda,
        profit_sensitivity,
        satisfaction_sensitivity,
        ga_config: Dict[str, float] | None = None,
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

        cfg = ga_config or {}
        self.ga_population = int(cfg.get("population", 40))
        self.ga_generations = int(cfg.get("generations", 40))
        self.ga_elite_frac = float(cfg.get("elite_frac", 0.15))
        self.ga_mutation_rate = float(cfg.get("mutation_rate", 0.2))
        self.ga_crossover_rate = float(cfg.get("crossover_rate", 0.7))
        self.ga_tournament_k = int(cfg.get("tournament_k", 3))
        self.ga_stall_limit = int(cfg.get("stall_limit", 10))
        self.ga_seed_full_frac = float(cfg.get("seed_full_frac", 0.0))
        self.allow_unused_prbs = bool(cfg.get("allow_unused_prbs", True))

        self._rng = np.random.default_rng()

    def solve(self, pred_positions: np.ndarray, user_tiers: List[str]) -> List[int]:
        num_ues = len(user_tiers)
        M = self.num_prbs

        n_target = np.zeros((num_ues, self.horizon), dtype=float)

        for u in range(num_ues):
            target = float(self.tiers_cfg[user_tiers[u]]["target_rate_mbps"]) * 1e6
            for h in range(self.horizon):
                x, y_pos = pred_positions[u, h]
                d_m = math.hypot(float(x) - self.ru_x, float(y_pos) - self.ru_y) * 1000.0
                sinr = self.channel.compute_sinr_linear(distance_m=d_m)
                r = self.prb_bw * math.log2(1.0 + sinr)
                n_target[u, h] = min(target / r, M) if r > 0 else M

        discounts = np.array([self.gamma**h for h in range(self.horizon)], dtype=float)

        if num_ues == 0 or self.horizon == 0:
            return []

        population = self._init_population(num_ues, self.horizon, M)
        warm_allocs = warm_start_allocations(
            pred_positions,
            user_tiers,
            num_prbs=self.num_prbs,
            prb_bw=self.prb_bw,
            ru_x=self.ru_x,
            ru_y=self.ru_y,
            channel=self.channel,
            tiers_cfg=self.tiers_cfg,
        )
        if warm_allocs:
            for i, alloc in enumerate(warm_allocs[: self.ga_population]):
                for h in range(self.horizon):
                    population[i, :, h] = np.array(alloc, dtype=int)
        best_score = -float("inf")
        best_alloc = None
        stall = 0

        for _ in range(self.ga_generations):
            scores = np.array(
                [
                    self._fitness(
                        alloc,
                        n_target,
                        discounts,
                        pred_positions,
                        user_tiers,
                    )
                    for alloc in population
                ],
                dtype=float,
            )

            max_idx = int(np.argmax(scores))
            if scores[max_idx] > best_score:
                best_score = float(scores[max_idx])
                best_alloc = population[max_idx].copy()
                stall = 0
            else:
                stall += 1
                if stall >= self.ga_stall_limit:
                    break

            elite_count = max(1, int(self.ga_elite_frac * self.ga_population))
            elite_idx = np.argsort(scores)[-elite_count:]
            elites = population[elite_idx]

            next_pop = [e.copy() for e in elites]
            while len(next_pop) < self.ga_population:
                parent_a = self._tournament_select(population, scores)
                parent_b = self._tournament_select(population, scores)

                if self._rng.random() < self.ga_crossover_rate:
                    child = self._crossover(parent_a, parent_b)
                else:
                    child = parent_a.copy()

                self._mutate(child, M)
                self._repair(child, M)
                next_pop.append(child)

            population = np.array(next_pop, dtype=int)

        if best_alloc is None:
            return [0] * num_ues

        return [int(best_alloc[u, 0]) for u in range(num_ues)]

    def _init_population(self, num_ues: int, horizon: int, M: int) -> np.ndarray:
        pop = np.zeros((self.ga_population, num_ues, horizon), dtype=int)
        for i in range(self.ga_population):
            for h in range(horizon):
                if not self.allow_unused_prbs:
                    total = M
                elif self._rng.random() < self.ga_seed_full_frac:
                    total = M
                else:
                    total = int(self._rng.integers(0, M + 1))
                pop[i, :, h] = self._sample_allocation(num_ues, total)
        return pop

    def _sample_allocation(self, num_ues: int, total: int) -> np.ndarray:
        if total <= 0:
            return np.zeros(num_ues, dtype=int)

        weights = self._rng.random(num_ues)
        weights_sum = float(weights.sum())
        if weights_sum <= 0:
            return np.zeros(num_ues, dtype=int)

        alloc = np.floor(weights / weights_sum * total).astype(int)
        remainder = total - int(alloc.sum())
        if remainder > 0:
            idx = self._rng.choice(num_ues, size=remainder, replace=True)
            for i in idx:
                alloc[i] += 1
        return alloc

    def _fitness(
        self,
        alloc: np.ndarray,
        n_target: np.ndarray,
        discounts: np.ndarray,
        pred_positions: np.ndarray,
        user_tiers: List[str],
    ) -> float:
        num_ues, horizon = alloc.shape
        total = 0.0

        for h in range(horizon):
            disc = float(discounts[h])
            n_h = alloc[:, h]
            nt_h = n_target[:, h]

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

        return total

    def _tournament_select(self, population: np.ndarray, scores: np.ndarray) -> np.ndarray:
        k = min(self.ga_tournament_k, len(population))
        idx = self._rng.choice(len(population), size=k, replace=False)
        best = idx[np.argmax(scores[idx])]
        return population[best]

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = self._rng.random(a.shape) < 0.5
        child = np.where(mask, a, b)
        return child.astype(int)

    def _mutate(self, child: np.ndarray, M: int) -> None:
        _, horizon = child.shape
        for h in range(horizon):
            if self._rng.random() >= self.ga_mutation_rate:
                continue
            total = int(child[:, h].sum())
            if total == 0:
                if self.allow_unused_prbs:
                    new_total = int(self._rng.integers(0, M + 1))
                    child[:, h] = self._sample_allocation(child.shape[0], new_total)
                else:
                    child[:, h] = self._sample_allocation(child.shape[0], M)
                continue
            if self.allow_unused_prbs:
                new_total = int(self._rng.integers(0, min(M, total) + 1))
                child[:, h] = self._sample_allocation(child.shape[0], new_total)
            else:
                child[:, h] = self._sample_allocation(child.shape[0], min(M, total))

    def _repair(self, child: np.ndarray, M: int) -> None:
        num_ues, horizon = child.shape
        for h in range(horizon):
            n_h = child[:, h]
            total = int(n_h.sum())
            if total <= M:
                continue
            excess = total - M
            while excess > 0:
                candidates = np.flatnonzero(n_h > 0)
                if candidates.size == 0:
                    break
                idx = int(self._rng.choice(candidates))
                n_h[idx] -= 1
                excess -= 1
            child[:, h] = n_h
