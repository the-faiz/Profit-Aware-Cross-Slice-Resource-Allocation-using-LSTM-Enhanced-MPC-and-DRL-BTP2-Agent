from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from sdn_controller.utilities.imports import require_gymnasium

gym, spaces = require_gymnasium()

from sdn_controller.utilities.utils import load_config, load_slice_catalogs
from sdn_controller.environment.sdn_env import (
    ModelChoice,
    User,
    compute_reward,
)
from sdn_controller.utilities.reward import weighted_model_accuracy


class SDNEnv(gym.Env):
    """
    Action (per user, repeated num_users times):
      - slice index (0..2)
      - prbs_allocated (0..prbs_available)
      - model choice per VNF (0..models_per_vnf-1) for vnf_count VNFs
    Observation:
      [prbs_available, users...]
      per user: [tier_index, snir] (padded to max_users)
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None, num_users: int | None = None) -> None:
        super().__init__()
        self.cfg = load_config()
        self.rng = random.Random(seed or int(self.cfg["training"]["seed"]))

        env_cfg = self.cfg["environment"]
        gen_cfg = self.cfg["generators"]

        self.num_users = int(num_users if num_users is not None else env_cfg["num_users"])
        self.prbs_available = int(env_cfg["prbs_available"])
        self.max_prbs_alloc = self.prbs_available
        self.snir_min = float(env_cfg.get("snir_min", 0.1))
        self.snir_max = float(env_cfg.get("snir_max", 20.0))

        self.vnf_count = int(gen_cfg["vnf_catalog"]["vnf_count"])
        self.models_per_vnf = int(gen_cfg["vnf_catalog"]["models_per_vnf"])

        self.slice_catalogs = load_slice_catalogs(self.cfg)
        self.tier_probs = self._tier_probabilities()

        # Sample user types once; resample only SNIR each episode.
        self.user_types = [self._sample_tier() for _ in range(self.num_users)]

        # Action: per-user blocks [slice, prbs, model_1, ..., model_10]
        per_user_dims = [3, self.max_prbs_alloc + 1] + [self.models_per_vnf] * self.vnf_count
        self.action_space = spaces.MultiDiscrete(per_user_dims * self.num_users)

        # Observation vector length
        obs_len = 1 + (self.num_users * 2)
        self.observation_space = spaces.Box(
            low=0.0, high=np.finfo(np.float32).max, shape=(obs_len,), dtype=np.float32
        )

        self.current_users: List[User] = []

    def _tier_probabilities(self) -> List[tuple[str, float]]:
        tiers_cfg = self.cfg["tiers"]
        probs = []
        for tier_name, tier_cfg in tiers_cfg.items():
            probs.append((tier_name, float(tier_cfg.get("probability", 0.0))))
        total = sum(p for _, p in probs)
        if total <= 0:
            raise ValueError("Tier probabilities must sum to > 0")
        return [(t, p / total) for t, p in probs]

    def _sample_tier(self) -> str:
        r = self.rng.random()
        acc = 0.0
        for tier, p in self.tier_probs:
            acc += p
            if r <= acc:
                return tier
        return self.tier_probs[-1][0]

    def _sample_users(self) -> List[User]:
        users: List[User] = []
        for u_id in range(self.num_users):
            users.append(
                User(
                    user_id=u_id,
                    user_type=self.user_types[u_id],
                    snir=self.rng.uniform(self.snir_min, self.snir_max),
                )
            )
        return users


    def _obs(self) -> np.ndarray:
        obs = [float(self.prbs_available)]
        for user in self.current_users:
            tier_idx = {"tier-1": 0.0, "tier-2": 1.0, "tier-3": 2.0}.get(
                user.user_type, 0.0
            )
            obs.extend([tier_idx, float(user.snir)])
        # Pad
        while len(obs) < 1 + self.num_users * 2:
            obs.extend([0.0, 0.0])
        return np.array(obs, dtype=np.float32)

    def reset(self, seed: int | None = None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.current_users = self._sample_users()
        return self._obs(), {}

    def _selected_models(
        self, slice_name: str, model_indices: List[int]
    ) -> List[ModelChoice]:
        catalog = self.slice_catalogs[slice_name]
        selected: List[ModelChoice] = []
        for vnf_id in range(1, self.vnf_count + 1):
            models = sorted(catalog[vnf_id], key=lambda m: m.model_id)
            idx = model_indices[vnf_id - 1]
            idx = max(0, min(idx, len(models) - 1))
            selected.append(models[idx])
        return selected

    def _enforce_prb_budget(self, prbs: List[int]) -> List[int]:
        total = sum(prbs)
        if total <= self.prbs_available:
            return prbs
        if total == 0:
            return prbs
        factor = self.prbs_available / float(total)
        scaled = [int(p * factor) for p in prbs]
        remainder = self.prbs_available - sum(scaled)
        if remainder > 0:
            frac = [p * factor - int(p * factor) for p in prbs]
            order = sorted(range(len(prbs)), key=lambda i: frac[i], reverse=True)
            for i in order[:remainder]:
                scaled[i] += 1
        return scaled

    def _required_prbs_for_target(
        self,
        user: User,
        slice_name: str,
        models: List[ModelChoice],
    ) -> int:
        tiers_cfg = self.cfg["tiers"]
        eff = float(self.cfg["slices"]["efficiencies"][slice_name])
        bandwidth = float(self.cfg["prb"]["bandwidth_hz"])
        target_rate_mbps = float(tiers_cfg[user.user_type]["target_rate_mbps"])

        spectral = float(np.log2(1.0 + float(user.snir)))
        model_acc = float(weighted_model_accuracy(models))
        denom = eff * bandwidth * spectral * model_acc
        if denom <= 0.0:
            return 0
        required = (target_rate_mbps * 1_000_000.0) / denom
        return max(0, min(self.max_prbs_alloc, int(np.ceil(required))))


    def step(self, action):
        action = [int(x) for x in action]
        per_user_dim = 2 + self.vnf_count
        slice_names: List[str] = []
        prbs_allocations: List[int] = []
        selected_models_per_user: List[List[ModelChoice]] = []

        for u in range(self.num_users):
            start = u * per_user_dim
            slice_idx = int(action[start])
            prbs = int(action[start + 1])
            model_indices = action[start + 2 : start + 2 + self.vnf_count]
            slice_name = ("slice_1", "slice_2", "slice_3")[slice_idx]
            slice_names.append(slice_name)
            prbs_allocations.append(prbs)
            selected_models_per_user.append(
                self._selected_models(slice_name, model_indices)
            )

        # Clamp PRBs to the minimum needed to hit the target data rate (per user)
        for i, user in enumerate(self.current_users):
            required = self._required_prbs_for_target(
                user=user,
                slice_name=slice_names[i],
                models=selected_models_per_user[i],
            )
            prbs_allocations[i] = min(prbs_allocations[i], required)

        prbs_allocations = self._enforce_prb_budget(prbs_allocations)

        reward, info = compute_reward(
            users=self.current_users,
            slice_names=slice_names,
            prbs_allocations=prbs_allocations,
            selected_models_per_user=selected_models_per_user,
        )
        obs = self._obs()
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, info
