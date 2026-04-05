from __future__ import annotations

from typing import List, Sequence

import numpy as np

from sdn_controller.environment.env_gym import SDNEnv
from sdn_controller.environment.sdn_env import ModelChoice, User


TIER_TO_SLICE_IDX = {"tier-1": 0, "tier-2": 1, "tier-3": 2}
SLICE_NAMES = ("slice_1", "slice_2", "slice_3")


def zero_action(env: SDNEnv) -> List[int]:
    per_user_dim = 2 + env.vnf_count
    total_dim = per_user_dim * int(env.num_users)
    return [0 for _ in range(total_dim)]


def select_slices(strategy: str, users: Sequence[User], rng: np.random.Generator) -> List[int]:
    if strategy == "tier":
        return [TIER_TO_SLICE_IDX.get(u.user_type, 0) for u in users]
    if strategy == "best_efficiency":
        return [0 for _ in users]
    if strategy == "random":
        return [int(rng.integers(0, 3)) for _ in users]
    return [TIER_TO_SLICE_IDX.get(u.user_type, 0) for u in users]


def model_indices_for_slice(
    env: SDNEnv,
    slice_name: str,
    strategy: str,
    rng: np.random.Generator,
) -> List[int]:
    catalog = env.slice_catalogs[slice_name]
    indices: List[int] = []
    for vnf_id in range(1, env.vnf_count + 1):
        models = sorted(catalog[vnf_id], key=lambda m: m.model_id)
        if strategy == "max_acc":
            best = max(range(len(models)), key=lambda i: models[i].accuracy_pct)
        elif strategy == "min_vcpu":
            best = min(range(len(models)), key=lambda i: models[i].vcpu_units)
        elif strategy == "mid_acc":
            ordered = sorted(range(len(models)), key=lambda i: models[i].accuracy_pct)
            best = ordered[len(ordered) // 2]
        elif strategy == "random":
            best = int(rng.integers(0, len(models)))
        else:
            best = 0
        indices.append(int(best))
    return indices


def select_models(
    env: SDNEnv,
    slice_idxs: Sequence[int],
    model_strategy: str,
    rng: np.random.Generator,
) -> List[List[int]]:
    per_user_models: List[List[int]] = []
    for slice_idx in slice_idxs:
        slice_name = SLICE_NAMES[slice_idx]
        per_user_models.append(model_indices_for_slice(env, slice_name, model_strategy, rng))
    return per_user_models


def select_prbs(
    env: SDNEnv,
    users: Sequence[User],
    slice_idxs: Sequence[int],
    model_indices: Sequence[Sequence[int]],
    prb_strategy: str,
    rng: np.random.Generator,
) -> List[int]:
    if prb_strategy == "random":
        return [int(rng.integers(0, env.max_prbs_alloc + 1)) for _ in users]

    if prb_strategy == "equal_share":
        per_user = int(env.prbs_available) // max(1, len(users))
        prbs = [per_user for _ in users]
        remainder = int(env.prbs_available) - sum(prbs)
        for i in range(remainder):
            prbs[i % len(prbs)] += 1
        return prbs

    if prb_strategy == "target_rate":
        prbs = []
        for u, slice_idx, models_idx in zip(users, slice_idxs, model_indices):
            slice_name = SLICE_NAMES[slice_idx]
            models = materialize_models(env, slice_name, list(models_idx))
            required = env._required_prbs_for_target(
                user=u, slice_name=slice_name, models=models
            )
            prbs.append(int(required))
        return env._enforce_prb_budget(prbs)

    return [0 for _ in users]


def materialize_models(
    env: SDNEnv, slice_name: str, model_indices: List[int]
) -> List[ModelChoice]:
    return env._selected_models(slice_name, model_indices)


def pack_action(
    slice_idxs: Sequence[int],
    prbs: Sequence[int],
    model_indices: Sequence[Sequence[int]],
) -> List[int]:
    action: List[int] = []
    for s_idx, p, models in zip(slice_idxs, prbs, model_indices):
        action.append(int(s_idx))
        action.append(int(p))
        action.extend(int(m) for m in models)
    return action
