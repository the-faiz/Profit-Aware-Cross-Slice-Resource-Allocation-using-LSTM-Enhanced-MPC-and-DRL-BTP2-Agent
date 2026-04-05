from __future__ import annotations

"""
Reward calculation (math summary)

Given:
  - Users u in tiers t with target data-rate R_t, subscription_fee F_t, weight W_t
  - Slice efficiency e_s, PRB bandwidth B, PRB cost C_prb
  - Selected VNF models with weighted average accuracy A in [0, 1]
  - Total vCPU cost C_vcpu (sum of model vCPU units * vCPU unit cost)
  - Sensitivities: profit_sensitivity S_p, satisfaction_sensitivity S_s

Allocated data-rate for user u:
  r_u = P * e_s * B * log(1 + SNIR_u) * A

User satisfaction:
  sat_u = min(1, r_u / R_t)

Reward:
  profit_u = F_t - P*C_prb - C_vcpu
  reward = sum_u (S_p * profit_u + S_s * W_t * sat_u)

Optional minimum-satisfaction penalty:
  If enabled, require at least a fraction M of users to meet their target rate.
  If a tier's satisfied_rate < M_tier, apply a penalty proportional to the shortfall:
    penalty = sum_tier (M_tier - satisfied_rate_tier) * lambda
    reward = reward - penalty
"""

import math
from typing import Dict, Iterable, List, Tuple, TYPE_CHECKING

from sdn_controller.utilities.utils import load_config

if TYPE_CHECKING:
    from sdn_controller.environment.sdn_env import ModelChoice, User


def weighted_model_accuracy(models: Iterable[ModelChoice]) -> float:
    weights = [m.vnf_weight for m in models]
    accs = [m.accuracy_pct for m in models]
    if not accs:
        return 0.0
    total_w = sum(weights) if weights else 0.0
    if total_w <= 0.0:
        return sum(accs) / len(accs) / 100.0
    weighted = sum(a * w for a, w in zip(accs, weights))
    return (weighted / total_w) / 100.0


def allocated_data_rate_mbps(
    prbs_allocated: int,
    slice_name: str,
    snir: float,
    model_accuracy: float,
) -> float:
    cfg = load_config()
    eff = cfg["slices"]["efficiencies"][slice_name]
    spectral = math.log2(1.0 + snir)
    bps = (
        prbs_allocated
        * eff
        * float(cfg["prb"]["bandwidth_hz"])
        * spectral
        * model_accuracy
    )
    return bps / 1_000_000.0


def compute_reward(
    users: List[User],
    slice_names: List[str],
    prbs_allocations: List[int],
    selected_models_per_user: List[List[ModelChoice]],
) -> Tuple[float, Dict[str, float]]:
    cfg = load_config()
    slice_eff = cfg["slices"]["efficiencies"]
    tiers_cfg = cfg["tiers"]

    if not (
        len(users)
        == len(slice_names)
        == len(prbs_allocations)
        == len(selected_models_per_user)
    ):
        raise ValueError("Users, slices, prbs, and model selections must have same length.")

    reward_without = 0.0
    total_satisfaction = 0.0
    total_profit = 0.0
    satisfied_total = 0
    satisfied_counts = {tier: 0 for tier in tiers_cfg}
    total_counts = {tier: 0 for tier in tiers_cfg}

    total_vcpu = 0.0
    total_prbs = 0

    for idx, user in enumerate(users):
        tier = user.user_type
        tier_cfg = tiers_cfg[tier]
        total_counts[tier] += 1
        slice_name = slice_names[idx]
        prbs = int(prbs_allocations[idx])
        models = selected_models_per_user[idx]
        model_acc = weighted_model_accuracy(models)
        vcpu_units = sum(m.vcpu_units for m in models)

        data_rate = allocated_data_rate_mbps(
            prbs_allocated=prbs,
            slice_name=slice_name,
            snir=user.snir,
            model_accuracy=model_acc,
        )
        satisfied = data_rate >= float(tier_cfg["target_rate_mbps"])
        if satisfied:
            satisfied_counts[tier] += 1
            satisfied_total += 1

        satisfaction = min(1.0, data_rate / float(tier_cfg["target_rate_mbps"]))
        prb_cost = prbs * float(cfg["prb"]["cost"])
        vcpu_cost = vcpu_units * float(cfg["vcpu"]["cost"])
        profit = float(tier_cfg["subscription_fee"]) - prb_cost - vcpu_cost
        profit_sens = float(cfg["reward"].get("profit_sensitivity", 1.0))
        sat_sens = float(cfg["reward"].get("satisfaction_sensitivity", 1.0))
        reward_without += (
            profit_sens * profit + sat_sens * float(tier_cfg["weight"]) * satisfaction
        )
        total_satisfaction += satisfaction
        total_profit += profit

        total_vcpu += vcpu_units
        total_prbs += prbs

    penalty = 0.0
    reward_cfg = cfg.get("reward", {})
    use_min_penalty = bool(reward_cfg.get("enable_min_satisfaction_penalty", False))
    if use_min_penalty:
        min_cfg = reward_cfg.get("min_percent_of_users_to_satisfy", 0.8)
        if isinstance(min_cfg, dict):
            min_by_tier = {
                tier: max(0.0, min(1.0, float(val)))
                for tier, val in min_cfg.items()
            }
        else:
            min_frac = max(0.0, min(1.0, float(min_cfg)))
            min_by_tier = {tier: min_frac for tier in tiers_cfg}

        penalty_lambda = float(reward_cfg.get("min_satisfaction_penalty_lambda", 0.0))
        for tier in tiers_cfg:
            total = total_counts.get(tier, 0)
            satisfied = satisfied_counts.get(tier, 0)
            rate = (satisfied / total) if total else 0.0
            shortfall = max(0.0, min_by_tier.get(tier, 0.0) - rate)
            penalty += shortfall * penalty_lambda

    reward = reward_without - penalty
    avg_satisfaction = (total_satisfaction / len(users)) if users else 0.0
    avg_profit = total_profit
    avg_profit_per_user = (total_profit / len(users)) if users else 0.0
    satisfied_rate = (satisfied_total / len(users)) if users else 0.0
    info = {
        "reward_without": reward_without,
        "min_violations": penalty,
        "reward": reward,
        "total_vcpu": total_vcpu,
        "total_prbs": total_prbs,
        "avg_satisfaction": avg_satisfaction,
        "avg_profit": avg_profit,
        "avg_profit_per_user": avg_profit_per_user,
        "satisfied_rate": satisfied_rate,
    }
    return reward, info
