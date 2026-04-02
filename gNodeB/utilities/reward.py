# Reward function
#
# Math (single timestep, U users, tiers T):
#   d_u           = distance(ru, pos_u)
#   sinr_u        = SINR(d_u)
#   data_rate_u   = n_u * prb_bw * log2(1 + sinr_u)
#   target_u      = target_rate_mbps[tier_u] * 1e6
#   satisfaction_u = min(1, data_rate_u / target_u)
#   profit_u      = fee[tier_u] - n_u * prb_cost
#   reward_base   = sum_u (
#                     profit_sensitivity * profit_u
#                     + satisfaction_sensitivity * weight[tier_u] * satisfaction_u
#                   )
#   required_t    = min_percent[t] * count[t]
#   deficit_t     = max(0, required_t - sum_satisfaction_t)
#   penalty       = lambda * sum_t deficit_t
#   reward        = reward_base + penalty

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from channel.channel_model import Channel


def compute_reward_components(
    positions_t: np.ndarray,
    alloc_counts: List[int] | np.ndarray,
    user_tiers: List[str],
    tiers_cfg: Dict[str, dict],
    ru_x: float,
    ru_y: float,
    prb_bw: float,
    prb_cost: float,
    penalty_lambda: float,
    profit_sensitivity: float,
    satisfaction_sensitivity: float,
    channel: Channel,
) -> Tuple[
    float,
    float,
    float,
    float,
    Dict[str, int],
    Dict[str, int],
    Dict[str, float],
]:
    """Return (reward, reward_base, penalty, profit_sum, tier_stats, tier_sat_count, tier_sat_sum)."""
    num_ues = len(user_tiers)

    reward_base = 0.0
    profit_sum = 0.0
    tier_stats = {tier: 0 for tier in tiers_cfg}
    tier_sat_count = {tier: 0 for tier in tiers_cfg}
    tier_sat_sum = {tier: 0.0 for tier in tiers_cfg}

    for u in range(num_ues):
        tier = user_tiers[u]
        tier_stats[tier] += 1

        x, y = positions_t[u, 0], positions_t[u, 1]
        d_km = math.hypot(float(x) - ru_x, float(y) - ru_y)
        d_m = d_km * 1000.0
        sinr = channel.compute_sinr_linear(distance_m=d_m)
        data_rate = float(alloc_counts[u]) * prb_bw * math.log2(1.0 + sinr)

        target = float(tiers_cfg[tier]["target_rate_mbps"]) * 1e6
        satisfaction = min(1.0, data_rate / target)
        if data_rate >= target:
            tier_sat_count[tier] += 1
        tier_sat_sum[tier] += satisfaction

        subscription_fee = float(tiers_cfg[tier]["subscription_fee"])
        profit = subscription_fee - (float(alloc_counts[u]) * prb_cost)
        profit_sum += profit
        weight = float(tiers_cfg[tier]["weight"])

        reward_base += (
            profit_sensitivity * profit
            + satisfaction_sensitivity * weight * satisfaction
        )

    penalty = 0.0
    for tier, count in tier_stats.items():
        required = float(tiers_cfg[tier]["min_percent_of_user_to_satisfy"]) * count
        deficit = max(0.0, required - tier_sat_sum[tier])
        penalty += penalty_lambda * deficit

    reward = reward_base + penalty
    return (
        reward,
        reward_base,
        penalty,
        profit_sum,
        tier_stats,
        tier_sat_count,
        tier_sat_sum,
    )


def compute_reward_scalar(
    positions_t: np.ndarray,
    alloc_counts: List[int] | np.ndarray,
    user_tiers: List[str],
    tiers_cfg: Dict[str, dict],
    ru_x: float,
    ru_y: float,
    prb_bw: float,
    prb_cost: float,
    penalty_lambda: float,
    profit_sensitivity: float,
    satisfaction_sensitivity: float,
    channel: Channel,
) -> float:
    """Compute the scalar reward for a single time step."""
    reward, _, _, _, _, _, _ = compute_reward_components(
        positions_t,
        alloc_counts,
        user_tiers,
        tiers_cfg,
        ru_x,
        ru_y,
        prb_bw,
        prb_cost,
        penalty_lambda,
        profit_sensitivity,
        satisfaction_sensitivity,
        channel,
    )
    return reward


def compute_reward(
    positions_t: np.ndarray,
    alloc_counts: List[int],
    user_tiers: List[str],
    tiers_cfg: Dict[str, dict],
    ru_x: float,
    ru_y: float,
    prb_bw: float,
    prb_cost: float,
    penalty_lambda: float,
    profit_sensitivity: float,
    satisfaction_sensitivity: float,
    channel: Channel,
) -> Dict[str, float | Dict[str, int]]:
    """Compute reward and components for a single time step.

    positions_t: shape (U, 2) [x_km, y_km]
    alloc_counts[u]: number of PRBs allocated to user u
    """
    reward, reward_base, penalty, profit_sum, tier_stats, tier_sat_count, tier_sat_sum = (
        compute_reward_components(
            positions_t,
            alloc_counts,
            user_tiers,
            tiers_cfg,
            ru_x,
            ru_y,
            prb_bw,
            prb_cost,
            penalty_lambda,
            profit_sensitivity,
            satisfaction_sensitivity,
            channel,
        )
    )

    return {
        "reward": reward,
        "reward_base": reward_base,
        "penalty": penalty,
        "profit": profit_sum,
        "tier_stats": tier_stats,
        "tier_sat": tier_sat_count,
        "tier_sat_sum": tier_sat_sum,
        "total_prbs": sum(alloc_counts),
    }
