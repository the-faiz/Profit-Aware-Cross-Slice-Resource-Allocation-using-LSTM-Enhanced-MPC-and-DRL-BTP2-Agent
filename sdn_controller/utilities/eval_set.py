from __future__ import annotations

from typing import Dict, List
import time

import numpy as np

from sdn_controller.environment.sdn_env import User
from sdn_controller.environment.env_gym import SDNEnv


def build_eval_user_sets(
    user_types: List[str],
    num_sets: int,
    snir_min: float,
    snir_max: float,
    seed: int,
) -> List[List[User]]:
    rng = np.random.default_rng(int(seed))
    num_users = len(user_types)
    user_sets: List[List[User]] = []
    for _ in range(int(num_sets)):
        users: List[User] = []
        for u_id in range(num_users):
            users.append(
                User(
                    user_id=u_id,
                    user_type=user_types[u_id],
                    snir=float(rng.uniform(snir_min, snir_max)),
                )
            )
        user_sets.append(users)
    return user_sets


def _get_base_env(env: SDNEnv):
    base = env
    if hasattr(base, "unwrapped"):
        base = base.unwrapped
    if not hasattr(base, "_obs") and hasattr(base, "env"):
        base = base.env
    return base


def evaluate_on_user_sets(
    agent,
    env: SDNEnv,
    user_sets: List[List[User]],
    num_users: int,
    measure_latency: bool = False,
) -> Dict[str, float]:
    rewards: List[float] = []
    rewards_pp: List[float] = []
    profits: List[float] = []
    profits_pp: List[float] = []
    sats: List[float] = []
    sats_pp: List[float] = []
    min_violations: List[float] = []
    latencies: List[float] = []
    base_env = _get_base_env(env)
    for users in user_sets:
        base_env.current_users = users
        obs = base_env._obs()
        # If the agent was trained with a larger fixed observation size (e.g., more users),
        # pad the observation so PPO/A2C models can still run on fewer users.
        pad_to = getattr(agent, "obs_dim", None)
        if pad_to is not None:
            obs_len = int(obs.shape[0])
            pad_to = int(pad_to)
            if obs_len < pad_to:
                obs = np.pad(obs, (0, pad_to - obs_len), mode="constant")
            elif obs_len > pad_to:
                raise ValueError(
                    f"Observation length {obs_len} exceeds agent obs_dim {pad_to}. "
                    "The model was trained for fewer users; retrain or reduce num_users."
                )
        t0 = time.perf_counter() if measure_latency else 0.0
        action = agent.predict(obs, deterministic=True)
        _, reward, _, _, info = base_env.step(action)
        if measure_latency:
            latencies.append(time.perf_counter() - t0)
        rewards.append(float(reward))
        rewards_pp.append(float(reward) / num_users if num_users else 0.0)
        if isinstance(info, dict):
            profits.append(float(info.get("avg_profit", 0.0)))
            profits_pp.append(float(info.get("avg_profit_per_user", 0.0)))
            sat = float(info.get("avg_satisfaction", 0.0))
            sats.append(sat)
            sats_pp.append(sat)
            min_violations.append(float(info.get("min_violations", 0.0)))

    if not rewards:
        return {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "mean_reward_per_user": 0.0,
            "mean_profit": 0.0,
            "std_profit": 0.0,
            "mean_profit_per_user": 0.0,
            "mean_satisfaction": 0.0,
            "std_satisfaction": 0.0,
            "mean_satisfaction_per_user": 0.0,
            "mean_min_violations": 0.0,
            "std_min_violations": 0.0,
            "mean_latency": 0.0,
        }

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_reward_per_user": float(np.mean(rewards_pp)),
        "mean_profit": float(np.mean(profits)) if profits else 0.0,
        "std_profit": float(np.std(profits)) if profits else 0.0,
        "mean_profit_per_user": float(np.mean(profits_pp)) if profits_pp else 0.0,
        "mean_satisfaction": float(np.mean(sats)) if sats else 0.0,
        "std_satisfaction": float(np.std(sats)) if sats else 0.0,
        "mean_satisfaction_per_user": float(np.mean(sats_pp)) if sats_pp else 0.0,
        "mean_min_violations": float(np.mean(min_violations)) if min_violations else 0.0,
        "std_min_violations": float(np.std(min_violations)) if min_violations else 0.0,
        "mean_latency": float(np.mean(latencies)) if latencies else 0.0,
    }
