from __future__ import annotations

import sys
from pathlib import Path as _Path

ROOT_DIR = _Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
from typing import Dict, List

from sdn_controller.utilities.utils import load_config, resolve_path
from sdn_controller.utilities.eval_set import (
    build_eval_user_sets,
    evaluate_on_user_sets,
)
from sdn_controller.environment.env_gym import SDNEnv
from sdn_controller.agents.ppo_agent import PPOAgent
from sdn_controller.agents.a2c_agent import A2CAgent
from sdn_controller.agents.equal_share_max_acc_agent import EqualShareMaxAccAgent
from sdn_controller.agents.greedy_target_rate_agent import GreedyTargetRateAgent
from sdn_controller.agents.min_cost_target_rate_agent import MinCostTargetRateAgent


def _load_model_path(train_cfg: dict, paths_cfg: dict, algo: str) -> str:
    model_paths = paths_cfg.get("model_paths", {})
    default_path = f"sdn_controller/models/{algo.lower()}_sdn_agent.zip"
    return resolve_path(model_paths.get(algo.lower(), default_path))


def _build_agent(algo: str, env: SDNEnv, train_cfg: dict, paths_cfg: dict):
    algo_u = algo.upper()
    if algo_u == "PPO":
        path = _load_model_path(train_cfg, paths_cfg, "ppo")
        if not os.path.exists(path):
            print(f"Missing PPO model at {path}")
            return None
        return PPOAgent.load(path)
    if algo_u == "A2C":
        path = _load_model_path(train_cfg, paths_cfg, "a2c")
        if not os.path.exists(path):
            print(f"Missing A2C model at {path}")
            return None
        return A2CAgent.load(path)
    algo_l = algo.lower()
    if algo_l == "equal_share_max_acc":
        return EqualShareMaxAccAgent(env=env, seed=int(train_cfg.get("seed", 0)))
    if algo_l == "greedy_target_rate":
        return GreedyTargetRateAgent(env=env, seed=int(train_cfg.get("seed", 0)))
    if algo_l == "min_cost_target_rate":
        return MinCostTargetRateAgent(env=env, seed=int(train_cfg.get("seed", 0)))
    print(f"Unknown algo in evaluation: {algo}")
    return None


def main() -> None:
    cfg = load_config()
    train_cfg = cfg["training"]
    eval_cfg = cfg.get("evaluation", {})
    paths_cfg = cfg.get("paths", {})

    eval_seed = int(eval_cfg.get("seed", int(train_cfg.get("seed", 0)) + 999))
    eval_set_size = int(eval_cfg.get("eval_set_size", 1000))
    user_counts = list(eval_cfg.get("user_counts", [int(cfg["environment"]["num_users"])]))
    trained_users = int(cfg["environment"]["num_users"])
    trained_users = int(cfg["environment"]["num_users"])
    algos = list(
        eval_cfg.get(
            "algos",
            ["PPO", "A2C", "equal_share_max_acc", "greedy_target_rate", "min_cost_target_rate"],
        )
    )
    output_path = resolve_path(
        paths_cfg.get("eval_output_path", "sdn_controller/logs/evaluation/eval_results.log")
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rows: List[Dict[str, float]] = []
    for user_count in user_counts:
        env = SDNEnv(seed=eval_seed, num_users=int(user_count))
        eval_user_sets = build_eval_user_sets(
            user_types=list(env.user_types),
            num_sets=eval_set_size,
            snir_min=env.snir_min,
            snir_max=env.snir_max,
            seed=eval_seed,
        )
        eval_env = SDNEnv(seed=eval_seed, num_users=int(user_count))

        print(f"Evaluating with num_users={user_count} over {len(eval_user_sets)} sets")
        for algo in algos:
            if algo.upper() == "PPO" and int(user_count) != trained_users:
                print(
                    f"- Skipping PPO for num_users={user_count} "
                    f"(trained with num_users={trained_users})"
                )
                continue
            agent = _build_agent(algo, eval_env, train_cfg, paths_cfg)
            if agent is None:
                continue
            metrics = evaluate_on_user_sets(
                agent,
                eval_env,
                eval_user_sets,
                num_users=int(user_count),
                measure_latency=True,
            )
            print(
                f"- {algo}: mean_reward={metrics['mean_reward']:.4f} "
                f"mean_profit={metrics['mean_profit']:.4f} "
                f"mean_sat={metrics['mean_satisfaction']:.4f} "
                f"avg_latency={metrics['mean_latency']:.6f}s"
            )
            rows.append(
                {
                    "algo": algo,
                    "num_users": int(user_count),
                    "mean_reward": metrics["mean_reward"],
                    "std_reward": metrics["std_reward"],
                    "mean_reward_per_user": metrics["mean_reward_per_user"],
                    "mean_profit": metrics["mean_profit"],
                    "std_profit": metrics["std_profit"],
                    "mean_profit_per_user": metrics["mean_profit_per_user"],
                    "mean_satisfaction": metrics["mean_satisfaction"],
                    "std_satisfaction": metrics["std_satisfaction"],
                    "mean_satisfaction_per_user": metrics["mean_satisfaction_per_user"],
                    "mean_latency": metrics["mean_latency"],
                }
            )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation results\n")
        for row in rows:
            f.write(f"Algo: {row['algo']} | Num_users: {row['num_users']}\n")
            f.write(
                "mean_reward = "
                f"{row['mean_reward']:.6f} "
                "std_reward= "
                f"{row['std_reward']:.6f} "
                "mean_reward_per_user= "
                f"{row['mean_reward_per_user']:.6f}\n"
            )
            f.write(
                "mean_profit = "
                f"{row['mean_profit']:.6f} "
                "std_profit = "
                f"{row['std_profit']:.6f} "
                "profit_per_user= "
                f"{row['mean_profit_per_user']:.6f}\n"
            )
            f.write(
                "mean_sat = "
                f"{row['mean_satisfaction']:.6f} "
                "std_sat = "
                f"{row['std_satisfaction']:.6f} "
                "mean_sat_per_user = "
                f"{row['mean_satisfaction_per_user']:.6f}\n"
            )
            f.write(f"mean_latency = {row['mean_latency']:.6f}\n\n")

    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
