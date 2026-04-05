from __future__ import annotations

import sys
from pathlib import Path as _Path

ROOT_DIR = _Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import argparse
import time
import numpy as np

from sdn_controller.utilities.utils import load_config, resolve_path
from sdn_controller.agents.ppo_agent import PPOAgent, PPOConfig
from sdn_controller.agents.a2c_agent import A2CAgent, A2CConfig

from sdn_controller.environment.env_gym import SDNEnv
from sdn_controller.utilities.eval_set import (
    build_eval_user_sets,
    evaluate_on_user_sets,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SDN controller agents.")
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Agent to train (PPO or A2C). Overrides config training.algo.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("You have started training the DRL agent. This may take some time. Go and grab a coffee! ☕")

    print("Loading configuration...")
    cfg = load_config()
    train_cfg = cfg["training"]
    log_cfg = cfg.get("logging", {})
    paths_cfg = cfg.get("paths", {})
    reward_cfg = cfg.get("reward", {})
    min_penalty_enabled = bool(reward_cfg.get("enable_min_satisfaction_penalty", False))
    algo = str(args.algo or train_cfg.get("algo", "PPO")).upper()
    if min_penalty_enabled:
        print("Minimum satisfaction penalty is ENABLED.")
    print("Configuration loaded successfully.")

    print("Initializing environment and model...")
    env = SDNEnv(seed=int(train_cfg["seed"]))
    num_users = int(env.num_users)
    prbs_available = int(env.prbs_available)
    policy_cfg = train_cfg.get("policy", {})
    use_custom = bool(policy_cfg.get("use_custom", False))
    net_arch = policy_cfg.get("net_arch", [128, 128])
    activation = str(policy_cfg.get("activation", "relu")).lower()
    print(
        f"Environment initialized with {num_users} users per episode "
        f"and {prbs_available} PRBs available."
    )

    log_f = None
    if bool(log_cfg.get("enabled", True)):
        log_dir = str(paths_cfg.get("log_dir_training", "sdn_controller/logs/training"))
        log_path = resolve_path(f"{log_dir}/simulation_{algo.lower()}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        mode = "a" if bool(log_cfg.get("append", False)) else "w"
        log_f = open(log_path, mode, encoding="utf-8")

    header_line = "Training DRL Agent"
    sep_line = "-" * 80
    if log_f is not None:
        algo = str(train_cfg.get("algo", "PPO")).upper()
        log_f.write(header_line + "\n")
        log_f.write(sep_line + "\n")
        log_f.write(
            f"Algo: {algo} | Timesteps: {int(train_cfg['timesteps'])} | "
            f"Seed: {int(train_cfg['seed'])}\n"
        )
        log_f.write(f"Users per episode: {num_users}\n")
        log_f.write(f"PRBs available: {prbs_available}\n")


    t0 = time.time()
    n_steps = int(train_cfg.get("n_steps", 2048))
    if not use_custom:
        net_arch = [64, 64]

    if algo == "A2C":
        a2c_cfg = A2CConfig(
            learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
            gamma=float(train_cfg.get("gamma", 0.99)),
            ent_coef=float(train_cfg.get("ent_coef", 0.0)),
            vf_coef=float(train_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        )
        model = A2CAgent(
            obs_dim=env.observation_space.shape[0],
            action_nvec=list(env.action_space.nvec),
            net_arch=list(net_arch),
            activation=activation,
            config=a2c_cfg,
            seed=int(train_cfg["seed"]),
        )
    else:
        ppo_cfg = PPOConfig(
            learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
            gamma=float(train_cfg.get("gamma", 0.99)),
            gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
            clip_range=float(train_cfg.get("clip_range", 0.2)),
            ent_coef=float(train_cfg.get("ent_coef", 0.0)),
            vf_coef=float(train_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
            n_epochs=int(train_cfg.get("n_epochs", 10)),
            batch_size=int(train_cfg.get("batch_size", 64)),
        )
        model = PPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_nvec=list(env.action_space.nvec),
            net_arch=list(net_arch),
            activation=activation,
            config=ppo_cfg,
            seed=int(train_cfg["seed"]),
        )

    timesteps = int(train_cfg["timesteps"])
    checkpoint_every = int(train_cfg.get("checkpoint_every", 1000))
    eval_cfg = cfg.get("evaluation", {})
    eval_set_size = int(eval_cfg.get("eval_set_size", 1000))
    eval_seed = int(eval_cfg.get("seed", int(train_cfg["seed"]) + 123))

    print("Building evaluation set...")
    eval_user_sets = build_eval_user_sets(
        user_types=list(env.user_types),
        num_sets=eval_set_size,
        snir_min=env.snir_min,
        snir_max=env.snir_max,
        seed=eval_seed,
    )
    eval_metrics_env = SDNEnv(seed=int(train_cfg["seed"]))
    print(f"Evaluation set size: {len(eval_user_sets)}")
    if algo not in ("PPO", "A2C"):
        print("Unknown algo. Train supports only PPO or A2C.")
        return
        print("Learning Started...")
    obs, _ = env.reset()
    global_step = 0
    last_reward = 0.0
    last_profit = 0.0
    last_sat = 0.0
    last_profit_pp = 0.0
    last_reward_pp = 0.0

    print("Traiining Started...")
    while global_step < timesteps:
        rollout_obs = []
        rollout_actions = []
        rollout_logprobs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []
        last_done = False

        for _ in range(n_steps):
            action, logprob, value = model.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            global_step += 1
            last_reward = float(reward)
            last_profit = float(info.get("avg_profit", 0.0)) if isinstance(info, dict) else 0.0
            last_sat = float(info.get("avg_satisfaction", 0.0)) if isinstance(info, dict) else 0.0
            last_profit_pp = float(info.get("avg_profit_per_user", 0.0)) if isinstance(info, dict) else 0.0
            last_reward_pp = last_reward / num_users if num_users else 0.0

            if global_step % checkpoint_every == 0:
                print(f"Checkpoint {global_step}:")
                print("Running Evaluations")
                metrics = evaluate_on_user_sets(
                    model,
                    eval_metrics_env,
                    eval_user_sets,
                    num_users=num_users,
                )
                checkpoint_line = (
                    f"Checkpoint {global_step}: "
                )
                print(checkpoint_line)
                print(
                    "mean_reward = "
                    f"{metrics['mean_reward']:.4f} "
                    "std_reward= "
                    f"{metrics['std_reward']:.4f} "
                    "mean_reward_per_user= "
                    f"{metrics['mean_reward_per_user']:.4f}"
                )
                print(
                    "mean_profit = "
                    f"{metrics['mean_profit']:.4f} "
                    "std_profit = "
                    f"{metrics['std_profit']:.4f} "
                    "profit_per_user= "
                    f"{metrics['mean_profit_per_user']:.4f}"
                )
                print(
                    "mean_sat = "
                    f"{metrics['mean_satisfaction']:.4f} "
                    "std_sat = "
                    f"{metrics['std_satisfaction']:.4f} "
                    "mean_sat_per_user = "
                    f"{metrics['mean_satisfaction_per_user']:.4f}"
                )
                if min_penalty_enabled:
                    print(
                        "mean_min_violations = "
                        f"{metrics['mean_min_violations']:.4f} "
                        "std_min_violations = "
                        f"{metrics['std_min_violations']:.4f}"
                    )
                if log_f is not None:
                    log_f.write(checkpoint_line + "\n")
                    log_f.write(
                        "mean_reward = "
                        f"{metrics['mean_reward']:.4f} "
                        "std_reward= "
                        f"{metrics['std_reward']:.4f} "
                        "mean_reward_per_user= "
                        f"{metrics['mean_reward_per_user']:.4f}\n"
                    )
                    log_f.write(
                        "mean_profit = "
                        f"{metrics['mean_profit']:.4f} "
                        "std_profit = "
                        f"{metrics['std_profit']:.4f} "
                        "profit_per_user= "
                        f"{metrics['mean_profit_per_user']:.4f}\n"
                    )
                    log_f.write(
                        "mean_sat = "
                        f"{metrics['mean_satisfaction']:.4f} "
                        "std_sat = "
                        f"{metrics['std_satisfaction']:.4f} "
                        "mean_sat_per_user = "
                        f"{metrics['mean_satisfaction_per_user']:.4f}\n"
                    )
                    if min_penalty_enabled:
                        log_f.write(
                            "mean_min_violations = "
                            f"{metrics['mean_min_violations']:.4f} "
                            "std_min_violations = "
                            f"{metrics['std_min_violations']:.4f}\n"
                        )
                    log_f.flush()
                print("\nTraining continues...\n")

            rollout_obs.append(obs)
            rollout_actions.append(action)
            rollout_logprobs.append(logprob)
            rollout_values.append(value)
            rollout_rewards.append(float(reward))
            rollout_dones.append(float(done))

            obs = next_obs
            last_done = done
            if done:
                obs, _ = env.reset()

            if global_step >= timesteps:
                break
        
        model.update(
            obs=np.asarray(rollout_obs, dtype=np.float32),
            actions=np.asarray(rollout_actions, dtype=np.int64),
            logprobs=np.asarray(rollout_logprobs, dtype=np.float32),
            values=np.asarray(rollout_values, dtype=np.float32),
            rewards=np.asarray(rollout_rewards, dtype=np.float32),
            dones=np.asarray(rollout_dones, dtype=np.float32),
            last_obs=np.asarray(obs, dtype=np.float32),
            last_done=last_done,
        )
    print("Learning Completed...")

    model_paths = paths_cfg.get("model_paths", {})
    default_model_path = f"sdn_controller/models/{algo.lower()}_sdn_agent.zip"
    out_path = resolve_path(model_paths.get(algo.lower(), default_model_path))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print(f"Saved model to {out_path}")

    metrics = evaluate_on_user_sets(
        model,
        eval_metrics_env,
        eval_user_sets,
        num_users=num_users,
    )
    print(f"Final eval mean reward: {metrics['mean_reward']:.4f}")
    print(f"Final eval mean reward per user: {metrics['mean_reward_per_user']:.4f}")
    print(f"Final eval reward std: {metrics['std_reward']:.4f}")
    print(f"Final eval mean profit: {metrics['mean_profit']:.4f}")
    print(f"Final eval profit std: {metrics['std_profit']:.4f}")
    print(f"Final eval mean satisfaction: {metrics['mean_satisfaction']:.4f}")
    print(f"Final eval satisfaction std: {metrics['std_satisfaction']:.4f}")
    if min_penalty_enabled:
        print(f"Final eval mean min violations: {metrics['mean_min_violations']:.4f}")
        print(f"Final eval min violations std: {metrics['std_min_violations']:.4f}")
    if log_f is not None:
        log_f.write(f"Final eval mean reward       : {metrics['mean_reward']:.4f}\n")
        log_f.write(f"Final eval mean reward per user : {metrics['mean_reward_per_user']:.4f}\n")
        log_f.write(f"Final eval reward std       : {metrics['std_reward']:.4f}\n")
        log_f.write(f"Final eval mean profit       : {metrics['mean_profit']:.4f}\n")
        log_f.write(f"Final eval profit std       : {metrics['std_profit']:.4f}\n")
        log_f.write(f"Final eval mean satisfaction : {metrics['mean_satisfaction']:.4f}\n")
        log_f.write(f"Final eval satisfaction std : {metrics['std_satisfaction']:.4f}\n")
        if min_penalty_enabled:
            log_f.write(
                f"Final eval mean min violations: {metrics['mean_min_violations']:.4f}\n"
            )
            log_f.write(
                f"Final eval min violations std: {metrics['std_min_violations']:.4f}\n"
            )
    elapsed = time.time() - t0
    if log_f is not None:
        log_f.write(f"Total time : {elapsed:.1f}s\n")
        log_f.write(sep_line + "\n")
        log_f.close()


if __name__ == "__main__":
    main()
