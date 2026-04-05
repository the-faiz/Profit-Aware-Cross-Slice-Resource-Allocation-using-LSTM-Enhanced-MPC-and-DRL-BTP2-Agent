from __future__ import annotations

import sys
from pathlib import Path as _Path

ROOT_DIR = _Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sdn_controller.utilities.utils import load_config, resolve_path


def _bar_width(xs: list[int | float], group_count: int = 1) -> float:
    if not xs:
        return 0.8
    xs_sorted = sorted({float(x) for x in xs})
    if len(xs_sorted) <= 1:
        return 0.8
    diffs = [b - a for a, b in zip(xs_sorted, xs_sorted[1:]) if b > a]
    min_diff = min(diffs) if diffs else 1.0
    return max(0.1, 0.8 * min_diff / max(1, group_count))


def _catmull_rom_spline(
    xs: list[float],
    ys: list[float],
    points_per_segment: int,
) -> tuple[list[float], list[float]]:
    if len(xs) < 3 or points_per_segment <= 1:
        return xs, ys
    pts = list(zip(xs, ys))
    extended = [pts[0]] + pts + [pts[-1]]
    out_x: list[float] = []
    out_y: list[float] = []
    for i in range(1, len(extended) - 2):
        p0 = np.array(extended[i - 1], dtype=np.float64)
        p1 = np.array(extended[i], dtype=np.float64)
        p2 = np.array(extended[i + 1], dtype=np.float64)
        p3 = np.array(extended[i + 2], dtype=np.float64)
        for t in np.linspace(0.0, 1.0, points_per_segment, endpoint=False):
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out_x.append(float(point[0]))
            out_y.append(float(point[1]))
    out_x.append(xs[-1])
    out_y.append(ys[-1])
    return out_x, out_y


def _plot_training_combined(
    log_dir: str,
    algos: list[str],
    metric_key: str,
    title: str,
    y_label: str,
    out_path: str,
    plot_type: str,
    smooth_training_curve: bool = False,
    smooth_points_per_segment: int = 20,
) -> bool:
    series = []
    for algo in algos:
        log_path = os.path.join(log_dir, f"simulation_{algo}.log")
        data = _parse_train_log(log_path)
        if not data:
            continue
        series.append((algo.upper(), data.get("steps", []), data.get(metric_key, [])))
    if not series:
        return False
    plt.figure(figsize=(7.5, 4.5))
    if plot_type == "bar":
        xs_all = sorted({int(x) for _, steps, _ in series for x in steps})
        width = _bar_width(xs_all, group_count=len(series))
        for i, (name, steps, values) in enumerate(series):
            if not values:
                continue
            offsets = [x + (i - (len(series) - 1) / 2) * width for x in steps]
            plt.bar(offsets, values, width=width, alpha=0.75, label=name)
        plt.xticks(xs_all)
    else:
        for name, steps, values in series:
            if not values:
                continue
            plot_xs = [float(x) for x in steps]
            plot_ys = [float(y) for y in values]
            if smooth_training_curve and len(plot_xs) >= 3:
                plot_xs, plot_ys = _catmull_rom_spline(
                    plot_xs, plot_ys, smooth_points_per_segment
                )
                plt.plot(plot_xs, plot_ys, linewidth=2, label=name)
                plt.scatter(steps, values, s=18, alpha=0.7)
            else:
                plt.plot(steps, values, marker="o", linewidth=2, label=name)
    plt.xlabel("Checkpoint")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _parse_train_log(train_log_path: str) -> dict[str, list[float]]:
    if not os.path.exists(train_log_path):
        return {}
    steps: list[int] = []
    rewards: list[float] = []
    rewards_pp: list[float] = []
    sats: list[float] = []
    profits: list[float] = []
    profits_pp: list[float] = []
    sats_pp: list[float] = []
    current_step: int | None = None
    current: dict[str, float] = {}
    with open(train_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Checkpoint "):
                if current_step is not None and current:
                    steps.append(current_step)
                    rewards.append(current.get("mean_reward", 0.0))
                    profits.append(current.get("mean_profit", 0.0))
                    sats.append(current.get("mean_satisfaction", 0.0))
                    rewards_pp.append(current.get("mean_reward_per_user", 0.0))
                    profits_pp.append(current.get("mean_profit_per_user", 0.0))
                    sats_pp.append(current.get("mean_satisfaction_per_user", 0.0))
                    current = {}
                try:
                    parts = line.split()
                    current_step = int(parts[1].rstrip(":"))
                except Exception:
                    current_step = None
                if ":" in line and "avg_" in line:
                    fields = line.split(":", 1)[1]
                    kv = {}
                    for item in fields.split(","):
                        if "=" in item:
                            k, v = item.strip().split("=", 1)
                            kv[k.strip()] = float(v)
                    if current_step is not None:
                        steps.append(current_step)
                        rewards.append(kv.get("avg_reward", 0.0))
                        profits.append(kv.get("avg_profit", 0.0))
                        sats.append(kv.get("avg_satisfaction", 0.0))
                        rewards_pp.append(kv.get("avg_reward_per_user", 0.0))
                        profits_pp.append(kv.get("avg_profit_per_user", 0.0))
                        sats_pp.append(kv.get("avg_satisfaction_per_user", 0.0))
                        current_step = None
                        current = {}
                continue
            if current_step is None:
                continue
            if line.startswith("mean_reward"):
                parts = line.replace("=", "").split()
                if len(parts) >= 6:
                    current["mean_reward"] = float(parts[1])
                    current["std_reward"] = float(parts[3])
                    current["mean_reward_per_user"] = float(parts[5])
                continue
            if line.startswith("mean_profit"):
                parts = line.replace("=", "").split()
                if len(parts) >= 6:
                    current["mean_profit"] = float(parts[1])
                    current["std_profit"] = float(parts[3])
                    current["mean_profit_per_user"] = float(parts[5])
                continue
            if line.startswith("mean_sat"):
                parts = line.replace("=", "").split()
                if len(parts) >= 6:
                    current["mean_satisfaction"] = float(parts[1])
                    current["std_satisfaction"] = float(parts[3])
                    current["mean_satisfaction_per_user"] = float(parts[5])
                continue
        if current_step is not None and current:
            steps.append(current_step)
            rewards.append(current.get("mean_reward", 0.0))
            profits.append(current.get("mean_profit", 0.0))
            sats.append(current.get("mean_satisfaction", 0.0))
            rewards_pp.append(current.get("mean_reward_per_user", 0.0))
            profits_pp.append(current.get("mean_profit_per_user", 0.0))
            sats_pp.append(current.get("mean_satisfaction_per_user", 0.0))
    return {
        "steps": steps,
        "reward": rewards,
        "reward_pp": rewards_pp,
        "satisfaction": sats,
        "profit": profits,
        "profit_pp": profits_pp,
        "satisfaction_pp": sats_pp,
    }


def _plot_eval_metric(
    episodes: list[int],
    values: list[float],
    title: str,
    y_label: str,
    out_path: str,
    plot_type: str,
) -> bool:
    if not values:
        return False
    plt.figure(figsize=(7.5, 4.5))
    if plot_type == "bar":
        width = _bar_width(episodes, group_count=1)
        plt.bar(episodes, values, width=width, color="#2f7ed8")
    else:
        plt.plot(episodes, values, marker="o", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.4)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _parse_eval_log(eval_log_path: str) -> list[dict[str, float]]:
    if not os.path.exists(eval_log_path):
        return []
    rows: list[dict[str, float]] = []
    current: dict[str, float] = {}
    with open(eval_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if current:
                    rows.append(current)
                    current = {}
                continue
            if line.startswith("Algo:"):
                header = line.split(":", 1)[1].strip()
                parts = [p.strip() for p in header.split("|")]
                for part in parts:
                    if part.lower().startswith("num_users"):
                        try:
                            current["num_users"] = int(part.split(":", 1)[1].strip())
                        except ValueError:
                            pass
                    else:
                        current["algo"] = part
                continue
            if line.startswith("mean_reward"):
                parts = line.replace("=", "").split()
                # mean_reward <val> std_reward <val> mean_reward_per_user <val>
                if len(parts) >= 6:
                    current["mean_reward"] = float(parts[1])
                    current["std_reward"] = float(parts[3])
                    current["mean_reward_per_user"] = float(parts[5])
                continue
            if line.startswith("mean_profit"):
                parts = line.replace("=", "").split()
                # mean_profit <val> std_profit <val> profit_per_user <val>
                if len(parts) >= 6:
                    current["mean_profit"] = float(parts[1])
                    current["std_profit"] = float(parts[3])
                    current["mean_profit_per_user"] = float(parts[5])
                continue
            if line.startswith("mean_sat"):
                parts = line.replace("=", "").split()
                # mean_sat <val> std_sat <val> mean_sat_per_user <val>
                if len(parts) >= 6:
                    current["mean_satisfaction"] = float(parts[1])
                    current["std_satisfaction"] = float(parts[3])
                    current["mean_satisfaction_per_user"] = float(parts[5])
                continue
            if line.startswith("mean_latency"):
                parts = line.replace("=", "").split()
                if len(parts) >= 2:
                    current["mean_latency"] = float(parts[1])
                continue
    if current:
        rows.append(current)
    return rows


def _plot_eval_combined(
    eval_log: str,
    out_path: str,
    plot_type: str,
    metric_key: str,
    title: str,
    y_label: str,
    trained_users: int | None = None,
) -> bool:
    if not os.path.exists(eval_log):
        return False
    rows = _parse_eval_log(eval_log)
    if not rows:
        return False
    algos = sorted({r["algo"] for r in rows if "algo" in r})
    xs_all = sorted({int(r["num_users"]) for r in rows if "num_users" in r})
    if not xs_all:
        return False
    plt.figure(figsize=(7.5, 4.5))
    if plot_type == "bar":
        width = _bar_width(xs_all, group_count=len(algos))
        for i, algo in enumerate(algos):
            sub = [r for r in rows if r.get("algo") == algo]
            if trained_users is not None and str(algo).upper() == "PPO":
                sub = [r for r in sub if int(r.get("num_users", -1)) == int(trained_users)]
            sub.sort(key=lambda r: int(r["num_users"]))
            xs = [int(r["num_users"]) for r in sub]
            ys = [float(r.get(metric_key, 0.0)) for r in sub]
            offsets = [x + (i - (len(algos) - 1) / 2) * width for x in xs]
            plt.bar(offsets, ys, width=width, alpha=0.75, label=algo)
        plt.xticks(xs_all)
    else:
        for algo in algos:
            sub = [r for r in rows if r.get("algo") == algo]
            if trained_users is not None and str(algo).upper() == "PPO":
                sub = [r for r in sub if int(r.get("num_users", -1)) == int(trained_users)]
            sub.sort(key=lambda r: int(r["num_users"]))
            xs = [int(r["num_users"]) for r in sub]
            ys = [float(r.get(metric_key, 0.0)) for r in sub]
            plt.plot(xs, ys, marker="o", linewidth=2, label=algo)
    plt.xlabel("Number of Users")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def main() -> None:
    cfg = load_config()
    train_cfg = cfg["training"]
    plot_cfg = cfg.get("plotting", {})
    eval_cfg = cfg.get("evaluation", {})
    paths_cfg = cfg.get("paths", {})
    plot_type = str(plot_cfg.get("plot_type", "line")).lower()
    smooth_training_curve = bool(plot_cfg.get("smooth_training_curve", False))
    smooth_points_per_segment = int(plot_cfg.get("smooth_points_per_segment", 20))
    trained_users = int(cfg["environment"]["num_users"])

    log_dir = resolve_path(
        paths_cfg.get("log_dir_training", "sdn_controller/logs/training")
    )
    train_plot_path = resolve_path(
        paths_cfg.get(
            "train_plot_path", "sdn_controller/results/training/train_reward_curve.png"
        )
    )
    eval_log = resolve_path(
        paths_cfg.get(
            "eval_output_path", "sdn_controller/logs/evaluation/eval_results.log"
        )
    )
    eval_plot_path = resolve_path(
        paths_cfg.get(
            "eval_plot_path", "sdn_controller/results/evaluations/eval_reward.png"
        )
    )

    algos = ["ppo", "a2c"]
    train_plots = [
        ("reward", "Training Checkpoint Reward", "Reward", "train_reward"),
        ("profit", "Training Checkpoint Profit", "Profit", "train_profit"),
        ("satisfaction", "Training Checkpoint Satisfaction", "Satisfaction", "train_satisfaction"),
        ("reward_pp", "Training Checkpoint Reward per User", "Reward per User", "train_reward_per_user"),
        ("profit_pp", "Training Checkpoint Profit per User", "Profit per User", "train_profit_per_user"),
        ("satisfaction_pp", "Training Checkpoint Satisfaction per User", "Satisfaction per User", "train_satisfaction_per_user"),
    ]
    any_train = False
    for metric_key, title, y_label, suffix in train_plots:
        out_path = train_plot_path.replace(".png", f"_{suffix}.png")
        did = _plot_training_combined(
            log_dir,
            algos,
            metric_key,
            title,
            y_label,
            out_path,
            plot_type,
            smooth_training_curve=smooth_training_curve,
            smooth_points_per_segment=smooth_points_per_segment,
        )
        any_train = any_train or did
    if any_train:
        print(f"Saved training plots to {os.path.dirname(train_plot_path)}")
    else:
        print("Training plots not generated (missing PPO/A2C logs).")

    eval_plots = [
        ("mean_reward", "Evaluation Mean Reward by Algo", "Mean Reward", "eval_reward"),
        ("std_reward", "Evaluation Reward Std by Algo", "Reward Std", "eval_reward_std"),
        ("mean_reward_per_user", "Evaluation Mean Reward per User", "Mean Reward per User", "eval_reward_per_user"),
        ("mean_profit", "Evaluation Mean Profit by Algo", "Mean Profit", "eval_profit"),
        ("std_profit", "Evaluation Profit Std by Algo", "Profit Std", "eval_profit_std"),
        ("mean_profit_per_user", "Evaluation Mean Profit per User", "Mean Profit per User", "eval_profit_per_user"),
        ("mean_satisfaction", "Evaluation Mean Satisfaction by Algo", "Mean Satisfaction", "eval_satisfaction"),
        ("std_satisfaction", "Evaluation Satisfaction Std by Algo", "Satisfaction Std", "eval_satisfaction_std"),
        ("mean_satisfaction_per_user", "Evaluation Mean Satisfaction per User", "Mean Satisfaction per User", "eval_satisfaction_per_user"),
        ("mean_latency", "Evaluation Mean Latency by Algo", "Mean Latency (s)", "eval_latency"),
    ]
    any_eval = False
    for metric_key, title, y_label, suffix in eval_plots:
        out_path = eval_plot_path.replace(".png", f"_{suffix}.png")
        did = _plot_eval_combined(
            eval_log,
            out_path,
            plot_type,
            metric_key,
            title,
            y_label,
            trained_users=trained_users,
        )
        any_eval = any_eval or did
    if any_eval:
        print(f"Saved evaluation plots to {os.path.dirname(eval_plot_path)}")
    else:
        print("Evaluation plots not generated (missing eval_results.log).")


if __name__ == "__main__":
    main()
