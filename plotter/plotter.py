from __future__ import annotations

import glob
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


def _parse_log(path: str) -> List[Tuple[int, float, float, float]]:
    """
    Return rows of (num_ues, avg_reward, avg_profit, avg_satisfied_users).
    """
    rows: List[Tuple[int, float, float, float]] = []
    current_ues: int | None = None
    avg_reward: float | None = None
    avg_profit: float | None = None
    avg_satisfied: float | None = None

    sim_re = re.compile(r"^Simulating\s+(\d+)\s+UEs\b")
    avg_reward_re = re.compile(r"^Average reward\s*:\s*([-+]?\d+(?:\.\d+)?)")
    avg_profit_re = re.compile(r"^Average profit\s*:\s*([-+]?\d+(?:\.\d+)?)")
    avg_satisfied_re = re.compile(
        r"^Average satisfied users\s*:\s*([-+]?\d+(?:\.\d+)?)"
    )

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            sim_match = sim_re.match(line)
            if sim_match:
                current_ues = int(sim_match.group(1))
                avg_reward = None
                avg_profit = None
                avg_satisfied = None
                continue

            if current_ues is None:
                continue

            reward_match = avg_reward_re.match(line)
            if reward_match:
                avg_reward = float(reward_match.group(1))
                continue

            profit_match = avg_profit_re.match(line)
            if profit_match:
                avg_profit = float(profit_match.group(1))
                continue

            satisfied_match = avg_satisfied_re.match(line)
            if satisfied_match:
                avg_satisfied = float(satisfied_match.group(1))

            if (
                current_ues is not None
                and avg_reward is not None
                and avg_profit is not None
                and avg_satisfied is not None
            ):
                rows.append((current_ues, avg_reward, avg_profit, avg_satisfied))
                current_ues = None
                avg_reward = None
                avg_profit = None
                avg_satisfied = None

    return rows


def _plot_series(
    xs: List[int],
    ys: List[float],
    title: str,
    y_label: str,
    out_path: str,
) -> None:
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Number of UEs")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    """
    Plot per-optimizer graphs from logs/ into results/<optimizer>/.
    Produces both aggregate and per-person metrics.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(root_dir, "logs")
    out_dir = os.path.join(root_dir, "results")

    log_paths = sorted(glob.glob(os.path.join(logs_dir, "simulation_*.log")))
    if not log_paths:
        raise SystemExit(f"No log files found in {logs_dir}")

    for path in log_paths:
        optimizer = os.path.basename(path).replace("simulation_", "").replace(".log", "")
        rows = _parse_log(path)
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])

        xs = [r[0] for r in rows]
        avg_reward = [r[1] for r in rows]
        avg_profit = [r[2] for r in rows]
        avg_satisfied = [r[3] for r in rows]

        avg_reward_pp = [r[1] / r[0] if r[0] else 0.0 for r in rows]
        avg_profit_pp = [r[2] / r[0] if r[0] else 0.0 for r in rows]
        avg_satisfied_pp = [r[3] / r[0] if r[0] else 0.0 for r in rows]

        opt_out_dir = os.path.join(out_dir, optimizer)
        os.makedirs(opt_out_dir, exist_ok=True)

        _plot_series(
            xs,
            avg_profit,
            "Number of UEs vs Profit",
            "Average Profit",
            os.path.join(opt_out_dir, f"nues_vs_profit_{optimizer}.png"),
        )
        _plot_series(
            xs,
            avg_reward,
            "Number of UEs vs Reward",
            "Average Reward",
            os.path.join(opt_out_dir, f"nues_vs_reward_{optimizer}.png"),
        )
        _plot_series(
            xs,
            avg_satisfied,
            "Number of UEs vs Satisfaction",
            "Average Satisfied Users",
            os.path.join(opt_out_dir, f"nues_vs_satisfaction_{optimizer}.png"),
        )
        _plot_series(
            xs,
            avg_reward_pp,
            "Number of UEs vs Average Reward per Person",
            "Average Reward per Person",
            os.path.join(opt_out_dir, f"nues_vs_reward_per_person_{optimizer}.png"),
        )
        _plot_series(
            xs,
            avg_profit_pp,
            "Number of UEs vs Average Profit per Person",
            "Average Profit per Person",
            os.path.join(opt_out_dir, f"nues_vs_profit_per_person_{optimizer}.png"),
        )
        _plot_series(
            xs,
            avg_satisfied_pp,
            "Number of UEs vs Average Satisfaction per Person",
            "Average Satisfaction per Person",
            os.path.join(opt_out_dir, f"nues_vs_satisfaction_per_person_{optimizer}.png"),
        )

    print(f"Saved per-optimizer plots to {out_dir}")


if __name__ == "__main__":
    main()
