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


def plot_per_person_from_logs(
    logs_dir: str = "logs",
    out_dir: str = "results",
) -> None:
    """
    Plot per-person metrics vs number of UEs for each optimizer log.
    Outputs to results/<optimizer>/.
    """
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
        avg_reward_pp = [r[1] / r[0] if r[0] else 0.0 for r in rows]
        avg_profit_pp = [r[2] / r[0] if r[0] else 0.0 for r in rows]
        avg_satisfied_pp = [r[3] / r[0] if r[0] else 0.0 for r in rows]

        opt_out_dir = os.path.join(out_dir, optimizer)
        os.makedirs(opt_out_dir, exist_ok=True)

        plt.figure()
        plt.plot(xs, avg_reward_pp, marker="o")
        plt.xlabel("Number of UEs")
        plt.ylabel("Average Reward per Person")
        plt.title("Number of UEs vs Average Reward per Person")
        plt.tight_layout()
        plt.savefig(os.path.join(opt_out_dir, f"nues_vs_reward_per_person_{optimizer}.png"))
        plt.close()

        plt.figure()
        plt.plot(xs, avg_profit_pp, marker="o")
        plt.xlabel("Number of UEs")
        plt.ylabel("Average Profit per Person")
        plt.title("Number of UEs vs Average Profit per Person")
        plt.tight_layout()
        plt.savefig(os.path.join(opt_out_dir, f"nues_vs_profit_per_person_{optimizer}.png"))
        plt.close()

        plt.figure()
        plt.plot(xs, avg_satisfied_pp, marker="o")
        plt.xlabel("Number of UEs")
        plt.ylabel("Average Satisfaction per Person")
        plt.title("Number of UEs vs Average Satisfaction per Person")
        plt.tight_layout()
        plt.savefig(
            os.path.join(opt_out_dir, f"nues_vs_satisfaction_per_person_{optimizer}.png")
        )
        plt.close()

    print("Saved per-person plots for each optimizer.")


if __name__ == "__main__":
    plot_per_person_from_logs()
