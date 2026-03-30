from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _parse_log(path: str) -> List[Tuple[int, float, float, float]]:
    """
    Parse a simulation log file and return rows of:
    (num_ues, avg_reward, avg_profit, avg_satisfied_users)
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
                # If a previous block was partially parsed, discard it.
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


def _plot_metric(
    data: Dict[str, List[Tuple[int, float]]],
    metric_name: str,
    y_label: str,
    out_path: str,
) -> None:
    plt.figure()
    for optimizer, series in sorted(data.items()):
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        if not xs:
            continue
        plt.plot(xs, ys, marker="o", label=optimizer)
    plt.xlabel("Number of UEs")
    plt.ylabel(y_label)
    plt.title(f"Number of UEs vs {metric_name} (All Optimizers)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(root_dir, "logs")
    out_dir = os.path.join(root_dir, "results", "combined")
    os.makedirs(out_dir, exist_ok=True)

    log_paths = sorted(glob.glob(os.path.join(logs_dir, "simulation_*.log")))
    if not log_paths:
        raise SystemExit(f"No log files found in {logs_dir}")

    reward_data: Dict[str, List[Tuple[int, float]]] = {}
    profit_data: Dict[str, List[Tuple[int, float]]] = {}
    satisfied_data: Dict[str, List[Tuple[int, float]]] = {}
    reward_pp_data: Dict[str, List[Tuple[int, float]]] = {}
    profit_pp_data: Dict[str, List[Tuple[int, float]]] = {}
    satisfied_pp_data: Dict[str, List[Tuple[int, float]]] = {}

    for path in log_paths:
        optimizer = os.path.basename(path).replace("simulation_", "").replace(".log", "")
        rows = _parse_log(path)
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])

        reward_data[optimizer] = [(r[0], r[1]) for r in rows]
        profit_data[optimizer] = [(r[0], r[2]) for r in rows]
        satisfied_data[optimizer] = [(r[0], r[3]) for r in rows]
        reward_pp_data[optimizer] = [
            (r[0], r[1] / r[0] if r[0] else 0.0) for r in rows
        ]
        profit_pp_data[optimizer] = [
            (r[0], r[2] / r[0] if r[0] else 0.0) for r in rows
        ]
        satisfied_pp_data[optimizer] = [
            (r[0], r[3] / r[0] if r[0] else 0.0) for r in rows
        ]

    _plot_metric(
        profit_data,
        "Profit",
        "Average Profit",
        os.path.join(out_dir, "nues_vs_profit_all.png"),
    )
    _plot_metric(
        reward_data,
        "Reward",
        "Average Reward",
        os.path.join(out_dir, "nues_vs_reward_all.png"),
    )
    _plot_metric(
        satisfied_data,
        "Satisfaction",
        "Average Satisfied Users",
        os.path.join(out_dir, "nues_vs_satisfaction_all.png"),
    )
    _plot_metric(
        reward_pp_data,
        "Reward per Person",
        "Average Reward per Person",
        os.path.join(out_dir, "nues_vs_reward_per_person_all.png"),
    )
    _plot_metric(
        profit_pp_data,
        "Profit per Person",
        "Average Profit per Person",
        os.path.join(out_dir, "nues_vs_profit_per_person_all.png"),
    )
    _plot_metric(
        satisfied_pp_data,
        "Satisfaction per Person",
        "Average Satisfaction per Person",
        os.path.join(out_dir, "nues_vs_satisfaction_per_person_all.png"),
    )

    print(f"Saved combined plots to {out_dir}")


if __name__ == "__main__":
    main()
