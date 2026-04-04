from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path as _Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT_DIR = _Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utilities.utils import load_config

def _parse_log(path: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse a simulation log file and return rows of:
    (num_ues, avg_reward, avg_profit, avg_satisfied_users, latency_per_step_s)
    """
    rows: List[Tuple[int, float, float, float, float]] = []
    current_ues: int | None = None
    avg_reward: float | None = None
    avg_profit: float | None = None
    avg_satisfied: float | None = None
    latency_per_step: float | None = None

    sim_re = re.compile(r"^Simulating\s+(\d+)\s+UEs\b")
    avg_reward_re = re.compile(r"^Average reward\s*:\s*([-+]?\d+(?:\.\d+)?)")
    avg_profit_re = re.compile(r"^Average profit\s*:\s*([-+]?\d+(?:\.\d+)?)")
    avg_satisfied_re = re.compile(
        r"^Average satisfied users\s*:\s*([-+]?\d+(?:\.\d+)?)"
    )
    per_step_re = re.compile(
        r"^\s*Total time\s*:\s*[-+]?\d+(?:\.\d+)?s\s*\(([-+]?\d+(?:\.\d+)?)s per step\)"
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
                latency_per_step = None
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

            per_step_match = per_step_re.match(line)
            if per_step_match:
                latency_per_step = float(per_step_match.group(1))

            if (
                current_ues is not None
                and avg_reward is not None
                and avg_profit is not None
                and avg_satisfied is not None
                and latency_per_step is not None
            ):
                rows.append(
                    (current_ues, avg_reward, avg_profit, avg_satisfied, latency_per_step)
                )
                current_ues = None
                avg_reward = None
                avg_profit = None
                avg_satisfied = None
                latency_per_step = None

    return rows


def _plot_metric(
    data: Dict[str, List[Tuple[int, float]]],
    metric_name: str,
    y_label: str,
    out_path: str,
    x_order: List[int] | None = None,
) -> None:
    color_map = {
        "greedy": "#d62728",
        "ga": "#f2b600",
        "pso": "#2f7ed8",
        "average": "#2ca02c",
        "hybrid_avg_deficit": "#bcbd22",
        "deficit_aware": "#ff7f0e",
        "tier_quota": "#17becf",
        "topk_priority": "#8c564b",
        "target_rate": "#e377c2",
        "random": "#7f7f7f",
        "static": "#9467bd",
    }
    label_map = {}

    plt.figure(figsize=(8.2, 5.4))
    for optimizer, series in sorted(data.items()):
        xs = [x for x, _ in series]
        ys = [y for _, y in series]
        if not xs:
            continue
        color = color_map.get(optimizer, None)
        label = label_map.get(optimizer, optimizer.upper())
        plt.plot(
            xs,
            ys,
            marker="D",
            markersize=5.5,
            linewidth=2.0,
            color=color,
            label=label,
        )
    plt.xlabel("Number of UEs")
    plt.ylabel(y_label)
    plt.title(f"Number of UEs vs {metric_name} (All Optimizers)")
    if x_order:
        all_xs = [x for x in x_order if any(x == sx for series in data.values() for sx, _ in series)]
    else:
        all_xs = sorted({x for series in data.values() for x, _ in series})
    if all_xs:
        plt.xticks(all_xs)
    plt.grid(True, color="#b0b0b0", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_metric_bar(
    data: Dict[str, List[Tuple[int, float]]],
    metric_name: str,
    y_label: str,
    out_path: str,
    x_order: List[int] | None = None,
) -> None:
    color_map = {
        "greedy": "#d62728",
        "ga": "#f2b600",
        "pso": "#2f7ed8",
        "average": "#2ca02c",
        "hybrid_avg_deficit": "#bcbd22",
        "deficit_aware": "#ff7f0e",
        "tier_quota": "#17becf",
        "topk_priority": "#8c564b",
        "target_rate": "#e377c2",
        "random": "#7f7f7f",
        "static": "#9467bd",
    }
    label_map = {}

    optimizers = sorted(data.keys())
    if x_order:
        all_xs = [x for x in x_order if any(x == sx for series in data.values() for sx, _ in series)]
    else:
        all_xs = sorted({x for series in data.values() for x, _ in series})
    if not optimizers or not all_xs:
        return

    index = list(range(len(all_xs)))
    n_opts = len(optimizers)
    total_width = 0.8
    bar_width = total_width / max(1, n_opts)
    start = -total_width / 2.0 + bar_width / 2.0

    plt.figure(figsize=(8.2, 5.4))
    for opt_idx, optimizer in enumerate(optimizers):
        series = dict(data.get(optimizer, []))
        offsets = []
        ys = []
        for i, x in enumerate(all_xs):
            if x in series:
                offsets.append(i + start + opt_idx * bar_width)
                ys.append(series[x])
        color = color_map.get(optimizer, None)
        label = label_map.get(optimizer, optimizer.upper())
        plt.bar(
            offsets,
            ys,
            width=bar_width,
            color=color,
            label=label,
        )

    plt.xlabel("Number of UEs")
    plt.ylabel(y_label)
    plt.title(f"Number of UEs vs {metric_name} (All Optimizers)")
    plt.xticks(index, [str(x) for x in all_xs])
    plt.grid(True, axis="y", color="#b0b0b0", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _build_legend_handles(
    data: Dict[str, List[Tuple[int, float]]],
) -> List[Line2D]:
    color_map = {
        "greedy": "#d62728",
        "ga": "#f2b600",
        "pso": "#2f7ed8",
        "average": "#2ca02c",
        "hybrid_avg_deficit": "#bcbd22",
        "deficit_aware": "#ff7f0e",
        "tier_quota": "#17becf",
        "topk_priority": "#8c564b",
        "target_rate": "#e377c2",
        "random": "#7f7f7f",
        "static": "#9467bd",
    }
    label_map = {}
    handles: List[Line2D] = []
    for optimizer in sorted(data.keys()):
        color = color_map.get(optimizer, None)
        label = label_map.get(optimizer, optimizer.upper())
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="D",
                markersize=5.5,
                linewidth=2.0,
                label=label,
            )
        )
    return handles


def _save_legend(handles: List[Line2D], out_path: str) -> None:
    if not handles:
        return
    fig = plt.figure(figsize=(8.2, max(1.2, 0.4 * len(handles))))
    fig.legend(
        handles=handles,
        loc="center",
        frameon=True,
        ncol=1,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(root_dir, "logs")
    out_dir = os.path.join(root_dir, "results", "combined")
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config()
    plot_cfg = cfg.get("plotting", {})
    plot_type = str(plot_cfg.get("plot_type", "line")).lower()
    num_ues_list = plot_cfg.get("num_ues_list", [])
    if not isinstance(num_ues_list, list) or not num_ues_list:
        raise SystemExit("plotting.num_ues_list must be a non-empty list in config.")

    log_paths = sorted(glob.glob(os.path.join(logs_dir, "simulation_*.log")))
    if not log_paths:
        raise SystemExit(f"No log files found in {logs_dir}")

    reward_data: Dict[str, List[Tuple[int, float]]] = {}
    profit_data: Dict[str, List[Tuple[int, float]]] = {}
    satisfied_data: Dict[str, List[Tuple[int, float]]] = {}
    latency_data: Dict[str, List[Tuple[int, float]]] = {}
    reward_pp_data: Dict[str, List[Tuple[int, float]]] = {}
    profit_pp_data: Dict[str, List[Tuple[int, float]]] = {}
    satisfied_pp_data: Dict[str, List[Tuple[int, float]]] = {}

    for path in log_paths:
        optimizer = os.path.basename(path).replace("simulation_", "").replace(".log", "")
        rows = _parse_log(path)
        if not rows:
            continue
        order_index = {v: i for i, v in enumerate(num_ues_list)}
        rows = [r for r in rows if r[0] in order_index]
        rows.sort(key=lambda r: order_index[r[0]])

        reward_data[optimizer] = [(r[0], r[1]) for r in rows]
        profit_data[optimizer] = [(r[0], r[2]) for r in rows]
        satisfied_data[optimizer] = [(r[0], r[3]) for r in rows]
        latency_data[optimizer] = [(r[0], r[4]) for r in rows]
        reward_pp_data[optimizer] = [
            (r[0], r[1] / r[0] if r[0] else 0.0) for r in rows
        ]
        profit_pp_data[optimizer] = [
            (r[0], r[2] / r[0] if r[0] else 0.0) for r in rows
        ]
        satisfied_pp_data[optimizer] = [
            (r[0], r[3] / r[0] if r[0] else 0.0) for r in rows
        ]

    plotter = _plot_metric_bar if plot_type == "bar" else _plot_metric

    plotter(
        profit_data,
        "Profit",
        "Average Profit",
        os.path.join(out_dir, "nues_vs_profit_all.png"),
        x_order=num_ues_list,
    )
    plotter(
        reward_data,
        "Reward",
        "Average Reward",
        os.path.join(out_dir, "nues_vs_reward_all.png"),
        x_order=num_ues_list,
    )
    plotter(
        satisfied_data,
        "Satisfaction",
        "Average Satisfied Users",
        os.path.join(out_dir, "nues_vs_satisfaction_all.png"),
        x_order=num_ues_list,
    )
    plotter(
        latency_data,
        "Allocation Latency per Step",
        "Latency per Step (s)",
        os.path.join(out_dir, "nues_vs_latency_per_step_all.png"),
        x_order=num_ues_list,
    )
    plotter(
        reward_pp_data,
        "Reward per Person",
        "Average Reward per Person",
        os.path.join(out_dir, "nues_vs_reward_per_person_all.png"),
        x_order=num_ues_list,
    )
    plotter(
        profit_pp_data,
        "Profit per Person",
        "Average Profit per Person",
        os.path.join(out_dir, "nues_vs_profit_per_person_all.png"),
        x_order=num_ues_list,
    )
    plotter(
        satisfied_pp_data,
        "Satisfaction per Person",
        "Average Satisfaction per Person",
        os.path.join(out_dir, "nues_vs_satisfaction_per_person_all.png"),
        x_order=num_ues_list,
    )

    handles = _build_legend_handles(reward_data)
    _save_legend(handles, os.path.join(out_dir, "one.png"))

    print(f"Saved combined plots to {out_dir}")


if __name__ == "__main__":
    main()
