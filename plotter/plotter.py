from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def save_results_plots(
    results: List[Tuple[int, dict]],
    optimizer_name: str,
    out_dir: str = "results",
) -> None:
    out_dir = os.path.join(out_dir, optimizer_name)
    os.makedirs(out_dir, exist_ok=True)
    xs = [r[0] for r in results]
    avg_profits = [r[1]["avg_profit"] for r in results]
    avg_rewards = [r[1]["avg_reward"] for r in results]
    avg_satisfied = [r[1]["avg_satisfied_users"] for r in results]

    plt.figure()
    plt.plot(xs, avg_profits, marker="o")
    plt.xlabel("Number of UEs")
    plt.ylabel("Average Profit")
    plt.title("Number of UEs vs Profit")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"nues_vs_profit_{optimizer_name}.png"))
    plt.close()

    plt.figure()
    plt.plot(xs, avg_rewards, marker="o")
    plt.xlabel("Number of UEs")
    plt.ylabel("Average Reward")
    plt.title("Number of UEs vs Reward")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"nues_vs_reward_{optimizer_name}.png"))
    plt.close()

    plt.figure()
    plt.plot(xs, avg_satisfied, marker="o")
    plt.xlabel("Number of UEs")
    plt.ylabel("Average Satisfied Users")
    plt.title("Number of UEs vs Satisfaction")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"nues_vs_satisfaction_{optimizer_name}.png"))
    plt.close()
