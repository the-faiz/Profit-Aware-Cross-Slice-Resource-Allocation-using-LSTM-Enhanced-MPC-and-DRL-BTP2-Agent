#Main pipeline

from __future__ import annotations

import argparse
import os

from dataset_generator.simulation_dataset_generator import generate_simulation_dataset
from simulation.mpc_simulator import MPCSimulator
from utilities.utils import get_base_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MPC simulation with selectable optimizer.")
    parser.add_argument(
        "--optimizer",
        default="ga",
        help=(
            "Optimizer to use: ga, pso, greedy, static, average, hybrid_avg_deficit, "
            "deficit_aware, tier_quota, topk_priority, target_rate, random"
        ),
    )
    args = parser.parse_args()

    print(f"Optimizer selected: {args.optimizer}")

    print("Loading the configuration")
    cfg = load_config()
    sim_cfg = cfg["simulation"]
    num_steps = int(sim_cfg["num_steps"])
    num_ues_list = sim_cfg.get("num_ues_list")

    if not isinstance(num_ues_list, list) or not num_ues_list:
        raise ValueError(
            "simulation.num_ues_list must be provided as a non-empty list "
            "in gNodeB/configurations/config.yaml."
        )

    results = []
    base_dir = get_base_dir()
    log_dir = base_dir / "logs"
    log_path = log_dir / f"simulation_{args.optimizer}.log"
    os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_f:
        for idx, n in enumerate(num_ues_list):
            num_ues = int(n)
            print(f"Number of UEs: {num_ues}, Number of steps: {num_steps}")
            print("Generating mobility patterns")
            generate_simulation_dataset(num_ues=num_ues)
            print("MPC Simulation phase")
            stats = MPCSimulator(
                optimizer_name=args.optimizer,
                num_ues_override=num_ues,
                log_file=log_f,
            ).run()
            results.append((num_ues, stats))
            if idx < len(num_ues_list) - 1:
                print("-" * 80)

    return None


if __name__ == "__main__":
    main()
