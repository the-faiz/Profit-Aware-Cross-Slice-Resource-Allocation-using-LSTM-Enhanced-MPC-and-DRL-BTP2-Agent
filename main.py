#Main pipeline

from __future__ import annotations

import argparse

from utilities.utils import load_config, write_csv
from dataset_generator.mobility_pattern_genererator import MobilityPatternGenerator
from simulation.mpc_simulator import MPCSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MPC simulation with selectable optimizer.")
    parser.add_argument(
        "--optimizer",
        default="ga",
        help="Optimizer to use: ga, pso, greedy, static, average, random",
    )
    args = parser.parse_args()

    print(f"Optimizer selected: {args.optimizer}")

    print("Loading the configuration")
    cfg = load_config("configurations/config.yaml")
    sim_cfg = cfg["simulation"]
    num_ues = int(sim_cfg["num_ues"])
    num_steps = int(sim_cfg["num_steps"])

    print(f"Number of UEs: {num_ues}, Number of steps: {num_steps}")

    print("Generating mobility patterns")
    gen = MobilityPatternGenerator(num_ues, num_steps)
    rows = gen.generate_mobility_pattern()
    write_csv(sim_cfg["dataset_csv"], rows)

    print("MPC Simulation phase")
    MPCSimulator(optimizer_name=args.optimizer).run()


if __name__ == "__main__":
    main()
