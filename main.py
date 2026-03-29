#Main pipeline

from __future__ import annotations

from utils import load_config, write_csv
from mobility_pattern_genererator import MobilityPatternGenerator


def main() -> None:
    cfg = load_config("config.yaml")    
    sim_cfg = cfg["simulation"]
    num_ues = int(sim_cfg["num_ues"])
    num_steps = int(sim_cfg["num_steps"])

    print(f"Simulation Starting for {num_ues} users and {num_steps} steps")

    gen = MobilityPatternGenerator(num_ues, num_steps)
    rows = gen.generate_mobility_pattern()
    write_csv(sim_cfg["dataset_csv"], rows)


if __name__ == "__main__":
    main()
