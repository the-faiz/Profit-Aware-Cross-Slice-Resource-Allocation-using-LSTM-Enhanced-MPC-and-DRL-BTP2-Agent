from __future__ import annotations

from utilities.utils import load_config, write_csv
from dataset_generator.mobility_pattern_genererator import MobilityPatternGenerator


def generate_simulation_dataset(config_path: str = "configurations/config.yaml") -> None:
    cfg = load_config(config_path)
    sim_cfg = cfg["simulation"]
    num_ues = int(sim_cfg["num_ues"])
    num_steps = int(sim_cfg["num_steps"])

    gen = MobilityPatternGenerator(num_ues, num_steps)
    rows = gen.generate_mobility_pattern()
    write_csv(sim_cfg["dataset_csv"], rows)
    print(f"simulation dataset saved to {sim_cfg['dataset_csv']}")


if __name__ == "__main__":
    generate_simulation_dataset()
