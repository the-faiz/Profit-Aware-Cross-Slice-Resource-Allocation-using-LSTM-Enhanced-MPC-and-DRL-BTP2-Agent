from __future__ import annotations

from utilities.utils import load_config, write_csv
from dataset_generator.mobility_pattern_genererator import MobilityPatternGenerator


def generate_simulation_dataset(
    num_ues: int | None = None,
) -> None:
    cfg = load_config()
    sim_cfg = cfg["simulation"]
    if num_ues is None:
        if "num_ues" not in sim_cfg:
            raise ValueError(
                "num_ues must be provided or simulation.num_ues must exist in config."
            )
        num_ues = int(sim_cfg["num_ues"])
    else:
        num_ues = int(num_ues)
    num_steps = int(sim_cfg["num_steps"])

    gen = MobilityPatternGenerator(num_ues, num_steps)
    rows = gen.generate_mobility_pattern()
    write_csv(sim_cfg["dataset_csv"], rows)
    print(f"simulation dataset saved to {sim_cfg['dataset_csv']}")


if __name__ == "__main__":
    generate_simulation_dataset()
