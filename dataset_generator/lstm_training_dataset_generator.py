from __future__ import annotations

from utilities.utils import load_config, write_csv
from dataset_generator.mobility_pattern_genererator import MobilityPatternGenerator


def generate_lstm_training_dataset() -> None:
    cfg = load_config()
    training_cfg = cfg["lstm"]["training"]
    num_ues = int(training_cfg["num_ues"])
    num_steps = int(training_cfg["num_steps"])

    gen = MobilityPatternGenerator(num_ues, num_steps)
    rows = gen.generate_mobility_pattern()
    write_csv(training_cfg["dataset_csv"], rows)
    print(f"lstm training dataset saved to {training_cfg['dataset_csv']}")


if __name__ == "__main__":
    generate_lstm_training_dataset()
