#Main pipeline

from __future__ import annotations

from utils import load_config
from lstm_train import main as train_lstm_main
from lstm_forecast import main as lstm_forecast_main


def main() -> None:
    cfg = load_config("config.yaml")

    print("Training Already Done")

    print("Inference phase")
    lstm_forecast_main()


if __name__ == "__main__":
    main()
