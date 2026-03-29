# LSTM-based Trajectory Forecasting

from __future__ import annotations

import csv
import json
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

Point = Tuple[float, float]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, pred_len: int):
        super().__init__()
        self.pred_len = pred_len
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, pred_len * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        y = self.fc(last)
        return y.view(x.size(0), self.pred_len, 2)


def main() -> None:
    from utils import load_config

    cfg = load_config("config.yaml")
    infer_cfg = cfg["lstm"]["inference"]

    model_path = str(infer_cfg["model"])
    norm_path = str(infer_cfg["norm"])
    device_cfg = str(infer_cfg["device"])
    dataset_csv = str(infer_cfg["dataset_csv"])

    checkpoint = torch.load(model_path, map_location="cpu")
    input_len = int(checkpoint["input_len"])
    pred_len = int(checkpoint["pred_len"])
    hidden_size = int(checkpoint["hidden_size"])
    num_layers = int(checkpoint["num_layers"])

    with open(norm_path, "r", encoding="utf-8") as f:
        norm = json.load(f)
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)

    if device_cfg == "auto":
        device_cfg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_cfg)
    model = LSTMForecaster(2, hidden_size, num_layers, pred_len).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    traj: List[Point] = []
    with open(dataset_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("ue_id", 0)) != 0:
                continue
            traj.append((float(row["x_km"]), float(row["y_km"])))

    x_window = np.array(traj[:input_len], dtype=np.float32)
    x_norm = (x_window - mean) / std
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy()[0]

    pred_denorm = pred * std + mean

    print(f"input window shape: {x_window.shape}")
    print(f"predicted shape: {pred_denorm.shape}")
    print("predicted positions (first 3):")
    for row in pred_denorm[:3]:
        print(f"  x={row[0]:.4f}, y={row[1]:.4f}")


if __name__ == "__main__":
    main()
