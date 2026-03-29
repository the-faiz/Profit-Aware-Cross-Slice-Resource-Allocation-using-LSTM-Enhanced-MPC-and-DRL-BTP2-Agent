# Train LSTM MODEL

from __future__ import annotations

import json
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_config, build_windows, rows_to_trajectories, write_csv
from mobility_pattern_genererator import MobilityPatternGenerator

Point = Tuple[float, float]
Trajectory = List[Point]
UserState = Tuple[int, int, float, float, str]


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
    print("LSTM TRAINING PIPELINE STARTS")
    
    cfg = load_config("config.yaml")
    lstm_cfg = cfg["lstm"]
    training_cfg = cfg["lstm"]["training"]
    seed = int(cfg["main"]["random_seed"])

    np.random.seed(seed)
    torch.manual_seed(seed)

    num_ues = int(training_cfg["num_ues"])
    num_steps = int(training_cfg["num_steps"])
    input_len = int(lstm_cfg["input_len"])
    pred_len = int(lstm_cfg["pred_len"])

    print(f"Starting LSTM Training for {num_ues} users and {num_steps} steps")

    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])
    lr = float(training_cfg["lr"])
    hidden_size = int(training_cfg["hidden_size"])
    num_layers = int(training_cfg["num_layers"])
    model_out = str(training_cfg["model_out"])
    norm_out = str(training_cfg["norm_out"])
    device_cfg = str(training_cfg["device"])

    print("Generating Dataset")
    gen = MobilityPatternGenerator(num_ues, num_steps)
    rows = gen.generate_mobility_pattern()
    write_csv(training_cfg["dataset_csv"], rows)
    print("Dataset Generated Successfully and saved to CSV")

    trajectories = rows_to_trajectories(rows, num_ues)

    X, Y = build_windows(trajectories, input_len, pred_len)
    if X.size == 0:
        raise ValueError("Not enough steps to build windows. Increase num_steps.")

    mean = X.reshape(-1, 2).mean(axis=0)
    std = X.reshape(-1, 2).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    Xn = (X - mean) / std
    Yn = (Y - mean) / std

    dataset = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Yn))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if device_cfg == "auto":
        device_cfg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_cfg)
    model = LSTMForecaster(2, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print("Starting Training");
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"epoch {epoch:03d} | loss {avg_loss:.6f}")
    print("Trainig Ends")

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_len": input_len,
            "pred_len": pred_len,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        },
        model_out,
    )

    with open(norm_out, "w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    print(f"saved model to {model_out}")
    print(f"saved norm stats to {norm_out}")


if __name__ == "__main__":
    main()
