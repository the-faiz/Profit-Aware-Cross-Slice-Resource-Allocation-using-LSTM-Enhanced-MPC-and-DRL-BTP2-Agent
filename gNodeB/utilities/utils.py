# Utility Helper Functions 

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]


def get_base_dir() -> Path:
    return BASE_DIR


def resolve_path(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR / p
    return str(p)


def load_config(path: str = "configurations/config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg.get("simulation", {})
    train_cfg = cfg.get("lstm", {}).get("training", {})
    log_cfg = cfg.get("logging", {})

    for key in ("model_out", "norm_out", "dataset_csv"):
        if isinstance(train_cfg.get(key), str):
            train_cfg[key] = resolve_path(train_cfg[key])

    if isinstance(sim_cfg.get("dataset_csv"), str):
        sim_cfg["dataset_csv"] = resolve_path(sim_cfg["dataset_csv"])

    if isinstance(log_cfg.get("path"), str):
        log_cfg["path"] = resolve_path(log_cfg["path"])

    return cfg


def write_csv(
    path: str,
    rows: Iterable[Sequence],
) -> None:
    """Write rows to CSV with header inferred by row width."""
    with open(path, "w", encoding="utf-8") as f:
        rows_list = list(rows)
        if not rows_list:
            f.write("t,ue_id,x_km,y_km,tier\n")
            return
        f.write("t,ue_id,x_km,y_km,tier\n")
        for t, ue_id, x, y, tier in rows_list:
            f.write(f"{t},{ue_id},{x:.6f},{y:.6f},{tier}\n")


def build_windows(
    trajectories: Sequence[Sequence[tuple[float, float]]],
    input_len: int,
    pred_len: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Create sliding windows for LSTM training."""
    import numpy as np

    X: list[list[tuple[float, float]]] = []
    Y: list[list[tuple[float, float]]] = []
    if not trajectories:
        return np.empty((0, input_len, 2)), np.empty((0, pred_len, 2))
    for traj in trajectories:
        max_start = len(traj) - input_len - pred_len + 1
        for start in range(max_start):
            x_seq = traj[start : start + input_len]
            y_seq = traj[start + input_len : start + input_len + pred_len]
            X.append(x_seq)
            Y.append(y_seq)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def rows_to_trajectories(
    rows: Sequence[Sequence],
    num_ues: int,
) -> list[list[tuple[float, float]]]:
    trajectories: list[list[tuple[float, float]]] = [[] for _ in range(num_ues)]
    for _t, ue_id, x, y, _tier in rows:
        trajectories[int(ue_id)].append((float(x), float(y)))
    return trajectories
