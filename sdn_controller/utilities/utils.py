from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, TYPE_CHECKING

import yaml

BASE_DIR = Path(__file__).resolve().parents[1]

if TYPE_CHECKING:
    from sdn_controller.environment.sdn_env import ModelChoice


@dataclass(frozen=True)
class ModelSpec:
    vnf_id: int
    model_id: int
    accuracy_pct: int
    vcpu_units: int
    vnf_weight: float


# Load YAML configuration file from default or provided path.
def load_config(path: str = "sdn_controller/configurations/config.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = BASE_DIR / "configurations" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Resolve a path relative to the repository root.
def resolve_path(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = BASE_DIR.parent / p
    return str(p)


# Load per-slice VNF catalogs from the configured catalog directory.
def load_slice_catalogs(cfg: dict) -> dict[str, dict[int, List["ModelChoice"]]]:
    from sdn_controller.environment.sdn_env import load_vnf_catalog

    paths_cfg = cfg.get("paths", {})
    vnf_catalog_dir = paths_cfg.get(
        "vnf_catalog_dir",
        cfg.get("environment", {}).get("vnf_catalog_dir", "sdn_controller/data/vnf_catalogs"),
    )
    base_dir = resolve_path(vnf_catalog_dir)
    catalogs: dict[str, dict[int, List["ModelChoice"]]] = {}
    for slice_name in ("slice_1", "slice_2", "slice_3"):
        path = os.path.join(base_dir, f"{slice_name}_vnf_catalog.csv")
        catalogs[slice_name] = load_vnf_catalog(path)
    return catalogs


# Build a list of model specs for a single VNF in the catalog generator.
def generate_models_for_vnf(
    vnf_id: int,
    num_models: int,
    acc_min: int,
    acc_max: int,
    vcpu_min: int,
    vcpu_max: int,
    vnf_weight: float,
    rng: random.Random,
) -> List[ModelSpec]:
    models: List[ModelSpec] = []
    for model_id in range(1, num_models + 1):
        accuracy = rng.randint(acc_min, acc_max)
        vcpu = rng.randint(vcpu_min, vcpu_max)
        models.append(
            ModelSpec(
                vnf_id=vnf_id,
                model_id=model_id,
                accuracy_pct=accuracy,
                vcpu_units=vcpu,
                vnf_weight=vnf_weight,
            )
        )
    return models


# Write VNF catalog rows to a CSV file on disk.
def write_catalog_csv(path: str, models: Iterable[ModelSpec]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["vnf_id", "model_id", "accuracy_pct", "vcpu_units", "vnf_weight"]
        )
        for m in models:
            writer.writerow(
                [m.vnf_id, m.model_id, m.accuracy_pct, m.vcpu_units, f"{m.vnf_weight:.4f}"]
            )


# Load a VNF catalog CSV into a vnf_id -> model list mapping.
def load_vnf_catalog(path: str | Path) -> Dict[int, List["ModelChoice"]]:
    from sdn_controller.environment.sdn_env import ModelChoice

    catalog: Dict[int, List[ModelChoice]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vnf_id = int(row["vnf_id"])
            model = ModelChoice(
                vnf_id=vnf_id,
                model_id=int(row["model_id"]),
                accuracy_pct=float(row["accuracy_pct"]),
                vcpu_units=float(row["vcpu_units"]),
                vnf_weight=float(row["vnf_weight"]),
            )
            catalog.setdefault(vnf_id, []).append(model)
    return catalog
