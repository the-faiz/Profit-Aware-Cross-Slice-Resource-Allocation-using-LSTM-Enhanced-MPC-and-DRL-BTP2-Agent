from __future__ import annotations

import sys
from pathlib import Path as _Path

ROOT_DIR = _Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import random
from typing import List


from sdn_controller.utilities.utils import (
    ModelSpec,
    generate_models_for_vnf,
    load_config,
    resolve_path,
    write_catalog_csv,
)

SLICE_NAMES = ("slice_1", "slice_2", "slice_3")


def generate_vnf_catalog(cfg: dict) -> dict[str, List[ModelSpec]]:
    gen_cfg = cfg["generators"]["vnf_catalog"]
    rng = random.Random(int(gen_cfg["seed"]))
    catalogs: dict[str, List[ModelSpec]] = {}

    for slice_name in SLICE_NAMES:
        slice_models: List[ModelSpec] = []
        slice_vcpu_cfg = gen_cfg.get("vcpu_units_by_slice", {}).get(slice_name, {})
        vcpu_min = int(slice_vcpu_cfg.get("min", gen_cfg["vcpu_min_units"]))
        vcpu_max = int(slice_vcpu_cfg.get("max", gen_cfg["vcpu_max_units"]))
        for vnf_id in range(1, int(gen_cfg["vnf_count"]) + 1):
            vnf_weight = rng.uniform(
                float(gen_cfg["vnf_weight_min"]),
                float(gen_cfg["vnf_weight_max"]),
            )
            slice_models.extend(
                generate_models_for_vnf(
                    vnf_id=vnf_id,
                    num_models=int(gen_cfg["models_per_vnf"]),
                    acc_min=int(gen_cfg["accuracy_min_pct"]),
                    acc_max=int(gen_cfg["accuracy_max_pct"]),
                    vcpu_min=vcpu_min,
                    vcpu_max=vcpu_max,
                    vnf_weight=vnf_weight,
                    rng=rng,
                )
            )
        catalogs[slice_name] = slice_models
    return catalogs


def write_catalogs() -> None:
    cfg = load_config()
    paths_cfg = cfg.get("paths", {})
    out_dir = resolve_path(
        paths_cfg.get(
            "vnf_catalog_dir",
            cfg.get("generators", {}).get("vnf_catalog", {}).get("out_dir", "sdn_controller/data/vnf_catalogs"),
        )
    )
    catalogs = generate_vnf_catalog(cfg)
    for slice_name, models in catalogs.items():
        out_path = os.path.join(out_dir, f"{slice_name}_vnf_catalog.csv")
        write_catalog_csv(out_path, models)
    print(f"Saved catalogs to {out_dir}")


if __name__ == "__main__":
    write_catalogs()
