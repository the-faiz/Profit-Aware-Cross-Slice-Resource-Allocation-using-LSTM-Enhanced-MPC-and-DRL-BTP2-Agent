from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from sdn_controller.utilities.utils import load_config, load_vnf_catalog
from sdn_controller.utilities.reward import (
    compute_reward,
    weighted_model_accuracy,
    allocated_data_rate_mbps,
)


@dataclass(frozen=True)
class User:
    user_id: int
    user_type: str
    snir: float


@dataclass(frozen=True)
class ModelChoice:
    vnf_id: int
    model_id: int
    accuracy_pct: float
    vcpu_units: float
    vnf_weight: float

