# Helper utilities to build warm-start allocations from fast heuristics.
from __future__ import annotations

from typing import Dict, List

import numpy as np

from channel.channel_model import Channel
from .average import AverageOptimizer
from .static import StaticOptimizer
from .random_alloc import RandomOptimizer
from .tier_quota import TierQuotaOptimizer
from .deficit_aware import DeficitAwareOptimizer
from .topk_priority import TopKPriorityOptimizer
from .target_rate import TargetRateProportionalOptimizer
from .hybrid_avg_deficit import HybridAverageDeficitOptimizer


def warm_start_allocations(
    pred_positions: np.ndarray,
    user_tiers: List[str],
    *,
    num_prbs: int,
    prb_bw: float,
    ru_x: float,
    ru_y: float,
    channel: Channel,
    tiers_cfg: Dict[str, dict],
) -> List[List[int]]:
    """
    Return a list of allocations from fast heuristics (excluding greedy).
    Allocations are for h=0 only (length = num_ues).
    """
    heuristics = [
        AverageOptimizer(num_prbs=num_prbs, tiers_cfg=tiers_cfg),
        StaticOptimizer(num_prbs=num_prbs),
        RandomOptimizer(num_prbs=num_prbs),
        TierQuotaOptimizer(num_prbs=num_prbs, tiers_cfg=tiers_cfg),
        DeficitAwareOptimizer(
            num_prbs=num_prbs,
            prb_bw=prb_bw,
            ru_x=ru_x,
            ru_y=ru_y,
            channel=channel,
            tiers_cfg=tiers_cfg,
        ),
        TopKPriorityOptimizer(num_prbs=num_prbs, ru_x=ru_x, ru_y=ru_y, channel=channel),
        TargetRateProportionalOptimizer(
            num_prbs=num_prbs,
            prb_bw=prb_bw,
            ru_x=ru_x,
            ru_y=ru_y,
            channel=channel,
            tiers_cfg=tiers_cfg,
        ),
        HybridAverageDeficitOptimizer(
            num_prbs=num_prbs,
            prb_bw=prb_bw,
            ru_x=ru_x,
            ru_y=ru_y,
            channel=channel,
            tiers_cfg=tiers_cfg,
        ),
    ]

    allocations: List[List[int]] = []
    seen = set()
    for h in heuristics:
        alloc = h.solve(pred_positions, user_tiers)
        key = tuple(alloc)
        if key not in seen:
            allocations.append(alloc)
            seen.add(key)

    return allocations
