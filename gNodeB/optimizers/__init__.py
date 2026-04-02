from __future__ import annotations

from typing import Dict

from .average import AverageOptimizer
from .deficit_aware import DeficitAwareOptimizer
from .ga import GAOptimizer
from .greedy import GreedyOptimizer
from .hybrid_avg_deficit import HybridAverageDeficitOptimizer
from .pso import PSOOptimizer
from .random_alloc import RandomOptimizer
from .static import StaticOptimizer
from .tier_quota import TierQuotaOptimizer
from .topk_priority import TopKPriorityOptimizer
from .target_rate import TargetRateProportionalOptimizer


def make_optimizer(name: str, cfg: Dict) -> object:
    name = name.lower()

    sim_cfg = cfg["simulation"]
    tiers_cfg = cfg["tiers"]
    prb_cfg = cfg["PRBS"]
    ru_cfg = cfg["ru"]
    reward_cfg = cfg.get("reward", {})

    penalty_lambda = float(cfg["lambda"])

    horizon = int(cfg.get("_horizon", min(sim_cfg["prediction_horizon"], sim_cfg["control_horizon"])))
    common = dict(
        num_prbs=int(prb_cfg["num_prbs"]),
        prb_cost=float(prb_cfg["cost"]),
        prb_bw=float(prb_cfg["bandwidth_hz"]),
        gamma=float(sim_cfg["discount_gamma"]),
        horizon=horizon,
        ru_x=float(ru_cfg["ru_x_km"]),
        ru_y=float(ru_cfg["ru_y_km"]),
        tiers_cfg=tiers_cfg,
        penalty_lambda=penalty_lambda,
        profit_sensitivity=float(reward_cfg.get("profit_sensitivity", 1.0)),
        satisfaction_sensitivity=float(reward_cfg.get("satisfaction_sensitivity", 1.0)),
    )

    channel = cfg.get("_channel")
    if channel is None:
        raise ValueError("Channel instance missing from config (_channel).")

    opt_cfg = cfg.get("optimizer", {})
    if name in {"ga", "genetic"}:
        return GAOptimizer(channel=channel, ga_config=opt_cfg.get("ga", {}), **common)
    if name in {"pso", "particle"}:
        return PSOOptimizer(channel=channel, pso_config=opt_cfg.get("pso", {}), **common)
    if name in {"greedy"}:
        return GreedyOptimizer(channel=channel, **common)
    if name in {"static", "fixed"}:
        return StaticOptimizer(num_prbs=common["num_prbs"])
    if name in {"average", "avg"}:
        return AverageOptimizer(num_prbs=common["num_prbs"], tiers_cfg=tiers_cfg)
    if name in {"hybrid_avg_deficit", "hybrid", "avg_deficit"}:
        return HybridAverageDeficitOptimizer(
            num_prbs=common["num_prbs"],
            prb_bw=common["prb_bw"],
            ru_x=common["ru_x"],
            ru_y=common["ru_y"],
            channel=channel,
            tiers_cfg=tiers_cfg,
        )
    if name in {"deficit_aware", "deficit-aware", "deficit"}:
        return DeficitAwareOptimizer(
            num_prbs=common["num_prbs"],
            prb_bw=common["prb_bw"],
            ru_x=common["ru_x"],
            ru_y=common["ru_y"],
            channel=channel,
            tiers_cfg=tiers_cfg,
        )
    if name in {"tier_quota", "quota", "tier-quota"}:
        return TierQuotaOptimizer(num_prbs=common["num_prbs"], tiers_cfg=tiers_cfg)
    if name in {"topk", "topk_priority", "top-k"}:
        return TopKPriorityOptimizer(
            num_prbs=common["num_prbs"],
            ru_x=common["ru_x"],
            ru_y=common["ru_y"],
            channel=channel,
        )
    if name in {"target_rate", "target-rate", "target"}:
        return TargetRateProportionalOptimizer(
            num_prbs=common["num_prbs"],
            prb_bw=common["prb_bw"],
            ru_x=common["ru_x"],
            ru_y=common["ru_y"],
            channel=channel,
            tiers_cfg=tiers_cfg,
        )
    if name in {"random", "rand"}:
        return RandomOptimizer(num_prbs=common["num_prbs"])
    raise ValueError(
        "Unknown optimizer '{name}'. Choose from: ga, pso, greedy, static, average, hybrid_avg_deficit, deficit_aware, tier_quota, topk_priority, target_rate, random."
    )
