from __future__ import annotations

from typing import List, Sequence, Tuple
import random

from utilities.utils import load_config

Point = Tuple[float, float]
Trajectory = List[Point]
UserState = Tuple[int, int, float, float, str]


class MobilityPatternGenerator:
    def __init__(self, num_ues: int, num_steps: int):
        cfg = load_config()
        main_cfg = cfg["main"]
        env_cfg = cfg["environment"]
        mob_cfg = cfg["mobility_pattern"]
        ru_cfg = cfg["ru"]

        self.cfg = {
            "seed": main_cfg["random_seed"],
            "area_km": env_cfg["area_km"],
            "std_s": mob_cfg["std_s"],
            "clip_to_area": mob_cfg["clip_to_area"],
            "ru_x_km": ru_cfg["ru_x_km"],
            "ru_y_km": ru_cfg["ru_y_km"],
            "num_ues": int(num_ues),
            "num_steps": int(num_steps),
        }
        self._rng = random.Random(self.cfg["seed"])
        self.tiers_cfg = cfg["tiers"]
        self.user_tiers = self._assign_tiers()

    def _assign_tiers(self) -> List[str]:
        names = []
        probs = []
        for name, data in self.tiers_cfg.items():
            names.append(str(name))
            probs.append(float(data["probability"]))
        total = sum(probs)
        probs = [p / total for p in probs]
        return self._rng.choices(names, weights=probs, k=self.cfg["num_ues"])

    def get_user_tier(self, ue_id: int) -> str:
        return self.user_tiers[ue_id]

    def _clip(self, x: float) -> float:
        if not self.cfg["clip_to_area"]:
            return x
        half = self.cfg["area_km"] / 2.0
        return max(-half, min(half, x))

    def _gauss(self, mu: float, sigma: float) -> float:
        return self._rng.gauss(mu, sigma)

    def ru_positions(self) -> List[Point]:
        """Single RU at configured coordinates."""
        return [(self.cfg["ru_x_km"], self.cfg["ru_y_km"])]

    def initial_positions(self) -> List[Point]:
        """Draw initial UE positions from N(0, s^2)."""
        positions = []
        for _ in range(self.cfg["num_ues"]):
            x = self._gauss(0.0, self.cfg["std_s"])
            y = self._gauss(0.0, self.cfg["std_s"])
            positions.append((self._clip(x), self._clip(y)))
        return positions

    def step_positions(self, positions: Sequence[Point]) -> List[Point]:
        """Advance UE positions by one step with Gaussian displacement."""
        sigma_step = self.cfg["std_s"] / 10.0  # sqrt(s^2/100)
        next_positions = []
        for (x, y) in positions:
            dx = self._gauss(0.0, sigma_step)
            dy = self._gauss(0.0, sigma_step)
            nx = self._clip(x + dx)
            ny = self._clip(y + dy)
            next_positions.append((nx, ny))
        return next_positions

    def generate_trajectories(self) -> List[Trajectory]:
        """Generate full trajectories for all UEs.

        Returns a list of trajectories, one per UE, each containing num_steps points.
        """
        positions = self.initial_positions()
        trajectories = [[] for _ in range(self.cfg["num_ues"])]
        for _t in range(self.cfg["num_steps"]):
            for i, p in enumerate(positions):
                trajectories[i].append(p)
            positions = self.step_positions(positions)
        return trajectories

    def generate_mobility_pattern(self) -> List[UserState]:
        """Build rows with tier: (t, ue_id, x_km, y_km, tier)."""
        trajectories = self.generate_trajectories()
        rows: List[UserState] = []
        num_steps = len(trajectories[0])
        for t in range(num_steps):
            for ue_id, traj in enumerate(trajectories):
                x, y = traj[t]
                tier = self.get_user_tier(ue_id)
                rows.append((t, ue_id, float(x), float(y), tier))
        return rows

class MobilitySimulation:
    """Streaming simulator using the same mobility model."""

    def __init__(self, num_ues: int, num_steps: int):
        self._gen = MobilityPatternGenerator(num_ues, num_steps)
        self.cfg = self._gen.cfg
        self.t = 0
        self.positions = self._gen.initial_positions()

    def step(self) -> List[Point]:
        """Advance one step and return current UE positions."""
        current = list(self.positions)
        self.positions = self._gen.step_positions(self.positions)
        self.t += 1
        return current


__all__ = [
    "MobilityPatternGenerator",
    "MobilitySimulation",
]
