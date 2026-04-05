from __future__ import annotations

import numpy as np

from sdn_controller.agents.base_agent import BaseAgent
from sdn_controller.utilities.baseline_agent_utils import (
    pack_action,
    select_models,
    select_prbs,
    select_slices,
    zero_action,
)
from sdn_controller.environment.env_gym import SDNEnv


class EqualShareMaxAccAgent(BaseAgent):
    def __init__(self, env: SDNEnv, seed: int = 0) -> None:
        self.env = env
        self.rng = np.random.default_rng(int(seed))

    def act(self, obs):
        action = self.predict(obs, deterministic=True)
        return action, None, None

    def predict(self, obs, deterministic: bool = True):
        users = self.env.current_users
        if not users:
            return zero_action(self.env)

        slice_idxs = select_slices("tier", users, self.rng)
        model_indices = select_models(self.env, slice_idxs, "max_acc", self.rng)
        prbs = select_prbs(
            self.env, users, slice_idxs, model_indices, "equal_share", self.rng
        )
        return pack_action(slice_idxs, prbs, model_indices)

    def update(self, *args, **kwargs) -> None:
        return None

    def save(self, path: str) -> None:
        raise NotImplementedError("EqualShareMaxAccAgent does not support saving.")

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError("EqualShareMaxAccAgent does not support loading.")
