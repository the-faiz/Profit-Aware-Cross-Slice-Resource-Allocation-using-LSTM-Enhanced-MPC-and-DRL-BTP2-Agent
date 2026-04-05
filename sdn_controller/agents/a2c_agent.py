from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from sdn_controller.utilities.imports import require_torch
from sdn_controller.agents.actor_critic import ActorCritic
from sdn_controller.agents.base_agent import BaseAgent


@dataclass
class A2CConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


class A2CAgent(BaseAgent):
    def __init__(
        self,
        obs_dim: int,
        action_nvec: List[int],
        net_arch: List[int],
        activation: str,
        config: A2CConfig,
        seed: int = 0,
    ) -> None:
        th = require_torch()
        th.manual_seed(int(seed))
        np.random.seed(int(seed))

        self.obs_dim = int(obs_dim)
        self.action_nvec = [int(n) for n in action_nvec]
        self.net_arch = [int(n) for n in net_arch]
        self.activation = activation
        self.config = config

        self.model = ActorCritic(
            obs_dim=self.obs_dim,
            action_nvec=self.action_nvec,
            net_arch=self.net_arch,
            activation=self.activation,
        )
        self.optimizer = th.optim.Adam(
            self.model.parameters(), lr=float(self.config.learning_rate)
        )

    def act(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        th = require_torch()
        obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
        with th.no_grad():
            actions, logprob, _, value = self.model.get_action_and_value(obs_t)
        return (
            actions.squeeze(0).cpu().numpy(),
            logprob.squeeze(0).cpu().numpy(),
            value.squeeze(0).cpu().numpy(),
        )

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        th = require_torch()
        obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
        with th.no_grad():
            logits, _ = self.model.forward(obs_t)
            if deterministic:
                actions = [lg.argmax(dim=-1) for lg in logits]
            else:
                actions = [th.distributions.Categorical(logits=lg).sample() for lg in logits]
            actions_t = th.stack(actions, dim=1)
        return actions_t.squeeze(0).cpu().numpy()

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        last_obs: np.ndarray,
        last_done: bool,
    ) -> None:
        th = require_torch()
        cfg = self.config
        obs_t = th.as_tensor(obs, dtype=th.float32)
        actions_t = th.as_tensor(actions, dtype=th.long)
        rewards_t = th.as_tensor(rewards, dtype=th.float32)
        dones_t = th.as_tensor(dones, dtype=th.float32)

        with th.no_grad():
            last_obs_t = th.as_tensor(last_obs, dtype=th.float32).unsqueeze(0)
            _, last_value = self.model.forward(last_obs_t)
            last_value = last_value.squeeze(0)
            if last_done:
                last_value = th.zeros_like(last_value)
            _, values_t = self.model.forward(obs_t)

        returns = th.zeros_like(rewards_t)
        for t in reversed(range(len(rewards_t))):
            if t == len(rewards_t) - 1:
                next_nonterminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_nonterminal = 1.0 - dones_t[t + 1]
                next_value = values_t[t + 1]
            returns[t] = rewards_t[t] + cfg.gamma * next_value * next_nonterminal

        logprob, entropy, values_pred = self.model.evaluate_actions(obs_t, actions_t)
        advantage = returns - values_pred
        policy_loss = -(advantage.detach() * logprob).mean()
        value_loss = 0.5 * advantage.pow(2).mean()
        entropy_loss = entropy.mean()

        loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

    def save(self, path: str) -> None:
        th = require_torch()
        payload = {
            "state_dict": self.model.state_dict(),
            "obs_dim": self.obs_dim,
            "action_nvec": self.action_nvec,
            "net_arch": self.net_arch,
            "activation": self.activation,
            "config": self.config.__dict__,
        }
        th.save(payload, path)

    @classmethod
    def load(cls, path: str) -> "A2CAgent":
        th = require_torch()
        data = th.load(path, map_location="cpu", weights_only=False)
        cfg = A2CConfig(**data["config"])
        agent = cls(
            obs_dim=int(data["obs_dim"]),
            action_nvec=list(data["action_nvec"]),
            net_arch=list(data["net_arch"]),
            activation=str(data["activation"]),
            config=cfg,
            seed=0,
        )
        agent.model.load_state_dict(data["state_dict"])
        return agent
