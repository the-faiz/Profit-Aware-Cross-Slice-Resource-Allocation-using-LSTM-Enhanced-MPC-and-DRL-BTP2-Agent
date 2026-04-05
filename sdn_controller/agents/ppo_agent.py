from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from sdn_controller.utilities.imports import require_torch
from sdn_controller.agents.actor_critic import ActorCritic
from sdn_controller.agents.base_agent import BaseAgent


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 64


class PPOAgent(BaseAgent):
    def __init__(
        self,
        obs_dim: int,
        action_nvec: List[int],
        net_arch: List[int],
        activation: str,
        config: PPOConfig,
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
        logprobs_t = th.as_tensor(logprobs, dtype=th.float32)
        values_t = th.as_tensor(values, dtype=th.float32)
        rewards_t = th.as_tensor(rewards, dtype=th.float32)
        dones_t = th.as_tensor(dones, dtype=th.float32)

        with th.no_grad():
            last_obs_t = th.as_tensor(last_obs, dtype=th.float32).unsqueeze(0)
            _, last_value = self.model.forward(last_obs_t)
            last_value = last_value.squeeze(0)
            if last_done:
                last_value = th.zeros_like(last_value)

        advantages = th.zeros_like(rewards_t)
        last_gae = 0.0
        for t in reversed(range(len(rewards_t))):
            if t == len(rewards_t) - 1:
                next_nonterminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_nonterminal = 1.0 - dones_t[t + 1]
                next_value = values_t[t + 1]
            delta = rewards_t[t] + cfg.gamma * next_value * next_nonterminal - values_t[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_steps = obs_t.shape[0]
        batch_size = min(int(cfg.batch_size), n_steps)
        idxs = np.arange(n_steps)

        for _ in range(int(cfg.n_epochs)):
            np.random.shuffle(idxs)
            for start in range(0, n_steps, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]
                mb_obs = obs_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_logprobs = logprobs_t[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                new_logprobs, entropy, values_pred = self.model.evaluate_actions(
                    mb_obs, mb_actions
                )
                ratio = (new_logprobs - mb_logprobs).exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * th.clamp(
                    ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range
                )
                policy_loss = th.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * (mb_returns - values_pred).pow(2).mean()
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
    def load(cls, path: str) -> "PPOAgent":
        th = require_torch()
        data = th.load(path, map_location="cpu", weights_only=False)
        cfg = PPOConfig(**data["config"])
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
