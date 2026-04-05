from __future__ import annotations

from typing import List

from sdn_controller.utilities.imports import require_torch


class ActorCritic(require_torch().nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_nvec: List[int],
        net_arch: List[int],
        activation: str,
    ) -> None:
        th = require_torch()
        nn = th.nn
        super().__init__()
        act_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }
        act_cls = act_map.get(activation, nn.ReLU)

        layers: List[nn.Module] = []
        last_dim = obs_dim
        for h in net_arch:
            layers.append(nn.Linear(last_dim, int(h)))
            layers.append(act_cls())
            last_dim = int(h)
        self.base = nn.Sequential(*layers) if layers else nn.Identity()

        self.actor_heads = nn.ModuleList(
            [nn.Linear(last_dim, int(n)) for n in action_nvec]
        )
        self.critic = nn.Linear(last_dim, 1)

    def forward(self, obs):
        th = require_torch()
        x = self.base(obs)
        logits = [head(x) for head in self.actor_heads]
        value = self.critic(x).squeeze(-1)
        return logits, value

    def get_action_and_value(self, obs):
        th = require_torch()
        logits, value = self.forward(obs)
        actions = []
        logprob = th.zeros(obs.shape[0], device=obs.device)
        entropy = th.zeros(obs.shape[0], device=obs.device)
        for lg in logits:
            dist = th.distributions.Categorical(logits=lg)
            act = dist.sample()
            actions.append(act)
            logprob = logprob + dist.log_prob(act)
            entropy = entropy + dist.entropy()
        actions_t = th.stack(actions, dim=1)
        return actions_t, logprob, entropy, value

    def evaluate_actions(self, obs, actions):
        th = require_torch()
        logits, value = self.forward(obs)
        logprob = th.zeros(obs.shape[0], device=obs.device)
        entropy = th.zeros(obs.shape[0], device=obs.device)
        for i, lg in enumerate(logits):
            dist = th.distributions.Categorical(logits=lg)
            act = actions[:, i]
            logprob = logprob + dist.log_prob(act)
            entropy = entropy + dist.entropy()
        return logprob, entropy, value
