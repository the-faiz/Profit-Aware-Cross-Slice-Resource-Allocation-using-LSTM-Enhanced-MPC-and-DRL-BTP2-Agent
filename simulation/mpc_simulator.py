# MPC Simulator
#
# Two operating modes:
#
#   Single-step mode (horizon=1, no LSTM):
#     At each time t, observe actual UE positions, solve the GA for h=0 only,
#     evaluate reward on actual positions. This is the baseline — no prediction.
#
#   MPC mode (horizon>1, with LSTM):
#     At each time t, use LSTM to predict positions at t+1 ... t+H-1.
#     Stack [actual_t, pred_t+1, ..., pred_t+H-1] as the horizon.
#     Solve the GA over the full horizon, apply only h=0 allocation.
#     Evaluate reward on actual positions at t.
#
# The reward formula exactly matches the problem statement:
#
#   data_rate[u]     = n[u] * prb_bw * log2(1 + sinr[u])    (actual positions)
#   satisfaction[u]  = min(1, data_rate[u] / target[tier[u]])
#   profit[u]        = fee[tier[u]] - n[u] * prb_cost
#   reward_base      = sum_u (profit[u] + satisfaction[u])
#   deficit[tier]    = max(0, min_sat[tier]*count[tier] - #satisfied_in_tier)
#   penalty          = lambda * sum_tier deficit[tier]
#   reward           = reward_base + penalty

from __future__ import annotations

import csv
import json
import os
import time
from typing import List, Tuple

import numpy as np
import torch

from channel.channel_model import Channel
from lstm.lstm_model import LSTMModel
from optimizers import make_optimizer
from utilities.reward import compute_reward
from utilities.utils import load_config

Point = Tuple[float, float]


class MPCSimulator:
    def __init__(
        self,
        config_path: str = "configurations/config.yaml",
        optimizer_name: str = "ga",
        num_ues_override: int | None = None,
        log_file=None,
    ) -> None:
        cfg = load_config(config_path)

        self.sim_cfg   = cfg["simulation"]
        self.lstm_cfg  = cfg["lstm"]
        self.train_cfg = cfg["lstm"]["training"]
        self.ru_cfg    = cfg["ru"]
        self.tiers_cfg = cfg["tiers"]
        self.prb_cfg   = cfg["PRBS"]
        self.reward_cfg = cfg.get("reward", {})
        self.log_cfg = cfg.get("logging", {})

        # lambda is negative in config; store as-is for reward calc,
        # and pass as-is to the optimizer (it appears directly in objective).
        self.penalty_lambda = float(cfg["lambda"])

        if num_ues_override is None:
            raise ValueError(
                "num_ues_override must be provided when running batch simulations."
            )
        self.num_ues   = int(num_ues_override)
        self.num_steps = int(self.sim_cfg["num_steps"])
        self.num_prbs  = int(self.prb_cfg["num_prbs"])
        self.prb_bw    = float(self.prb_cfg["bandwidth_hz"])
        self.prb_cost  = float(self.prb_cfg["cost"])
        self.profit_sensitivity = float(self.reward_cfg.get("profit_sensitivity", 1.0))
        self.satisfaction_sensitivity = float(
            self.reward_cfg.get("satisfaction_sensitivity", 1.0)
        )
        self.pred_h    = int(self.sim_cfg["prediction_horizon"])
        self.ctrl_h    = int(self.sim_cfg["control_horizon"])
        self.gamma     = float(self.sim_cfg["discount_gamma"])
        self.input_len = int(self.lstm_cfg["input_len"])
        self.dataset_csv = str(self.sim_cfg["dataset_csv"])
        self.log_enabled = bool(self.log_cfg.get("enabled", True))
        self.log_path = str(self.log_cfg.get("path", "logs/simulation.log"))
        self.log_append = bool(self.log_cfg.get("append", False))
        self._external_log_file = log_file

        # horizon = min of pred horizon, control horizon, and LSTM output length
        self.model, self.mean, self.std, self.device = self._load_model()
        self.horizon = min(self.pred_h, self.ctrl_h, self._pred_len)

        self.positions, self.user_tiers = self._load_positions()
        self.ru_x    = float(self.ru_cfg["ru_x_km"])
        self.ru_y    = float(self.ru_cfg["ru_y_km"])
        self.channel = Channel()

        cfg["_channel"] = self.channel
        cfg["_horizon"] = self.horizon
        self.optimizer = make_optimizer(optimizer_name, cfg)

    # ── Loaders ───────────────────────────────────────────────────────

    def _load_model(self) -> tuple[LSTMModel, np.ndarray, np.ndarray, torch.device]:
        checkpoint = torch.load(
            self.train_cfg["model_out"], map_location="cpu", weights_only=True
        )
        self._pred_len = int(checkpoint["pred_len"])
        with open(self.train_cfg["norm_out"], "r", encoding="utf-8") as f:
            norm = json.load(f)
        mean = np.array(norm["mean"], dtype=np.float32)
        std  = np.array(norm["std"],  dtype=np.float32)

        device_cfg = str(self.train_cfg["device"])
        if device_cfg == "auto":
            device_cfg = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_cfg)

        model = LSTMModel(
            2, int(checkpoint["hidden_size"]),
            int(checkpoint["num_layers"]), self._pred_len
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model, mean, std, device

    def _load_positions(self) -> tuple[np.ndarray, List[str]]:
        positions  = np.zeros((self.num_steps, self.num_ues, 2), dtype=np.float32)
        user_tiers = [""] * self.num_ues
        with open(self.dataset_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t, u = int(row["t"]), int(row["ue_id"])
                if t >= self.num_steps or u >= self.num_ues:
                    raise ValueError(
                        f"CSV row (t={t}, ue_id={u}) out of bounds — "
                        "re-generate the dataset or update configurations/config.yaml."
                    )
                positions[t, u, 0] = float(row["x_km"])
                positions[t, u, 1] = float(row["y_km"])
                if not user_tiers[u]:
                    user_tiers[u] = row["tier"]
        return positions, user_tiers

    # ── Prediction ────────────────────────────────────────────────────

    def _predict_future(self, history: np.ndarray) -> np.ndarray:
        """
        Given shape-(input_len, 2) position history, return
        shape-(pred_len, 2) predicted future positions.
        """
        x_norm   = (history - self.mean) / self.std
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x_tensor).cpu().numpy()[0]   # (pred_len, 2)
        return pred * self.std + self.mean

    def _build_horizon_positions(self, t: int) -> np.ndarray:
        """
        Build pred_positions array of shape (U, H, 2).

        h=0   : actual position at time t  (ground truth, no prediction error)
        h=1..H-1 : LSTM-predicted positions at t+1 .. t+H-1
        """
        horizon_pos = np.zeros((self.num_ues, self.horizon, 2), dtype=np.float32)
        for u in range(self.num_ues):
            horizon_pos[u, 0, :] = self.positions[t, u, :]
            if self.horizon > 1:
                history = self.positions[t - self.input_len + 1 : t + 1, u, :]
                preds   = self._predict_future(history)          # (pred_len, 2)
                steps   = min(self.horizon - 1, preds.shape[0])
                horizon_pos[u, 1:steps + 1, :] = preds[:steps]
        return horizon_pos

    # ── Reward evaluation (always on actual positions) ─────────────────

    def _reward_for_step(self, t: int, alloc: List[int]) -> dict:
        return compute_reward(
            positions_t=self.positions[t],
            alloc_counts=alloc,
            user_tiers=self.user_tiers,
            tiers_cfg=self.tiers_cfg,
            ru_x=self.ru_x,
            ru_y=self.ru_y,
            prb_bw=self.prb_bw,
            prb_cost=self.prb_cost,
            penalty_lambda=self.penalty_lambda,
            profit_sensitivity=self.profit_sensitivity,
            satisfaction_sensitivity=self.satisfaction_sensitivity,
            channel=self.channel,
        )

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self) -> None:
        # Need input_len steps of history before we can predict,
        # so simulation starts at t = input_len - 1 (0-indexed).
        start_t = self.input_len - 1
        rewards: List[float] = []
        t0 = time.time()

        header_line = (
            f"{'t':>4}  {'reward':>10}  {'base':>10}  "
            f"{'penalty':>10}  {'profit':>10}  {'prbs_used':>9}  "
            f"{'tier-1 sat':>10}  {'tier-2 sat':>10}  {'tier-3 sat':>10}"
        )
        sep_line = "-" * 100

        sim_line = (
            f"Simulating {self.num_ues} UEs | "
            f"{self.num_steps - start_t} steps | "
            f"horizon={self.horizon} | PRBs={self.num_prbs}"
        )
        print(sim_line)

        log_f = self._external_log_file
        if log_f is None and self.log_enabled:
            log_dir = os.path.dirname(self.log_path) or "."
            os.makedirs(log_dir, exist_ok=True)
            mode = "a" if self.log_append else "w"
            log_f = open(self.log_path, mode, encoding="utf-8")
        if log_f is not None:
            log_f.write(sim_line + "\n")
            log_f.write(header_line + "\n")
            log_f.write(sep_line + "\n")

        profit_list: List[float] = []
        satisfied_counts: List[int] = []

        for t in range(start_t, self.num_steps):
            # Build horizon: actual position at h=0, LSTM predictions for h>0
            pred_positions = self._build_horizon_positions(t)

            # Solve the GA optimizer — returns h=0 allocations
            alloc = self.optimizer.solve(pred_positions, self.user_tiers)

            # Evaluate reward on actual ground-truth positions
            info = self._reward_for_step(t, alloc)
            rewards.append(info["reward"])
            profit_list.append(info["profit"])

            tier_sat = info["tier_sat"]
            tier_stats = info["tier_stats"]
            satisfied_total = sum(tier_sat.values())
            satisfied_counts.append(int(satisfied_total))
            if log_f is not None:
                log_f.write(
                    f"{t:>4}  {info['reward']:>10.2f}  {info['reward_base']:>10.2f}  "
                    f"{info['penalty']:>10.2f}  {info['profit']:>10.2f}  {info['total_prbs']:>9d}  "
                    f"{tier_sat.get('tier-1', 0)}/{tier_stats.get('tier-1', 0):<8d}  "
                    f"{tier_sat.get('tier-2', 0)}/{tier_stats.get('tier-2', 0):<8d}  "
                    f"{tier_sat.get('tier-3', 0)}/{tier_stats.get('tier-3', 0):<8d}\n"
                )

        elapsed = time.time() - t0
        avg_r   = float(np.mean(rewards)) if rewards else 0.0
        avg_profit = float(np.mean(profit_list)) if profit_list else 0.0
        avg_satisfied = float(np.mean(satisfied_counts)) if satisfied_counts else 0.0
        avg_satisfied_ratio = avg_satisfied / max(self.num_ues, 1)
        print(f"Average reward : {avg_r:.4f}")
        print(f"Average profit : {avg_profit:.4f}")
        print(f"Average satisfied users : {avg_satisfied:.2f}")
        print(f"Total time     : {elapsed:.1f}s  "
              f"({elapsed / max(len(rewards), 1):.3f}s per step)")
        print(sep_line)
        if log_f is not None:
            log_f.write(f"Average reward : {avg_r:.4f}\n")
            log_f.write(f"Average profit : {avg_profit:.4f}\n")
            log_f.write(f"Average satisfied users : {avg_satisfied:.2f}\n")
            log_f.write(
                f"Total time     : {elapsed:.1f}s  "
                f"({elapsed / max(len(rewards), 1):.3f}s per step)\n"
            )
            log_f.write(sep_line + "\n")
            if self._external_log_file is None:
                log_f.close()

        return {
            "avg_reward": avg_r,
            "avg_profit": avg_profit,
            "avg_satisfied_users": avg_satisfied,
            "avg_satisfied_ratio": avg_satisfied_ratio,
        }
