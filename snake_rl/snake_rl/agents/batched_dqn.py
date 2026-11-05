"""支持多环境并行批推理与 AMP (Bfloat16) 混合精度的高性能 BatchedDQNAgent。

核心优化：
1. 向量化动作决策：单次调用处理 N 个环境的状态，完全消除单样本前向传播的 Kernel Launch 延迟；
2. 批量 Epsilon-Greedy：通过 GPU 张量掩码一次性完成 N 个环境的探索/利用决策；
3. 原生 Bfloat16 / AMP 加速：在 Ada / Ampere 架构 GPU (如 L4) 上吞吐翻倍，数值稳定且无溢出风险；
4. 零拷贝交互：直接与 TensorReplayBuffer 的 BatchedTransition 对接。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_rl.agents.tensor_replay_buffer import BatchedTransition
from snake_rl.config import DQNConfig
from snake_rl.models.board_qnet import BoardQNet
from snake_rl.models.qnet import QNet


class BatchedDQNAgent:
    """高性能批处理 DQN 智能体。"""

    def __init__(
        self,
        config: DQNConfig,
        device: torch.device | str = "auto",
        use_amp: bool = True,
        grad_clip: float | None = 10.0,
    ):
        self.config = config
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 仅在 CUDA 且支持 Bfloat16 时启用 AMP
        self.use_amp = use_amp and self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.grad_clip = grad_clip

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.criterion = nn.SmoothL1Loss() if config.huber else nn.MSELoss()

        self.updates = 0
        self.target_syncs = 0

    def _build_model(self) -> nn.Module:
        if self.config.obs_type == "board":
            return BoardQNet(
                in_channels=self.config.board_channels,
                height=self.config.board_height,
                width=self.config.board_width,
                hidden_size=self.config.hidden_size,
                output_size=self.config.n_actions,
                adaptive_pool_size=self.config.adaptive_pool_size,
            ).to(self.device)
        return QNet(
            input_size=self.config.obs_size,
            hidden_size=self.config.hidden_size,
            output_size=self.config.n_actions,
        ).to(self.device)

    def select_actions(
        self,
        batched_obs: np.ndarray | torch.Tensor,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """为 N 个环境同时做出动作决策。

        参数:
            batched_obs: (N, C, H, W) 或 (N, 11)
            epsilon: 当前探索率

        返回:
            actions: (N,) int64 numpy 数组
        """
        if not isinstance(batched_obs, torch.Tensor):
            t_obs = torch.as_tensor(batched_obs, device=self.device)
        else:
            t_obs = batched_obs.to(self.device)

        if t_obs.dtype == torch.uint8:
            t_obs = t_obs.to(torch.float32) / 255.0
        elif t_obs.dtype != torch.float32:
            t_obs = t_obs.to(torch.float32)

        n_envs = t_obs.shape[0]

        self.model.eval()
        with torch.no_grad():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    q_values = self.model(t_obs)
            else:
                q_values = self.model(t_obs)

            greedy_actions = q_values.argmax(dim=1)

            if epsilon > 0.0:
                random_actions = torch.randint(
                    0, self.config.n_actions, (n_envs,), device=self.device
                )
                mask = torch.rand(n_envs, device=self.device) < epsilon
                actions = torch.where(mask, random_actions, greedy_actions)
            else:
                actions = greedy_actions

        return actions.cpu().numpy()

    def train_step(self, batch: BatchedTransition) -> float:
        """执行单次批处理梯度更新。"""
        self.model.train()

        with torch.no_grad():
            target_net = (
                self.target_model if self.config.use_target_network else self.model
            )
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    target_values = target_net(batch.next_obs)
                    if self.config.double_dqn:
                        next_actions = self.model(batch.next_obs).argmax(dim=1, keepdim=True)
                        next_q = target_values.gather(1, next_actions).squeeze(1)
                    else:
                        next_q = target_values.max(dim=1).values
            else:
                target_values = target_net(batch.next_obs)
                if self.config.double_dqn:
                    next_actions = self.model(batch.next_obs).argmax(dim=1, keepdim=True)
                    next_q = target_values.gather(1, next_actions).squeeze(1)
                else:
                    next_q = target_values.max(dim=1).values

            done = (
                batch.terminated
                if self.config.bootstrap_truncated
                else (batch.terminated | batch.truncated)
            )
            targets = batch.rewards + self.config.gamma * next_q * (~done).float()

        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = self.model(batch.obs)
                pred_q = pred.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
                loss = self.criterion(pred_q, targets)
        else:
            pred = self.model(batch.obs)
            pred_q = pred.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
            loss = self.criterion(pred_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.updates += 1
        if (
            self.config.use_target_network
            and self.updates % self.config.target_sync_every == 0
        ):
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_syncs += 1

        return float(loss.item())

    def save(self, path: str | Path) -> None:
        """保存权重和配置。与 DQNAgent.load 保持向后兼容。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "target_syncs": self.target_syncs,
            "dqn_config": asdict(self.config),
            "torch_rng_state": torch.get_rng_state(),
        }
        if self.device.type == "cuda":
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(payload, p)

    def load_model_weights(self, path: str | Path) -> None:
        """从 checkpoint 中加载模型权重，用于迁移学习 (保持当前 config 不变)。"""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        if "target_model" in payload and self.config.use_target_network:
            self.target_model.load_state_dict(payload["target_model"])
        elif self.config.use_target_network:
            self.target_model.load_state_dict(payload["model"])

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> BatchedDQNAgent:
        """加载已保存的检查点。"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        payload = torch.load(path, map_location=device, weights_only=False)
        config = DQNConfig(**payload["dqn_config"])
        agent = cls(config, device=device)
        agent.model.load_state_dict(payload["model"])
        agent.target_model.load_state_dict(payload["target_model"])
        if "optimizer" in payload:
            agent.optimizer.load_state_dict(payload["optimizer"])
        agent.updates = payload.get("updates", 0)
        agent.target_syncs = payload.get("target_syncs", 0)
        return agent
