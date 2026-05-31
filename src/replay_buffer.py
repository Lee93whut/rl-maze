"""replay_buffer.py —— 经验回放池

设计要点
--------
* **环形 list 存储**：使用 ``list`` + 写指针实现定容环形缓冲区。
  相比 ``collections.deque``，避免了 ``random.sample`` 内部触发的
  隐式 O(capacity) 全量拷贝（CPython 对 deque 采样前先 list(deque)），
  采样复杂度从 O(capacity) 降至 O(batch_size)。
* 默认容量 20000（适合中小规模迷宫任务）。
* 采样时一次性将 Batch 转换为连续 NumPy 数组再转 Tensor，
  避免在循环内逐条转换（Python 循环 overhead 过大）。
* Transition 使用 ``NamedTuple`` 定义，字段具名访问，杜绝下标魔法数字。

存储格式
--------
每条经验 ``Transition(state, action, reward, next_state, done)``：

* ``state``      : ``np.ndarray`` shape ``(4, N, N)`` float32
* ``action``     : ``int``
* ``reward``     : ``float``
* ``next_state`` : ``np.ndarray`` shape ``(4, N, N)`` float32
* ``done``       : ``bool``  （terminated OR truncated）
"""

from __future__ import annotations

import random
from typing import NamedTuple

import numpy as np
import torch


__all__ = ["Transition", "ReplayBuffer"]


class Transition(NamedTuple):
    """单条经验转移（immutable，字段具名访问）。"""

    state:      np.ndarray   # (4, N, N) float32
    action:     int
    reward:     float
    next_state: np.ndarray   # (4, N, N) float32
    done:       bool         # terminated | truncated


class ReplayBuffer:
    """固定容量的经验回放池（环形 list 实现，O(batch_size) 采样）。

    Args:
        capacity: 最大存储条数。超出后循环覆盖最旧的条目。

    Example:
        >>> buf = ReplayBuffer(capacity=10000)
        >>> buf.push(state, action, reward, next_state, done)
        >>> batch = buf.sample(64, device=torch.device("cpu"))
        >>> batch["states"].shape
        torch.Size([64, 4, N, N])
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity 必须 >= 1，当前值：{capacity}")
        self.capacity: int = capacity
        self._buffer: list[Transition] = []
        self._pos: int = 0          # 环形写指针

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """存入一条经验。

        Args:
            state:      当前观测，shape ``(4, N, N)``。
            action:     执行的动作编号。
            reward:     获得的即时奖励。
            next_state: 下一步观测，shape ``(4, N, N)``。
            done:       本步是否为幕终止（terminated | truncated）。
        """
        t = Transition(
            state=state,
            action=int(action),
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
        )
        if len(self._buffer) < self.capacity:
            self._buffer.append(t)
        else:
            self._buffer[self._pos] = t
        self._pos = (self._pos + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """随机采样一个 mini-batch，返回字典形式的 Tensor。

        复杂度 O(batch_size)，list 存储避免了 deque 触发的 O(capacity) 拷贝。

        Args:
            batch_size: 采样数量，不得超过当前缓冲区大小。
            device:     目标 Tensor 设备。

        Returns:
            包含以下键的字典：

            * ``"states"``      : ``(B, 4, N, N)`` float32
            * ``"actions"``     : ``(B,)``          int64
            * ``"rewards"``     : ``(B,)``          float32
            * ``"next_states"`` : ``(B, 4, N, N)`` float32
            * ``"dones"``       : ``(B,)``          float32  (0.0 / 1.0)

        Raises:
            ValueError: 若 batch_size > len(buffer)。
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size={batch_size} 超过缓冲区当前大小 {len(self._buffer)}"
            )

        transitions: list[Transition] = random.sample(self._buffer, batch_size)

        # 批量转换：一次 np.stack 比逐条 tensor() 快 ~10x
        states      = np.stack([t.state      for t in transitions])   # (B,4,N,N)
        next_states = np.stack([t.next_state for t in transitions])   # (B,4,N,N)
        actions     = np.array([t.action     for t in transitions], dtype=np.int64)
        rewards     = np.array([t.reward     for t in transitions], dtype=np.float32)
        dones       = np.array([t.done       for t in transitions], dtype=np.float32)

        return {
            "states":      torch.from_numpy(states).to(device),
            "actions":     torch.from_numpy(actions).to(device),
            "rewards":     torch.from_numpy(rewards).to(device),
            "next_states": torch.from_numpy(next_states).to(device),
            "dones":       torch.from_numpy(dones).to(device),
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """返回当前缓冲区存储的条数。"""
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """判断缓冲区是否已积累足够条目以供采样。"""
        return len(self._buffer) >= batch_size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, "
            f"current_size={len(self._buffer)})"
        )
