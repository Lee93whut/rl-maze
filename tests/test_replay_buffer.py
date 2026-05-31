"""tests/test_replay_buffer.py —— ReplayBuffer 单元测试

覆盖：
* push / sample 基本功能
* 环形覆盖逻辑（容量满后覆盖最旧条目）
* O(batch_size) 采样：list 存储（非 deque）
* sample 返回 Tensor 形状与 dtype
* 边界条件（capacity=1, batch_size > len → ValueError）
* is_ready / __len__ / __repr__
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# src/ 目录下的 replay_buffer.py 不属于可安装包，注入 sys.path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from replay_buffer import ReplayBuffer, Transition  # noqa: E402


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

def _make_state(n: int = 4) -> np.ndarray:
    return np.zeros((4, n, n), dtype=np.float32)


def _push_n(buf: ReplayBuffer, n: int, grid: int = 4) -> None:
    for i in range(n):
        buf.push(
            state=_make_state(grid),
            action=i % 4,
            reward=float(i),
            next_state=_make_state(grid),
            done=(i % 5 == 0),
        )


# ---------------------------------------------------------------------------
# 容量与长度
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCapacity:
    def test_empty_at_start(self) -> None:
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0

    def test_len_grows(self) -> None:
        buf = ReplayBuffer(capacity=100)
        _push_n(buf, 10)
        assert len(buf) == 10

    def test_len_capped_at_capacity(self) -> None:
        buf = ReplayBuffer(capacity=5)
        _push_n(buf, 20)
        assert len(buf) == 5   # 超出后不再增长

    def test_capacity_1(self) -> None:
        buf = ReplayBuffer(capacity=1)
        _push_n(buf, 3)
        assert len(buf) == 1

    def test_invalid_capacity(self) -> None:
        with pytest.raises(ValueError):
            ReplayBuffer(capacity=0)


# ---------------------------------------------------------------------------
# 环形覆盖：写指针行为
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCircularOverwrite:
    def test_overwrite_oldest(self) -> None:
        """capacity=3，push 4 条后，缓冲区只保留最新 3 条。"""
        buf = ReplayBuffer(capacity=3)
        for i in range(4):
            buf.push(
                state=_make_state(),
                action=0,
                reward=float(i),        # reward 用来区分条目
                next_state=_make_state(),
                done=False,
            )
        rewards = {t.reward for t in buf._buffer}
        assert 0.0 not in rewards, "最旧条目（reward=0）应已被覆盖"
        assert {1.0, 2.0, 3.0} == rewards

    def test_pos_wraps(self) -> None:
        buf = ReplayBuffer(capacity=3)
        _push_n(buf, 6)
        # 写指针应在 6 % 3 = 0
        assert buf._pos == 0


# ---------------------------------------------------------------------------
# sample：返回值格式
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSample:
    def test_sample_shapes(self) -> None:
        buf = ReplayBuffer(capacity=100)
        _push_n(buf, 64)
        batch = buf.sample(32, device=torch.device("cpu"))

        assert batch["states"].shape      == (32, 4, 4, 4)
        assert batch["next_states"].shape == (32, 4, 4, 4)
        assert batch["actions"].shape     == (32,)
        assert batch["rewards"].shape     == (32,)
        assert batch["dones"].shape       == (32,)

    def test_sample_dtypes(self) -> None:
        buf = ReplayBuffer(capacity=100)
        _push_n(buf, 32)
        batch = buf.sample(16, device=torch.device("cpu"))

        assert batch["states"].dtype      == torch.float32
        assert batch["next_states"].dtype == torch.float32
        assert batch["actions"].dtype     == torch.int64
        assert batch["rewards"].dtype     == torch.float32
        assert batch["dones"].dtype       == torch.float32

    def test_sample_dones_binary(self) -> None:
        """done 应被编码为 0.0 / 1.0 的 float32。"""
        buf = ReplayBuffer(capacity=50)
        _push_n(buf, 50)
        batch = buf.sample(50, device=torch.device("cpu"))
        unique = torch.unique(batch["dones"])
        for v in unique:
            assert v.item() in {0.0, 1.0}

    def test_sample_raises_when_insufficient(self) -> None:
        buf = ReplayBuffer(capacity=100)
        _push_n(buf, 5)
        with pytest.raises(ValueError):
            buf.sample(10, device=torch.device("cpu"))

    def test_sample_all(self) -> None:
        buf = ReplayBuffer(capacity=10)
        _push_n(buf, 10)
        batch = buf.sample(10, device=torch.device("cpu"))
        assert batch["states"].shape[0] == 10


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIsReady:
    def test_not_ready(self) -> None:
        buf = ReplayBuffer(capacity=100)
        _push_n(buf, 10)
        assert not buf.is_ready(64)

    def test_ready(self) -> None:
        buf = ReplayBuffer(capacity=100)
        _push_n(buf, 64)
        assert buf.is_ready(64)

    def test_exactly_threshold(self) -> None:
        buf = ReplayBuffer(capacity=100)
        _push_n(buf, 32)
        assert buf.is_ready(32)
        assert not buf.is_ready(33)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_repr() -> None:
    buf = ReplayBuffer(capacity=500)
    _push_n(buf, 10)
    r = repr(buf)
    assert "500" in r
    assert "10" in r


# ---------------------------------------------------------------------------
# Transition NamedTuple
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_transition_fields() -> None:
    s = _make_state()
    t = Transition(state=s, action=2, reward=1.5, next_state=s, done=True)
    assert t.action == 2
    assert t.reward == 1.5
    assert t.done is True
