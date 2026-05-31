"""tests/test_model.py —— DQNNetwork 单元测试

覆盖：
* 正向传播输出维度
* 不同 grid_size 的输出形状
* eval/train 模式切换
* 权重初始化（Conv → Kaiming，Linear → Xavier）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# src/ 目录下的 model.py 不属于可安装包，注入 sys.path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model import DQNNetwork  # noqa: E402
from model import DuelingDQNNetwork  # noqa: E402


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def net5() -> DQNNetwork:
    """5×5 迷宫网络（快速测试用）。"""
    return DQNNetwork(grid_size=5)


@pytest.fixture
def net10() -> DQNNetwork:
    """10×10 迷宫网络（与训练默认配置一致）。"""
    return DQNNetwork(grid_size=10)


# ---------------------------------------------------------------------------
# 正向传播：输出维度
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestForwardShape:
    def test_output_shape_5x5(self, net5: DQNNetwork) -> None:
        x = torch.randn(32, 4, 5, 5)
        out = net5(x)
        assert out.shape == (32, 4), f"期望 (32, 4)，实际 {out.shape}"

    def test_output_shape_10x10(self, net10: DQNNetwork) -> None:
        x = torch.randn(16, 4, 10, 10)
        out = net10(x)
        assert out.shape == (16, 4), f"期望 (16, 4)，实际 {out.shape}"

    def test_batch_size_1(self, net5: DQNNetwork) -> None:
        x = torch.randn(1, 4, 5, 5)
        out = net5(x)
        assert out.shape == (1, 4)

    def test_custom_num_actions(self) -> None:
        net = DQNNetwork(grid_size=6, num_actions=8)
        x = torch.randn(4, 4, 6, 6)
        out = net(x)
        assert out.shape == (4, 8)

    def test_output_dtype_float32(self, net5: DQNNetwork) -> None:
        x = torch.randn(2, 4, 5, 5)
        out = net5(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# 权重初始化
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWeightInit:
    def test_conv_bias_zeros(self, net5: DQNNetwork) -> None:
        for m in net5.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                assert torch.all(m.bias == 0.0), "Conv bias 应初始化为 0"

    def test_linear_bias_zeros(self, net5: DQNNetwork) -> None:
        for m in net5.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                assert torch.all(m.bias == 0.0), "Linear bias 应初始化为 0"

    def test_conv_weights_not_all_zeros(self, net5: DQNNetwork) -> None:
        for m in net5.modules():
            if isinstance(m, nn.Conv2d):
                assert not torch.all(m.weight == 0.0), "Conv 权重不应全为 0"

    def test_linear_weights_not_all_zeros(self, net5: DQNNetwork) -> None:
        for m in net5.modules():
            if isinstance(m, nn.Linear):
                assert not torch.all(m.weight == 0.0), "Linear 权重不应全为 0"


# ---------------------------------------------------------------------------
# eval / train 模式
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestModeSwitch:
    def test_eval_mode(self, net5: DQNNetwork) -> None:
        net5.eval()
        assert not net5.training

    def test_train_mode(self, net5: DQNNetwork) -> None:
        net5.eval()
        net5.train()
        assert net5.training

    def test_inference_no_grad(self, net5: DQNNetwork) -> None:
        net5.eval()
        x = torch.randn(1, 4, 5, 5)
        with torch.no_grad():
            out = net5(x)
        # 确保可正常取值
        assert out.shape == (1, 4)


# ---------------------------------------------------------------------------
# 参数量回归（防止意外改动网络结构）
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_param_count_5x5() -> None:
    net = DQNNetwork(grid_size=5, input_channels=4, num_actions=4)
    total = sum(p.numel() for p in net.parameters())
    # 不固定精确值，但要求在合理区间（数千 ~ 百万）
    assert 10_000 < total < 10_000_000, f"参数量异常：{total}"


# ---------------------------------------------------------------------------
# DuelingDQNNetwork 夹具
# ---------------------------------------------------------------------------

@pytest.fixture
def dueling5() -> DuelingDQNNetwork:
    """5×5 Dueling 网络（快速测试用）。"""
    return DuelingDQNNetwork(grid_size=5)


@pytest.fixture
def dueling10() -> DuelingDQNNetwork:
    """10×10 Dueling 网络（与训练默认配置一致）。"""
    return DuelingDQNNetwork(grid_size=10)


# ---------------------------------------------------------------------------
# DuelingDQNNetwork：正向传播输出维度
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDuelingForwardShape:
    def test_output_shape_5x5(self, dueling5: DuelingDQNNetwork) -> None:
        x = torch.randn(32, 4, 5, 5)
        out = dueling5(x)
        assert out.shape == (32, 4), f"期望 (32, 4)，实际 {out.shape}"

    def test_output_shape_10x10(self, dueling10: DuelingDQNNetwork) -> None:
        x = torch.randn(16, 4, 10, 10)
        out = dueling10(x)
        assert out.shape == (16, 4), f"期望 (16, 4)，实际 {out.shape}"

    def test_batch_size_1(self, dueling5: DuelingDQNNetwork) -> None:
        x = torch.randn(1, 4, 5, 5)
        out = dueling5(x)
        assert out.shape == (1, 4)

    def test_output_dtype_float32(self, dueling5: DuelingDQNNetwork) -> None:
        x = torch.randn(2, 4, 5, 5)
        out = dueling5(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# DuelingDQNNetwork：V+A 分解核心语义
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDuelingDecomposition:
    def test_value_advantage_decomposition(self, dueling5: DuelingDQNNetwork) -> None:
        """直接访问 value_stream / advantage_stream，验证输出形状。"""
        x = torch.randn(8, 4, 5, 5)
        feat = dueling5.flatten(dueling5.conv(x))   # (8, flat_dim)
        V = dueling5.value_stream(feat)              # 期望 (8, 1)
        A = dueling5.advantage_stream(feat)          # 期望 (8, 4)
        assert V.shape == (8, 1),  f"value_stream 输出形状错误：{V.shape}"
        assert A.shape == (8, 4),  f"advantage_stream 输出形状错误：{A.shape}"

    def test_mean_subtraction(self, dueling5: DuelingDQNNetwork) -> None:
        """验证 mean(A) 被减去：Q = V + A - mean(A)，Q 的 mean 等于 V.squeeze。"""
        x = torch.randn(4, 4, 5, 5)
        feat = dueling5.flatten(dueling5.conv(x))
        V = dueling5.value_stream(feat)              # (4, 1)
        A = dueling5.advantage_stream(feat)          # (4, 4)
        Q_expected = V + A - A.mean(dim=1, keepdim=True)
        Q_actual   = dueling5(x)
        assert torch.allclose(Q_actual, Q_expected, atol=1e-5), \
            "Q 值与 V+A-mean(A) 不匹配"
        # 确认 mean(A) 确实被减去：Q 的行均值应等于 V.squeeze()
        assert torch.allclose(Q_actual.mean(dim=1), V.squeeze(1), atol=1e-5), \
            "Q 的行均值应等于 V(s)"


# ---------------------------------------------------------------------------
# DuelingDQNNetwork：权重初始化
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDuelingWeightInit:
    def test_conv_bias_zeros(self, dueling5: DuelingDQNNetwork) -> None:
        for m in dueling5.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                assert torch.all(m.bias == 0.0), "Conv bias 应初始化为 0"

    def test_linear_bias_zeros(self, dueling5: DuelingDQNNetwork) -> None:
        for m in dueling5.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                assert torch.all(m.bias == 0.0), "Linear bias 应初始化为 0"

    def test_conv_weights_not_all_zeros(self, dueling5: DuelingDQNNetwork) -> None:
        for m in dueling5.modules():
            if isinstance(m, nn.Conv2d):
                assert not torch.all(m.weight == 0.0), "Conv 权重不应全为 0"

    def test_linear_weights_not_all_zeros(self, dueling5: DuelingDQNNetwork) -> None:
        for m in dueling5.modules():
            if isinstance(m, nn.Linear):
                assert not torch.all(m.weight == 0.0), "Linear 权重不应全为 0"


# ---------------------------------------------------------------------------
# DuelingDQNNetwork：参数量 & 结构对比
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_param_count_dueling_5x5() -> None:
    net = DuelingDQNNetwork(grid_size=5, input_channels=4, num_actions=4)
    total = sum(p.numel() for p in net.parameters())
    assert 10_000 < total < 10_000_000, f"Dueling 参数量异常：{total}"


@pytest.mark.unit
def test_dueling_vs_dqn_same_conv() -> None:
    """两个网络 conv 部分参数量相同（相同 grid_size 下共享相同主干结构）。"""
    dqn     = DQNNetwork(grid_size=5)
    dueling = DuelingDQNNetwork(grid_size=5)
    dqn_conv_params     = sum(p.numel() for m in dqn.modules()
                              if isinstance(m, nn.Conv2d) for p in m.parameters())
    dueling_conv_params = sum(p.numel() for m in dueling.modules()
                              if isinstance(m, nn.Conv2d) for p in m.parameters())
    assert dqn_conv_params == dueling_conv_params, (
        f"Conv 参数量不一致：DQN={dqn_conv_params}, Dueling={dueling_conv_params}"
    )
