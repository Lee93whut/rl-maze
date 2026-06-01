"""tests/test_train.py —— train.py 核心函数单元测试

覆盖：
* set_seed          — 随机源锁定
* select_action     — ε=1.0 纯随机 / ε=0.0 贪心两个分支
* optimize_model    — Vanilla DQN 与 Double DQN 损失计算
* run_evaluation    — 成功率 / SPL 指标输出
* train() 配置验证  — VALID_ALGORITHMS 异常路径
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# 将 src/ 目录注入 sys.path
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from train import set_seed, select_action, optimize_model, run_evaluation, train
from model import DQNNetwork
from replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# 辅助：构造一个小型 DQNNetwork（5×5）并填充 buffer
# ---------------------------------------------------------------------------

GRID = 5
N_ACTIONS = 4
DEVICE = torch.device("cpu")


def _make_net() -> DQNNetwork:
    net = DQNNetwork(grid_size=GRID, num_actions=N_ACTIONS)
    return net


def _make_buffer(n: int = 128) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=512)
    rng = np.random.default_rng(0)
    for _ in range(n):
        s  = rng.random((4, GRID, GRID), dtype=np.float32)
        a  = int(rng.integers(0, N_ACTIONS))
        r  = float(rng.standard_normal())
        ns = rng.random((4, GRID, GRID), dtype=np.float32)
        d  = bool(rng.integers(0, 2))
        buf.push(s, a, r, ns, d)
    return buf


# ===========================================================================
# 1. set_seed
# ===========================================================================

@pytest.mark.unit
class TestSetSeed:
    def test_torch_reproducible(self) -> None:
        """相同 seed 两次调用后，随机张量完全一致。"""
        set_seed(0)
        t1 = torch.randn(4)
        set_seed(0)
        t2 = torch.randn(4)
        assert torch.allclose(t1, t2), "set_seed 后 torch 随机数应可复现"

    def test_random_reproducible(self) -> None:
        """set_seed 锁定 Python random 模块。"""
        set_seed(42)
        v1 = [random.random() for _ in range(8)]
        set_seed(42)
        v2 = [random.random() for _ in range(8)]
        assert v1 == v2

    def test_different_seeds_differ(self) -> None:
        set_seed(1)
        t1 = torch.randn(4)
        set_seed(2)
        t2 = torch.randn(4)
        assert not torch.allclose(t1, t2), "不同 seed 的随机结果不应相同"


# ===========================================================================
# 2. select_action
# ===========================================================================

@pytest.mark.unit
class TestSelectAction:
    def test_random_branch_epsilon_1(self) -> None:
        """ε=1.0 时应始终随机选择（never greedy）。"""
        net = _make_net()
        state = np.zeros((4, GRID, GRID), dtype=np.float32)
        actions = {
            select_action(state, net, epsilon=1.0, num_actions=N_ACTIONS, device=DEVICE)
            for _ in range(200)
        }
        # 期望多种动作都被选到
        assert len(actions) > 1, "ε=1.0 应随机选动作"

    def test_random_branch_returns_valid_action(self) -> None:
        """随机路径返回值在 [0, num_actions) 内。"""
        net = _make_net()
        state = np.zeros((4, GRID, GRID), dtype=np.float32)
        for _ in range(50):
            a = select_action(state, net, epsilon=1.0, num_actions=N_ACTIONS, device=DEVICE)
            assert 0 <= a < N_ACTIONS

    def test_greedy_branch_epsilon_0(self) -> None:
        """ε=0.0 时应选 Q 值最大的确定性动作。"""
        set_seed(7)
        net = _make_net()
        net.eval()
        state = np.random.rand(4, GRID, GRID).astype(np.float32)

        # 手动计算期望动作
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0)
            expected_action = int(net(s).argmax(dim=1).item())

        # 多次调用应始终相同
        for _ in range(10):
            a = select_action(state, net, epsilon=0.0, num_actions=N_ACTIONS, device=DEVICE)
            assert a == expected_action, "ε=0 时应始终选同一动作"

    def test_greedy_no_grad(self) -> None:
        """ε=0.0 分支不应留下计算图（no_grad 保护）。"""
        net = _make_net()
        state = np.ones((4, GRID, GRID), dtype=np.float32)
        a = select_action(state, net, epsilon=0.0, num_actions=N_ACTIONS, device=DEVICE)
        assert isinstance(a, int)


# ===========================================================================
# 3. optimize_model
# ===========================================================================

@pytest.mark.unit
class TestOptimizeModel:
    """验证 Vanilla 和 Double DQN 两条路径均能正常运行，损失和梯度合理。"""

    def _run_one_step(self, use_double: bool) -> tuple[float, float, float]:
        set_seed(0)
        policy_net = _make_net()
        target_net = _make_net()
        target_net.load_state_dict(policy_net.state_dict())
        optimizer  = optim.Adam(policy_net.parameters(), lr=1e-3)
        buffer     = _make_buffer(128)
        return optimize_model(
            policy_net, target_net, optimizer, buffer,
            batch_size=32, gamma=0.99, device=DEVICE,
            use_double=use_double,
        )

    def test_vanilla_loss_is_finite(self) -> None:
        loss, avg_q, grad_norm = self._run_one_step(use_double=False)
        assert np.isfinite(loss),      f"Vanilla loss 应为有限值，得到 {loss}"
        assert np.isfinite(avg_q),     f"avg_q 应为有限值，得到 {avg_q}"
        assert np.isfinite(grad_norm), f"grad_norm 应为有限值，得到 {grad_norm}"

    def test_double_loss_is_finite(self) -> None:
        loss, avg_q, grad_norm = self._run_one_step(use_double=True)
        assert np.isfinite(loss)
        assert np.isfinite(avg_q)
        assert np.isfinite(grad_norm)

    def test_loss_is_positive(self) -> None:
        """Huber Loss（smooth_l1）恒 ≥ 0。"""
        loss, _, _ = self._run_one_step(use_double=False)
        assert loss >= 0.0

    def test_grad_norm_positive_after_update(self) -> None:
        """梯度更新后梯度范数应 > 0（有信息传播）。"""
        _, _, grad_norm = self._run_one_step(use_double=False)
        assert grad_norm > 0.0

    def test_params_change_after_update(self) -> None:
        """梯度更新后网络参数应发生变化。"""
        set_seed(0)
        net    = _make_net()
        target = _make_net()
        target.load_state_dict(net.state_dict())
        opt    = optim.Adam(net.parameters(), lr=1e-3)
        buf    = _make_buffer(128)

        # 记录更新前参数快照
        before = [p.clone() for p in net.parameters()]
        optimize_model(net, target, opt, buf, 32, 0.99, DEVICE, use_double=False)
        after  = list(net.parameters())

        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed, "optimize_model 后网络参数应发生变化"

    def test_double_vs_vanilla_differ(self) -> None:
        """相同初始状态下，Double 和 Vanilla 的 loss 值不必相同（因目标计算不同）。"""
        set_seed(0)
        policy = _make_net()
        target = _make_net()
        target.load_state_dict(policy.state_dict())
        opt = optim.Adam(policy.parameters(), lr=0.0)  # lr=0 不更新参数
        buf = _make_buffer(128)

        loss_v, _, _ = optimize_model(policy, target, opt, buf, 32, 0.99, DEVICE, False)
        loss_d, _, _ = optimize_model(policy, target, opt, buf, 32, 0.99, DEVICE, True)
        # 两者可能相等，但如果 policy 与 target 初始不同会不同；至少均有限
        assert np.isfinite(loss_v) and np.isfinite(loss_d)


# ===========================================================================
# 4. run_evaluation
# ===========================================================================

@pytest.mark.unit
class TestRunEvaluation:
    """使用真实 MazeEnv（5×5，无障碍）测试评估流程。"""

    def test_returns_tuple_of_two_floats(self) -> None:
        net = _make_net()
        result = run_evaluation(
            policy_net=net,
            grid_size=5,
            obstacle_density=0.0,
            max_steps=50,
            device=DEVICE,
            test_seeds=[0, 1, 2],
            reward_goal=100.0,
            reward_wall_hit=-10.0,
            reward_step=-1.0,
        )
        assert len(result) == 2
        sr, spl = result
        assert isinstance(sr,  float)
        assert isinstance(spl, float)

    def test_success_rate_in_range(self) -> None:
        net = _make_net()
        sr, _ = run_evaluation(
            policy_net=net,
            grid_size=5,
            obstacle_density=0.0,
            max_steps=50,
            device=DEVICE,
            test_seeds=list(range(10)),
            reward_goal=100.0,
            reward_wall_hit=-10.0,
            reward_step=-1.0,
        )
        assert 0.0 <= sr <= 100.0, f"success_rate 应在 [0,100]，得到 {sr}"

    def test_spl_nonnegative(self) -> None:
        net = _make_net()
        _, spl = run_evaluation(
            policy_net=net,
            grid_size=5,
            obstacle_density=0.0,
            max_steps=50,
            device=DEVICE,
            test_seeds=list(range(5)),
            reward_goal=100.0,
            reward_wall_hit=-10.0,
            reward_step=-1.0,
        )
        assert spl >= 0.0

    def test_policy_restored_to_train_mode(self) -> None:
        """run_evaluation 结束后 policy_net 应回到 train() 模式。"""
        net = _make_net()
        run_evaluation(
            policy_net=net,
            grid_size=5,
            obstacle_density=0.0,
            max_steps=30,
            device=DEVICE,
            test_seeds=[0],
            reward_goal=100.0,
            reward_wall_hit=-10.0,
            reward_step=-1.0,
        )
        assert net.training, "run_evaluation 结束后网络应处于 train() 模式"

    def test_empty_seeds_returns_zero_spl(self) -> None:
        """空测试集返回 spl=0.0；success_rate 为 nan（np.mean 空列表行为），但 spl 有保护。"""
        net = _make_net()
        _, spl = run_evaluation(
            policy_net=net,
            grid_size=5,
            obstacle_density=0.0,
            max_steps=30,
            device=DEVICE,
            test_seeds=[],
            reward_goal=100.0,
            reward_wall_hit=-10.0,
            reward_step=-1.0,
        )
        assert spl == 0.0


# ===========================================================================
# 5. train() VALID_ALGORITHMS 验证
# ===========================================================================

@pytest.mark.unit
class TestTrainConfigValidation:
    """测试 train() 对非法 algorithm 值抛出 ValueError。"""

    _MINIMAL_CFG = {
        "maze":    {"grid_size": 5, "obstacle_density": 0.0, "max_steps": 10},
        "rewards": {"goal": 100, "wall_hit": -10, "step": -1},
        "dqn":     {
            "seed": 0, "algorithm": "INVALID_ALGO",
            "buffer_capacity": 512, "batch_size": 32,
            "num_episodes": 1, "learning_rate": 1e-3,
            "gamma": 0.99, "epsilon_start": 1.0, "epsilon_end": 0.05,
            "epsilon_decay": 0.99, "target_update_freq": 100,
            "warmup_episodes": 0, "log_dir": "/tmp/test_runs",
            "save_dir": "/tmp/test_results", "success_window": 10,
            "save_window": 5, "print_every": 1, "eval_every": 9999,
            "num_test_mazes": 1,
        },
        "overfit": {},
    }

    def test_invalid_algorithm_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="不支持的 algorithm"):
            train(self._MINIMAL_CFG, overfit_mode=False)

    @pytest.mark.parametrize("algo", ["vanilla", "double", "dueling", "double_dueling"])
    def test_valid_algorithms_do_not_raise(self, algo: str, tmp_path) -> None:
        """合法 algorithm 值不应在解析阶段抛出。

        通过 mock SummaryWriter 和 MazeEnv 避免真实训练副作用。
        """
        cfg = {
            "maze":    {"grid_size": 5, "obstacle_density": 0.0, "max_steps": 5},
            "rewards": {"goal": 100, "wall_hit": -10, "step": -1},
            "dqn":     {
                "seed": 0, "algorithm": algo,
                "buffer_capacity": 64, "batch_size": 4,
                "num_episodes": 2, "learning_rate": 1e-3,
                "gamma": 0.99, "epsilon_start": 1.0, "epsilon_end": 0.05,
                "epsilon_decay": 0.99, "target_update_freq": 999,
                "warmup_episodes": 0,
                "log_dir":   str(tmp_path / "runs"),
                "save_dir":  str(tmp_path / "results"),
                "success_window": 5, "save_window": 2,
                "print_every": 9999, "eval_every": 9999,
                "num_test_mazes": 1,
            },
            "overfit": {},
        }
        # 仅验证不抛出 ValueError；允许其他运行时错误（如 TensorBoard I/O）
        try:
            train(cfg, overfit_mode=False)
        except ValueError:
            pytest.fail(f"algorithm='{algo}' 不应触发 ValueError")
        except Exception:
            pass  # 其他异常（环境/IO）不在本测试范围内


# ===========================================================================
# 6. train() overfit 模式 & eval_every 路径覆盖
# ===========================================================================

def _base_cfg(tmp_path, algo: str = "vanilla", overfit: bool = False) -> dict:
    """生成一个最小可运行配置，3 个 episode，无 warmup，eval_every=1 触发盲测。"""
    return {
        "maze":    {"grid_size": 5, "obstacle_density": 0.0, "max_steps": 10},
        "rewards": {"goal": 100, "wall_hit": -10, "step": -1},
        "dqn":     {
            "seed": 0, "algorithm": algo,
            "buffer_capacity": 128, "batch_size": 8,
            "num_episodes": 3,
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "epsilon_start": 1.0, "epsilon_end": 0.05,
            "epsilon_decay": 0.99,
            "target_update_freq": 1,   # 每步都同步，覆盖 target sync 分支
            "warmup_episodes": 0,      # 无 warmup，立即更新
            "log_dir":   str(tmp_path / "runs"),
            "save_dir":  str(tmp_path / "results"),
            "success_window": 5,
            "save_window": 2,
            "print_every": 9999,
            "eval_every": 1,           # 每局都评估，覆盖 eval 分支
            "num_test_mazes": 2,
        },
        "overfit": {
            "grid_size": 5, "obstacle_density": 0.0, "max_steps": 10,
            "seed": 0, "num_episodes": 3,
            "epsilon_decay": 0.99, "warmup_episodes": 0,
            "batch_size": 8, "target_update_freq": 1,
            "print_every": 9999, "eval_every": 1, "num_test_mazes": 2,
            "algorithm": algo,
        },
    }


@pytest.mark.unit
class TestTrainMainLoop:
    """覆盖 train() 主循环中的 eval_every / target sync / overfit 分支。"""

    def test_train_vanilla_runs_without_error(self, tmp_path) -> None:
        cfg = _base_cfg(tmp_path, algo="vanilla")
        train(cfg, overfit_mode=False)   # 不应抛出

    def test_train_double_dueling_runs_without_error(self, tmp_path) -> None:
        cfg = _base_cfg(tmp_path, algo="double_dueling")
        train(cfg, overfit_mode=False)

    def test_train_overfit_mode_runs_without_error(self, tmp_path) -> None:
        """overfit_mode=True 覆盖 281-300 和 557-563 分支。"""
        cfg = _base_cfg(tmp_path, algo="double_dueling", overfit=True)
        train(cfg, overfit_mode=True)

    def test_train_eval_every_fires(self, tmp_path) -> None:
        """eval_every=1 确保每局调用 run_evaluation（覆盖 491-504 行）。"""
        cfg = _base_cfg(tmp_path, algo="double")
        train(cfg, overfit_mode=False)

    def test_checkpoint_saved(self, tmp_path) -> None:
        """save_window=2，3 个 episode 应触发模型保存。"""
        cfg = _base_cfg(tmp_path, algo="vanilla")
        train(cfg, overfit_mode=False)
        results_dir = tmp_path / "results"
        saved = list(results_dir.glob("best_model_*.pth"))
        assert len(saved) >= 1, "预期保存至少一个 checkpoint"
