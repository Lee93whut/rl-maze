"""
测试模块 05 —— 终止条件（terminated / truncated）

需求覆盖
--------
* R9：truncated 仅在 max_steps 耗尽且未到终点时触发
* R10：terminated 与 truncated 严格互斥

对应用例
--------
TC-14, TC-15
"""

from __future__ import annotations

import pytest

from maze_env import MazeEnv


class TestTermination:
    """验证 terminated 与 truncated 的互斥语义及触发条件。"""

    # ------------------------------------------------------------------ #
    # TC-14  truncated：步数耗尽                                           #
    # ------------------------------------------------------------------ #

    @pytest.mark.integration
    def test_truncated_on_max_steps(self) -> None:
        """TC-14a：步数耗尽（max_steps=5）时，truncated=True，terminated=False。

        输入:  MazeEnv(grid_size=6, obstacle_density=0.0, seed=0, max_steps=5)
               反复撞墙（action=0）5 次
        期望:  第 5 步 truncated=True，terminated=False
        实测:  step() 返回值
        """
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0, max_steps=5)
        env.reset()
        truncated = False
        terminated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(0)  # 持续撞上边界墙
        assert truncated  is True,  "步数耗尽时应 truncated=True"
        assert terminated is False, "步数耗尽但未到终点，terminated 应为 False"

    @pytest.mark.integration
    def test_truncated_success_flag_false(self) -> None:
        """TC-14b：步数耗尽时，info['success'] 应为 False。

        输入:  max_steps=3，反复撞墙
        期望:  info["success"] is False
        实测:  最后一步 info
        """
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0, max_steps=3)
        env.reset()
        info = {}
        for _ in range(3):
            _, _, _, _, info = env.step(0)
        assert info["success"] is False

    # ------------------------------------------------------------------ #
    # TC-15  terminated 优先于 truncated                                   #
    # ------------------------------------------------------------------ #

    @pytest.mark.integration
    def test_terminated_when_goal_at_last_step(self) -> None:
        """TC-15：在最后一步恰好到达终点，应 terminated=True，truncated=False。

        输入:  max_steps=6，引导 agent 在第 6 步走到 (4,4)
        期望:  terminated=True，truncated=False（严格互斥）
        实测:  step() 返回值
        """
        # 6 步恰好：右×3 + 下×3 = 6 步，最后一步到达 goal
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0, max_steps=6)
        env.reset()
        for _ in range(3):
            env.step(3)   # 右
        for _ in range(2):
            env.step(1)   # 下
        _, _, terminated, truncated, info = env.step(1)  # 第 6 步到终点

        assert terminated is True,  "终点步应 terminated=True"
        assert truncated  is False, "到达终点时 truncated 必须为 False（严格互斥）"
        assert info["success"] is True
