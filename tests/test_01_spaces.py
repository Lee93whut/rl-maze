"""
测试模块 01 —— 空间声明（Spaces）

需求覆盖
--------
* R2：observation_space = Box(0,1,(3,N,N), float32)
* R3：action_space = Discrete(4)

对应用例
--------
TC-01, TC-02
"""

from __future__ import annotations

import numpy as np
import pytest

from maze_env import MazeEnv


class TestSpaces:
    """验证 observation_space 与 action_space 的声明是否符合 Gymnasium 规范。"""

    @pytest.mark.unit
    def test_observation_space_shape(self, env_zero: MazeEnv) -> None:
        """TC-01a：obs_space.shape 应为 (4, N, N)。

        输入:  MazeEnv(grid_size=6, ...)
        期望:  observation_space.shape == (4, 6, 6)
        实测:  直接读取 env.observation_space.shape
        """
        assert env_zero.observation_space.shape == (4, 6, 6)

    @pytest.mark.unit
    def test_observation_space_dtype(self, env_zero: MazeEnv) -> None:
        """TC-01b：obs_space.dtype 应为 float32。

        输入:  MazeEnv(grid_size=6, ...)
        期望:  observation_space.dtype == np.float32
        实测:  直接读取 env.observation_space.dtype
        """
        assert env_zero.observation_space.dtype == np.float32

    @pytest.mark.unit
    def test_action_space_size(self, env_zero: MazeEnv) -> None:
        """TC-02：action_space.n 应为 4（上/下/左/右）。

        输入:  MazeEnv(grid_size=6, ...)
        期望:  action_space.n == 4
        实测:  直接读取 env.action_space.n
        """
        assert env_zero.action_space.n == 4
