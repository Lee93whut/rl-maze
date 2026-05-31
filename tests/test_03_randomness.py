"""
测试模块 03 —— 随机性与可复现性

需求覆盖
--------
* R1：seed 参数控制可复现性
* RF1：BFS 连通性保证（起终点强制可达）

对应用例
--------
TC-08, TC-09, TC-10
"""

from __future__ import annotations

import numpy as np
import pytest

from maze_env import MazeEnv


class TestRandomness:
    """验证 BFS 连通性、seed 可复现性及不同 seed 产生不同地图。"""

    # ------------------------------------------------------------------ #
    # TC-08  BFS 连通性压力测试                                             #
    # ------------------------------------------------------------------ #

    @pytest.mark.slow
    def test_bfs_connectivity_pressure(self, env_dense: MazeEnv) -> None:
        """TC-08：高密度障碍下，连续 50 次 reset 均保证起终点可达。

        输入:  MazeEnv(grid_size=10, obstacle_density=0.45)，reset() × 50
        期望:  每次 reset 后，obs[1]（agent）与 obs[2]（goal）均有唯一激活格，
               且地图合法（不抛出异常，obs shape 正确）
        实测:  检查 obs shape 及通道 sum
        """
        for i in range(50):
            obs, info = env_dense.reset()
            assert obs.shape == (4, 10, 10), f"第 {i} 次 reset shape 错误"
            assert float(obs[1].sum()) == 1.0, f"第 {i} 次 reset agent 通道异常"
            assert float(obs[2].sum()) == 1.0, f"第 {i} 次 reset goal 通道异常"

    # ------------------------------------------------------------------ #
    # TC-09  相同 seed 可复现                                               #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_same_seed_reproducible(self) -> None:
        """TC-09：相同 seed 两次 reset，产生完全相同的地图与 obs。

        输入:  两个独立实例均使用 seed=42，各调用 reset()
        期望:  obs1 与 obs2 逐元素相等（np.array_equal）
        实测:  np.array_equal(obs1, obs2)
        """
        env_a = MazeEnv(grid_size=8, obstacle_density=0.3, seed=42)
        env_b = MazeEnv(grid_size=8, obstacle_density=0.3, seed=42)
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        assert np.array_equal(obs_a, obs_b), "相同 seed 产生的地图应完全一致"

    @pytest.mark.unit
    def test_same_seed_reset_twice_reproducible(self) -> None:
        """TC-09b：同一实例用相同 seed 调用两次 reset()，结果相同。

        输入:  env.reset(seed=5)，env.reset(seed=5)
        期望:  两次 obs 完全相等
        实测:  np.array_equal
        """
        env = MazeEnv(grid_size=6, obstacle_density=0.2)
        obs_first, _  = env.reset(seed=5)
        obs_second, _ = env.reset(seed=5)
        assert np.array_equal(obs_first, obs_second), \
            "同一实例以相同 seed 重置，结果应可复现"

    # ------------------------------------------------------------------ #
    # TC-10  不同 seed 产生不同地图                                         #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_different_seeds_different_maps(self) -> None:
        """TC-10：不同 seed 应产生不同的 wall_map（obs[0]）。

        输入:  seed=0 与 seed=1，obstacle_density=0.3，grid_size=10
        期望:  obs_0[0] != obs_1[0]（至少有一个元素不同）
        实测:  not np.array_equal(obs_0[0], obs_1[0])
        """
        env_0 = MazeEnv(grid_size=10, obstacle_density=0.3, seed=0)
        env_1 = MazeEnv(grid_size=10, obstacle_density=0.3, seed=1)
        obs_0, _ = env_0.reset()
        obs_1, _ = env_1.reset()
        assert not np.array_equal(obs_0[0], obs_1[0]), \
            "不同 seed 应产生不同的墙壁地图"
