"""
test_required.py —— 四个指定验收测试用例

TC-R1  test_dimension_and_channels
TC-R2  test_map_connectivity
TC-R3  test_termination_and_truncation
TC-R4  test_seeding_reproducibility
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from maze_env import MazeEnv


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数：独立 BFS，用于外部验证迷宫连通性
# ──────────────────────────────────────────────────────────────────────────────

def _bfs_connected(wall_map: np.ndarray, start: tuple[int, int],
                   goal: tuple[int, int]) -> bool:
    """对给定 wall_map 执行 BFS，返回 start 到 goal 是否可达。

    Args:
        wall_map: shape (N, N) float32 数组，1.0 代表墙，0.0 代表可通行。
        start:    起始格坐标 (row, col)。
        goal:     目标格坐标 (row, col)。

    Returns:
        True 表示可达，False 表示不可达。
    """
    N = wall_map.shape[0]
    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if (0 <= nr < N and 0 <= nc < N
                    and wall_map[nr, nc] == 0.0
                    and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc))
    return (start == goal)


# ──────────────────────────────────────────────────────────────────────────────
# TC-R1  test_dimension_and_channels
# ──────────────────────────────────────────────────────────────────────────────

class TestDimensionAndChannels:
    """TC-R1：10×10 环境的观测形状与通道语义验证。"""

    @pytest.mark.unit
    def test_dimension_and_channels(self) -> None:
        """实例化 10×10 环境，断言 obs.shape==(3,10,10)，
        Agent 通道和终点通道各自 sum==1。

        输入:  MazeEnv(grid_size=10, obstacle_density=0.3, seed=0).reset()
        期望:
          obs.shape == (4, 10, 10)
          obs[1].sum() == 1.0   （Agent 通道：唯一激活格）
          obs[2].sum() == 1.0   （终点通道：唯一激活格）
        实测:  obs 各维度及通道 sum
        """
        env = MazeEnv(grid_size=10, obstacle_density=0.3, seed=0)
        obs, _ = env.reset()

        assert obs.shape == (4, 10, 10), \
            f"obs.shape 期望 (3,10,10)，实际 {obs.shape}"
        assert float(obs[1].sum()) == 1.0, \
            f"Agent 通道 (obs[1]) 应恰好有 1 个激活格，实际 sum={obs[1].sum()}"
        assert float(obs[2].sum()) == 1.0, \
            f"终点通道 (obs[2]) 应恰好有 1 个激活格，实际 sum={obs[2].sum()}"


# ──────────────────────────────────────────────────────────────────────────────
# TC-R2  test_map_connectivity
# ──────────────────────────────────────────────────────────────────────────────

class TestMapConnectivity:
    """TC-R2：连续 100 次 reset，每次用独立 BFS 验证起终点可达。"""

    @pytest.mark.slow
    def test_map_connectivity(self) -> None:
        """循环 reset() 100 次，用外部 BFS 独立验证每张地图的起终点连通性。

        输入:  MazeEnv(grid_size=10, obstacle_density=0.45)，reset() × 100
        期望:  100% 连通（任意一次不连通即失败）
        实测:  外部 BFS 验证 wall_map（obs[0]），start=(1,1)，goal=(N-2,N-2)
        """
        N = 10
        start = (1, 1)
        goal  = (N - 2, N - 2)
        env = MazeEnv(grid_size=N, obstacle_density=0.45)

        for i in range(100):
            obs, info = env.reset()
            wall_map = obs[0]  # 通道 0 = 静态墙壁地图

            connected = _bfs_connected(wall_map, start, goal)
            assert connected, (
                f"第 {i+1} 次 reset：BFS 验证起点 {start} → 终点 {goal} 不连通，"
                f"说明 MazeEnv 的连通性过滤器未生效。"
            )


# ──────────────────────────────────────────────────────────────────────────────
# TC-R3  test_termination_and_truncation
# ──────────────────────────────────────────────────────────────────────────────

class TestTerminationAndTruncation:
    """TC-R3：验证 truncated（步数耗尽）与 terminated（到达终点）的互斥语义。"""

    @pytest.mark.integration
    def test_termination_and_truncation(self) -> None:
        """Part A：疯狂撞墙直至 max_steps=50，断言 truncated=True & terminated=False。
        Part B：手动将 agent 置于终点附近，执行最后一步，断言 terminated=True & truncated=False。

        输入:
          Part A: MazeEnv(grid_size=6, obstacle_density=0.0, seed=0, max_steps=50)
                  反复 step(0)（持续撞上边界墙）× 50
          Part B: 同一环境 reset()，通过合法移动引导 agent 到终点 (4,4)

        期望:
          Part A: truncated is True, terminated is False
          Part B: terminated is True, truncated is False
        """
        # ── Part A：步数耗尽 ────────────────────────────────────────────
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0, max_steps=50)
        env.reset()

        terminated = truncated = False
        for _ in range(50):
            _, _, terminated, truncated, _ = env.step(0)  # 持续撞边界墙

        assert truncated  is True,  "步数耗尽（max_steps=50）时，truncated 应为 True"
        assert terminated is False, "步数耗尽但未到终点时，terminated 应为 False"

        # ── Part B：手动引导到终点 ──────────────────────────────────────
        env2 = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0, max_steps=200)
        env2.reset()

        # 从 (1,1) 右移 3 步到 (1,4)，再下移 3 步到 (4,4) = goal
        for _ in range(3):
            env2.step(3)   # 右
        for _ in range(2):
            env2.step(1)   # 下
        _, _, terminated, truncated, info = env2.step(1)  # 到达 (4,4)

        assert terminated is True,  "到达终点时，terminated 应为 True"
        assert truncated  is False, "到达终点时，truncated 必须为 False（严格互斥）"
        assert info["success"] is True, "到达终点后 success 应为 True"


# ──────────────────────────────────────────────────────────────────────────────
# TC-R4  test_seeding_reproducibility
# ──────────────────────────────────────────────────────────────────────────────

class TestSeedingReproducibility:
    """TC-R4：相同 seed=42 的两个独立实例产生完全相同的地图、起点、终点。"""

    @pytest.mark.unit
    def test_seeding_reproducibility(self) -> None:
        """用相同 seed=42 初始化两个独立 MazeEnv 实例，分别 reset()，
        断言地图矩阵、agent 位置、goal 位置完全一致。

        输入:
          env_a = MazeEnv(grid_size=10, obstacle_density=0.3, seed=42)
          env_b = MazeEnv(grid_size=10, obstacle_density=0.3, seed=42)
          各调用 reset()

        期望:
          np.array_equal(obs_a[0], obs_b[0])  — 地图完全一致
          info_a["agent_pos"] == info_b["agent_pos"]
          info_a["goal_pos"]  == info_b["goal_pos"]
          np.array_equal(obs_a, obs_b)         — 三通道全部一致
        """
        env_a = MazeEnv(grid_size=10, obstacle_density=0.3, seed=42)
        env_b = MazeEnv(grid_size=10, obstacle_density=0.3, seed=42)

        obs_a, info_a = env_a.reset()
        obs_b, info_b = env_b.reset()

        assert np.array_equal(obs_a[0], obs_b[0]), \
            "相同 seed 的两个实例应产生完全相同的墙壁地图（obs[0]）"
        assert info_a["agent_pos"] == info_b["agent_pos"], \
            "相同 seed 的两个实例应产生相同的起点"
        assert info_a["goal_pos"]  == info_b["goal_pos"], \
            "相同 seed 的两个实例应产生相同的终点"
        assert np.array_equal(obs_a, obs_b), \
            "相同 seed 的两个实例整体观测（三通道）应完全一致"
