"""
测试模块 06 —— info 统计字段

需求覆盖
--------
* RF3：info 新增字段（hit_wall_count / success）的累计语义

对应用例
--------
TC-16, TC-17
"""

from __future__ import annotations

import pytest

from maze_env import MazeEnv


class TestInfoStats:
    """验证 hit_wall_count 与 success 字段在整个幕期间的累计行为。"""

    # ------------------------------------------------------------------ #
    # TC-16  hit_wall_count 累计                                           #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_hit_wall_count_accumulates(self, env_zero_rewards: MazeEnv) -> None:
        """TC-16a：连续 3 次撞墙，hit_wall_count 应累计为 3。

        输入:  reset()，action=0（撞边界墙）× 3
        期望:  最终 info["hit_wall_count"] == 3
        实测:  每步 info["hit_wall_count"]
        """
        env_zero_rewards.reset()
        for i in range(3):
            _, _, _, _, info = env_zero_rewards.step(0)
            assert info["hit_wall_count"] == i + 1, \
                f"第 {i+1} 次撞墙后计数应为 {i+1}"

    @pytest.mark.unit
    def test_hit_wall_count_reset_between_episodes(
            self, env_zero_rewards: MazeEnv) -> None:
        """TC-16b：reset() 后 hit_wall_count 归零，不跨幕累计。

        输入:  第一幕：撞墙 2 次；reset()；再撞墙 1 次
        期望:  第二幕第 1 步后 hit_wall_count == 1（而非 3）
        实测:  info["hit_wall_count"]
        """
        env_zero_rewards.reset()
        env_zero_rewards.step(0)
        env_zero_rewards.step(0)
        env_zero_rewards.reset()
        _, _, _, _, info = env_zero_rewards.step(0)
        assert info["hit_wall_count"] == 1, \
            "reset() 后撞墙计数应从 0 重新开始"

    @pytest.mark.unit
    def test_normal_move_no_wall_count(self, env_zero_rewards: MazeEnv) -> None:
        """TC-16c：正常移动不增加 hit_wall_count。

        输入:  reset()，action=3（右，无障碍）
        期望:  info["hit_wall_count"] == 0
        实测:  step() 返回的 info
        """
        env_zero_rewards.reset()
        _, _, _, _, info = env_zero_rewards.step(3)
        assert info["hit_wall_count"] == 0

    # ------------------------------------------------------------------ #
    # TC-17  success 字段                                                  #
    # ------------------------------------------------------------------ #

    @pytest.mark.integration
    def test_success_true_on_goal(self) -> None:
        """TC-17a：到达终点后 success=True，且后续 reset 后恢复 False。

        输入:  引导 agent 到 goal；验证 success；reset()；验证 success 归零
        期望:  到达终点时 success=True，reset() 后 info["success"]=False
        实测:  step() 与 reset() 返回的 info
        """
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        env.reset()
        for _ in range(3):
            env.step(3)
        for _ in range(2):
            env.step(1)
        _, _, _, _, info = env.step(1)
        assert info["success"] is True

        # 新一幕
        _, info_new = env.reset()
        assert info_new["success"] is False, "reset() 后 success 应复位为 False"

    @pytest.mark.unit
    def test_success_false_during_episode(self, env_zero: MazeEnv) -> None:
        """TC-17b：幕中（未到终点）success 始终为 False。

        输入:  reset()，向右移动一步（未到终点）
        期望:  info["success"] is False
        实测:  step() 返回的 info
        """
        env_zero.reset()
        _, _, _, _, info = env_zero.step(3)
        assert info["success"] is False
