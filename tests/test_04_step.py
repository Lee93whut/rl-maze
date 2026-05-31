"""
测试模块 04 —— step() 基础行为

需求覆盖
--------
* R5：step 返回 (obs, reward, terminated, truncated, info)
* R6：撞墙惩罚与位置不变
* R7：普通移动奖励
* R8：到达终点奖励与 terminated=True

对应用例
--------
TC-11, TC-12, TC-13
"""

from __future__ import annotations

import numpy as np
import pytest

from maze_env import MazeEnv


class TestStep:
    """验证 step() 的返回格式、奖励计算与位置更新。"""

    # ------------------------------------------------------------------ #
    # TC-11  撞墙行为                                                       #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_wall_hit_reward(self, env_zero_rewards: MazeEnv) -> None:
        """TC-11a：agent 在 (1,1)，向上（action=0）必然撞上边界墙，reward=-10。

        输入:  env_zero_rewards.reset()，action=0（上）
        期望:  reward == -10.0，agent_pos 不变 == (1,1)
        实测:  step() 返回值
        """
        env_zero_rewards.reset()
        _, reward, _, _, info = env_zero_rewards.step(0)  # 0 = UP
        assert reward == -11.0, f"撞墙奖励应为 reward_step(-1)+reward_wall_hit(-10)=-11，实际 {reward}"
        assert info["agent_pos"] == (1, 1), "撞墙后位置不应改变"

    @pytest.mark.unit
    def test_wall_hit_count_increments(self, env_zero_rewards: MazeEnv) -> None:
        """TC-11b：撞墙后 hit_wall_count 加 1。

        输入:  reset()，action=0（撞边界墙）
        期望:  info["hit_wall_count"] == 1
        实测:  step() 返回的 info
        """
        env_zero_rewards.reset()
        _, _, _, _, info = env_zero_rewards.step(0)
        assert info["hit_wall_count"] == 1

    @pytest.mark.unit
    def test_wall_hit_obs_shape(self, env_zero_rewards: MazeEnv) -> None:
        """TC-11c：撞墙后 obs shape 仍为 (4, 6, 6)。

        输入:  reset()，action=0
        期望:  obs.shape == (4, 6, 6)
        实测:  step() 返回的 obs
        """
        env_zero_rewards.reset()
        obs, _, _, _, _ = env_zero_rewards.step(0)
        assert obs.shape == (4, 6, 6)

    # ------------------------------------------------------------------ #
    # TC-12  普通移动行为                                                   #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_normal_move_reward(self, env_zero_rewards: MazeEnv) -> None:
        """TC-12a：向右（action=3）从 (1,1) 移动到 (1,2)，reward=-1。

        输入:  reset()，action=3（右）
        期望:  reward == -1.0，agent_pos == (1,2)
        实测:  step() 返回值
        """
        env_zero_rewards.reset()
        _, reward, _, _, info = env_zero_rewards.step(3)  # 3 = RIGHT
        assert reward == -1.0, f"正常移动奖励应为 -1，实际 {reward}"
        assert info["agent_pos"] == (1, 2), "移动后位置应为 (1,2)"

    @pytest.mark.unit
    def test_normal_move_agent_channel(self, env_zero_rewards: MazeEnv) -> None:
        """TC-12b：移动后 obs[1] 在新位置激活，旧位置清零。

        输入:  reset()，action=3（右，到 (1,2)）
        期望:  obs[1][1,2] == 1.0，obs[1][1,1] == 0.0
        实测:  obs[1] 通道值
        """
        env_zero_rewards.reset()
        obs, _, _, _, info = env_zero_rewards.step(3)
        ar, ac = info["agent_pos"]
        assert obs[1, ar, ac] == 1.0
        assert float(obs[1].sum()) == 1.0

    @pytest.mark.unit
    def test_normal_move_step_count(self, env_zero_rewards: MazeEnv) -> None:
        """TC-12c：每次 step（含撞墙）step_count 加 1。

        输入:  reset()，连续调用两次 step()
        期望:  第一次后 step_count==1，第二次后 step_count==2
        实测:  info["step_count"]
        """
        env_zero_rewards.reset()
        _, _, _, _, info1 = env_zero_rewards.step(3)
        assert info1["step_count"] == 1
        _, _, _, _, info2 = env_zero_rewards.step(3)
        assert info2["step_count"] == 2

    # ------------------------------------------------------------------ #
    # TC-13  到达终点                                                       #
    # ------------------------------------------------------------------ #

    @pytest.mark.integration
    def test_reach_goal_terminated(self, env_zero_rewards: MazeEnv) -> None:
        """TC-13a：agent 走到 (N-2,N-2) 时，terminated=True，reward 含 +100。

        输入:  grid=6，无障碍，seed=0；手动引导 agent 到 (4,4)
        期望:  终点步的 terminated==True，truncated==False，
               reward == 100 + (-1) == 99.0
        实测:  step() 返回值
        """
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0,
                      reward_goal=100.0, reward_wall_hit=-10.0, reward_step=-1.0)
        env.reset()
        # 从 (1,1) 向右走到 (1,4)，再向下走到 (4,4)
        # 右：action=3，下：action=1
        for _ in range(3):   # (1,1)->(1,2)->(1,3)->(1,4)
            env.step(3)
        for _ in range(3):   # (1,4)->(2,4)->(3,4)->(4,4)
            _, reward, terminated, truncated, info = env.step(1)

        assert terminated is True,  "到达终点应 terminated=True"
        assert truncated  is False, "到达终点不应 truncated"
        assert reward == 99.0,      f"终点奖励应为 99.0，实际 {reward}"
        assert info["success"] is True

    @pytest.mark.integration
    def test_reach_goal_agent_channel(self, env_zero_rewards: MazeEnv) -> None:
        """TC-13b：到达终点后，obs[1] 与 obs[2] 在同一位置均为 1.0。

        输入:  grid=6，引导 agent 至 goal (4,4)
        期望:  obs[1][4,4]==1.0，obs[2][4,4]==1.0
        实测:  终点步 obs 通道值
        """
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        env.reset()
        for _ in range(3):
            env.step(3)
        for _ in range(2):
            env.step(1)
        obs, _, _, _, _ = env.step(1)  # 最后一步到达 (4,4)
        assert obs[1, 4, 4] == 1.0
        assert obs[2, 4, 4] == 1.0
