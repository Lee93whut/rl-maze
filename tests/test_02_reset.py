"""
测试模块 02 —— reset() 行为

需求覆盖
--------
* R1：seed 参数控制可复现性
* R2：三通道观测编码语义
* R4：reset 返回 (obs, info)
* RF1：BFS 连通性保证（起终点强制可达）
* RF3：info 新增字段初值

对应用例
--------
TC-03, TC-04, TC-05, TC-06, TC-07
"""

from __future__ import annotations

import numpy as np
import pytest

from maze_env import MazeEnv


class TestReset:
    """验证 reset() 的返回值格式、通道语义与 info 初始值。"""

    # ------------------------------------------------------------------ #
    # TC-03  返回值格式                                                     #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_reset_obs_shape(self, env_zero: MazeEnv) -> None:
        """TC-03a：reset() 返回 obs，形状为 (4, N, N)。

        输入:  env_zero.reset()，seed=0
        期望:  obs.shape == (4, 6, 6)
        实测:  解包 reset() 返回值
        """
        obs, _ = env_zero.reset()
        assert obs.shape == (4, 6, 6)

    @pytest.mark.unit
    def test_reset_obs_dtype(self, env_zero: MazeEnv) -> None:
        """TC-03b：reset() 返回 obs，dtype 为 float32。

        输入:  env_zero.reset()
        期望:  obs.dtype == float32
        """
        obs, _ = env_zero.reset()
        assert obs.dtype == np.float32

    @pytest.mark.unit
    def test_reset_info_keys(self, env_zero: MazeEnv) -> None:
        """TC-03c：info 包含所有规定字段。

        输入:  env_zero.reset()
        期望:  info.keys() 包含五个字段
        """
        _, info = env_zero.reset()
        required = {"agent_pos", "goal_pos", "step_count",
                    "hit_wall_count", "success"}
        assert required.issubset(info.keys())

    # ------------------------------------------------------------------ #
    # TC-04  边界与起终点                                                   #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_border_walls(self) -> None:
        """TC-04a：四条边界全部为墙（wall_map 四边均为 1）。

        输入:  MazeEnv(grid_size=8, obstacle_density=0.0, seed=0).reset()
        期望:  wall[0,:], wall[-1,:], wall[:,0], wall[:,-1] 全为 1.0
        实测:  obs[0]（墙壁通道）各边切片
        """
        env = MazeEnv(grid_size=8, obstacle_density=0.0, seed=0)
        obs, _ = env.reset()
        wall = obs[0]
        assert np.all(wall[0, :]  == 1.0), "上边界应全为墙"
        assert np.all(wall[-1, :] == 1.0), "下边界应全为墙"
        assert np.all(wall[:, 0]  == 1.0), "左边界应全为墙"
        assert np.all(wall[:, -1] == 1.0), "右边界应全为墙"

    @pytest.mark.unit
    def test_start_goal_not_wall(self) -> None:
        """TC-04b：起点 (1,1) 与终点 (N-2,N-2) 永远不为墙。

        输入:  MazeEnv(grid_size=8, obstacle_density=0.0, seed=0).reset()
        期望:  wall[1,1] == 0.0，wall[6,6] == 0.0
        实测:  obs[0] 对应坐标处的值
        """
        env = MazeEnv(grid_size=8, obstacle_density=0.0, seed=0)
        obs, _ = env.reset()
        N, wall = 8, obs[0]
        assert wall[1, 1]       == 0.0, "起点 (1,1) 不应为墙"
        assert wall[N-2, N-2]   == 0.0, "终点 (N-2,N-2) 不应为墙"

    # ------------------------------------------------------------------ #
    # TC-05  Agent 通道                                                     #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_agent_channel_position(self, env_zero: MazeEnv) -> None:
        """TC-05a：obs[1] 在 agent_pos 处为 1.0。

        输入:  env_zero.reset()，agent_pos=(1,1)
        期望:  obs[1][1,1] == 1.0
        实测:  obs[1][ar, ac]
        """
        obs, info = env_zero.reset()
        ar, ac = info["agent_pos"]
        assert obs[1, ar, ac] == 1.0

    @pytest.mark.unit
    def test_agent_channel_unique(self, env_zero: MazeEnv) -> None:
        """TC-05b：obs[1] 全图仅一个激活格（sum == 1.0）。

        输入:  env_zero.reset()
        期望:  obs[1].sum() == 1.0
        实测:  np.sum(obs[1])
        """
        obs, _ = env_zero.reset()
        assert float(obs[1].sum()) == 1.0

    # ------------------------------------------------------------------ #
    # TC-06  终点通道                                                       #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_goal_channel_position(self, env_zero: MazeEnv) -> None:
        """TC-06a：obs[2] 在 goal_pos 处为 1.0，grid=6 时 goal=(4,4)。

        输入:  env_zero.reset()
        期望:  obs[2][4,4] == 1.0，obs[2].sum() == 1.0
        实测:  obs[2][gr, gc] 及 sum
        """
        obs, info = env_zero.reset()
        gr, gc = info["goal_pos"]
        assert obs[2, gr, gc] == 1.0
        assert float(obs[2].sum()) == 1.0

    # ------------------------------------------------------------------ #
    # TC-07  info 初始值                                                    #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_info_initial_values(self, env_zero: MazeEnv) -> None:
        """TC-07：reset() 后 info 所有幕级统计量为初始值。

        输入:  env_zero.reset()
        期望:
          agent_pos == (1, 1)
          goal_pos  == (4, 4)（N=6 时 N-2=4）
          step_count     == 0
          hit_wall_count == 0
          success        == False
        实测:  info 字典各字段
        """
        _, info = env_zero.reset()
        assert info["agent_pos"]       == (1, 1)
        assert info["goal_pos"]        == (4, 4)
        assert info["step_count"]      == 0
        assert info["hit_wall_count"]  == 0
        assert info["success"]         is False


# ======================================================================
# TC-14  外部注入地图（options["wall_map"]）
# ======================================================================

class TestResetWallMapInjection:
    """验证 reset(options={'wall_map': ...}) 外部注入地图路径。"""

    @pytest.mark.unit
    def test_inject_wall_map_used(self) -> None:
        """TC-14a：注入自定义 wall_map 后，env._wall_map 与注入值一致。

        输入:  全零 6×6 wall_map（无障碍）
        期望:  env.wall_map 与注入值匹配
        实测:  np.array_equal
        """
        import numpy as np
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        custom_map = np.zeros((6, 6), dtype=np.float32)
        # 添加边界墙（真实使用场景）
        custom_map[0, :] = 1.0
        custom_map[-1, :] = 1.0
        custom_map[:, 0] = 1.0
        custom_map[:, -1] = 1.0
        env.reset(options={"wall_map": custom_map})
        assert np.array_equal(env.wall_map, custom_map), \
            "注入的 wall_map 应被环境直接使用"

    @pytest.mark.unit
    def test_inject_wall_map_wrong_shape_raises(self) -> None:
        """TC-14b：注入形状不匹配的 wall_map 应抛出 ValueError。

        输入:  grid_size=6 的环境，注入 5×5 wall_map
        期望:  ValueError
        实测:  pytest.raises
        """
        import numpy as np
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        bad_map = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="wall_map"):
            env.reset(options={"wall_map": bad_map})

    @pytest.mark.unit
    def test_inject_custom_start_goal(self) -> None:
        """TC-14c：通过 options 覆盖 start / goal 坐标生效。

        输入:  options={'start': (1,1), 'goal': (2,2)}
        期望:  info['agent_pos'] == (1,1)，info['goal_pos'] == (2,2)
        实测:  info 字段
        """
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        _, info = env.reset(options={"start": (1, 1), "goal": (2, 2)})
        assert info["agent_pos"] == (1, 1)
        assert info["goal_pos"]  == (2, 2)


# ======================================================================
# TC-15  只读属性（wall_map / goal_pos / agent_pos）
# ======================================================================

class TestReadOnlyProperties:
    """验证环境暴露的只读属性行为。"""

    @pytest.mark.unit
    def test_wall_map_property_readonly(self) -> None:
        """TC-15a：wall_map 属性返回不可写视图，写入应抛出 ValueError。

        期望:  对返回的 ndarray 赋值触发 ValueError
        实测:  ValueError
        """
        import numpy as np
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        env.reset()
        wmap = env.wall_map
        with pytest.raises(ValueError):
            wmap[0, 0] = 1.0   # 写入只读视图应抛出异常

    @pytest.mark.unit
    def test_goal_pos_property(self) -> None:
        """TC-15b：goal_pos 属性返回终点坐标 tuple。

        期望:  isinstance(env.goal_pos, tuple)，值为 (N-2, N-2)
        实测:  属性值类型与内容
        """
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        env.reset()
        gp = env.goal_pos
        assert isinstance(gp, tuple)
        assert gp == (4, 4)

    @pytest.mark.unit
    def test_agent_pos_property(self) -> None:
        """TC-15c：agent_pos 属性返回 Agent 当前坐标 tuple。

        期望:  reset 后 agent_pos == (1, 1)
        实测:  属性值
        """
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        env.reset()
        assert env.agent_pos == (1, 1)
