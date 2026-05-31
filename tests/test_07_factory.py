"""
测试模块 07 —— 工厂方法

需求覆盖
--------
* RF2：from_config(cfg_dict) 正确映射所有参数
* RF4：from_yaml(path) 读取 config.yaml 并构造实例

对应用例
--------
TC-18, TC-19
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from maze_env import MazeEnv


class TestFactory:
    """验证 from_config() 与 from_yaml() 工厂方法。"""

    # ------------------------------------------------------------------ #
    # TC-18  from_config                                                   #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_from_config_grid_size(self, cfg_dict: dict) -> None:
        """TC-18a：from_config 正确映射 grid_size=12。

        输入:  cfg_dict（grid_size=12）
        期望:  env.observation_space.shape == (4, 12, 12)
        实测:  observation_space.shape
        """
        env = MazeEnv.from_config(cfg_dict)
        assert env.observation_space.shape == (4, 12, 12)

    @pytest.mark.unit
    def test_from_config_reward_goal(self, cfg_dict: dict) -> None:
        """TC-18b：from_config 正确映射 reward_goal=200。

        输入:  cfg_dict（goal=200）
        期望:  env.reward_goal == 200.0
        实测:  env.reward_goal 属性
        """
        env = MazeEnv.from_config(cfg_dict)
        assert env.reward_goal == 200.0

    @pytest.mark.unit
    def test_from_config_reward_wall_hit(self, cfg_dict: dict) -> None:
        """TC-18c：from_config 正确映射 reward_wall_hit=-5。

        输入:  cfg_dict（wall_hit=-5）
        期望:  env.reward_wall_hit == -5.0
        实测:  env.reward_wall_hit 属性
        """
        env = MazeEnv.from_config(cfg_dict)
        assert env.reward_wall_hit == -5.0

    @pytest.mark.unit
    def test_from_config_reward_step(self, cfg_dict: dict) -> None:
        """TC-18d：from_config 正确映射 reward_step=-2。

        输入:  cfg_dict（step=-2）
        期望:  env.reward_step == -2.0
        实测:  env.reward_step 属性
        """
        env = MazeEnv.from_config(cfg_dict)
        assert env.reward_step == -2.0

    @pytest.mark.unit
    def test_from_config_max_steps(self, cfg_dict: dict) -> None:
        """TC-18e：from_config 正确映射 max_steps=300。

        输入:  cfg_dict（max_steps=300）
        期望:  env.max_steps == 300
        实测:  env.max_steps 属性
        """
        env = MazeEnv.from_config(cfg_dict)
        assert env.max_steps == 300

    @pytest.mark.unit
    def test_from_config_reset_works(self, cfg_dict: dict) -> None:
        """TC-18f：from_config 构造的环境可以正常 reset。

        输入:  from_config(cfg_dict).reset()
        期望:  obs.shape == (4, 12, 12)，不抛出异常
        实测:  reset() 返回值
        """
        env = MazeEnv.from_config(cfg_dict)
        obs, info = env.reset()
        assert obs.shape == (4, 12, 12)

    # ------------------------------------------------------------------ #
    # TC-19  from_yaml                                                     #
    # ------------------------------------------------------------------ #

    @pytest.mark.integration
    def test_from_yaml_loads(self, config_yaml_path: Path) -> None:
        """TC-19a：from_yaml 读取 config.yaml 不抛出异常。

        输入:  config.yaml 文件路径
        期望:  成功返回 MazeEnv 实例
        实测:  isinstance 检查
        """
        env = MazeEnv.from_yaml(config_yaml_path)
        assert isinstance(env, MazeEnv)

    @pytest.mark.integration
    def test_from_yaml_matches_config(self, config_yaml_path: Path) -> None:
        """TC-19b：from_yaml 参数与 config.yaml 内容一致。

        输入:  config.yaml
        期望:  env.observation_space.shape[1] == cfg["maze"]["grid_size"]
               env.max_steps == cfg["maze"]["max_steps"]
               env.reward_goal == cfg["rewards"]["goal"]
        实测:  属性值（动态读取，不硬编码）
        """
        with open(config_yaml_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        expected_grid_size = cfg["maze"]["grid_size"]
        expected_max_steps = cfg["maze"]["max_steps"]
        expected_goal      = float(cfg["rewards"]["goal"])

        env = MazeEnv.from_yaml(config_yaml_path)
        assert env.observation_space.shape[1] == expected_grid_size, \
            f"grid_size 应为 {expected_grid_size}"
        assert env.max_steps == expected_max_steps, \
            f"max_steps 应为 {expected_max_steps}（来自 config.yaml），实际 {env.max_steps}"
        assert env.reward_goal == expected_goal, \
            f"reward_goal 应为 {expected_goal}"

    @pytest.mark.integration
    def test_from_yaml_reset_works(self, config_yaml_path: Path) -> None:
        """TC-19c：from_yaml 构造的环境可以正常 reset 并产生正确形状的 obs。

        输入:  from_yaml(config_yaml_path).reset()
        期望:  obs.shape == (4, 10, 10)
        实测:  reset() 返回值
        """
        env = MazeEnv.from_yaml(config_yaml_path)
        obs, _ = env.reset()
        assert obs.shape == (4, 10, 10)
