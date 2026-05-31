"""
共享 Fixtures —— 所有测试模块自动加载，无需显式 import。

Fixtures 说明
-------------
env_zero : MazeEnv
    grid_size=6, obstacle_density=0（无随机障碍），seed=0。
    适合需要确定性地图的单步行为测试。

env_zero_rewards : MazeEnv
    同上，并显式设置奖励参数（goal=100, wall_hit=-10, step=-1），
    方便断言具体数值。

env_dense : MazeEnv
    grid_size=10, obstacle_density=0.45，用于压力测试连通性。

cfg_dict : dict
    模拟 from_config() 输入的完整配置字典。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maze_env import MazeEnv


@pytest.fixture
def env_zero() -> MazeEnv:
    """确定性零障碍环境，grid=6，seed=0。"""
    return MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)


@pytest.fixture
def env_zero_rewards() -> MazeEnv:
    """确定性零障碍环境，显式奖励参数。"""
    return MazeEnv(
        grid_size=6,
        obstacle_density=0.0,
        seed=0,
        reward_goal=100.0,
        reward_wall_hit=-10.0,
        reward_step=-1.0,
    )


@pytest.fixture
def env_dense() -> MazeEnv:
    """高密度障碍环境，用于连通性压力测试。"""
    return MazeEnv(grid_size=10, obstacle_density=0.45)


@pytest.fixture
def cfg_dict() -> dict:
    """完整配置字典，模拟 yaml.safe_load 输出。"""
    return {
        "maze": {
            "grid_size": 12,
            "obstacle_density": 0.2,
            "max_steps": 300,
            "seed": 7,
        },
        "rewards": {
            "goal": 200,
            "wall_hit": -5,
            "step": -2,
        },
    }


@pytest.fixture
def config_yaml_path() -> Path:
    """返回 config.yaml 的绝对路径。"""
    return Path(__file__).parent.parent / "config.yaml"
