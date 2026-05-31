"""
测试模块 00 —— Gymnasium 官方 API 合规性验证

需求覆盖
--------
* Gymnasium check_env：接口规范完整性（obs/action space 声明、
  reset/step 返回格式、dtype、边界）

对应用例
--------
TC-00
"""

from __future__ import annotations

import pytest
from gymnasium.utils.env_checker import check_env

from maze_env import MazeEnv


class TestGymnasiumCompliance:
    """使用 Gymnasium 官方 check_env 工具验证接口合规性。"""

    @pytest.mark.integration
    def test_gymnasium_api_compliance(self) -> None:
        """TC-00：check_env 全量扫描，不应触发任何 Error 级警告。

        输入:  MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        期望:  check_env() 正常返回，不抛出异常
        实测:  gymnasium.utils.env_checker.check_env
        """
        env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        # warn=True：将 Gymnasium 的 UserWarning 转换为可见警告（不阻断）
        # skip_render_check=True：render_mode=None 时跳过渲染检查
        check_env(env, warn=True, skip_render_check=True)
        env.close()
