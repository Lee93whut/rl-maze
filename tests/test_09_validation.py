"""
测试模块 09 —— 参数校验（ValueError）

需求覆盖
--------
* R13：非法参数应抛出 ValueError，并提供有意义的错误信息

对应用例
--------
TC-22, TC-23, TC-24, TC-25
"""

from __future__ import annotations

import pytest

from maze_env import MazeEnv


class TestValidation:
    """验证非法构造参数触发 ValueError。"""

    # ------------------------------------------------------------------ #
    # TC-22  grid_size 过小                                                #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_grid_size_too_small(self) -> None:
        """TC-22：grid_size < 4 应抛出 ValueError。

        输入:  MazeEnv(grid_size=3)
        期望:  ValueError
        实测:  pytest.raises
        """
        with pytest.raises(ValueError, match="grid_size"):
            MazeEnv(grid_size=3)

    # ------------------------------------------------------------------ #
    # TC-23  obstacle_density 超界                                         #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_obstacle_density_negative(self) -> None:
        """TC-23a：obstacle_density < 0 应抛出 ValueError。

        输入:  MazeEnv(obstacle_density=-0.1)
        期望:  ValueError
        实测:  pytest.raises
        """
        with pytest.raises(ValueError, match="obstacle_density"):
            MazeEnv(obstacle_density=-0.1)

    @pytest.mark.unit
    def test_obstacle_density_too_high(self) -> None:
        """TC-23b：obstacle_density >= 1.0 应抛出 ValueError。

        输入:  MazeEnv(obstacle_density=1.0)
        期望:  ValueError
        实测:  pytest.raises
        """
        with pytest.raises(ValueError, match="obstacle_density"):
            MazeEnv(obstacle_density=1.0)

    # ------------------------------------------------------------------ #
    # TC-24  max_steps 非正数                                              #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_max_steps_zero(self) -> None:
        """TC-24a：max_steps == 0 应抛出 ValueError。

        输入:  MazeEnv(max_steps=0)
        期望:  ValueError
        实测:  pytest.raises
        """
        with pytest.raises(ValueError, match="max_steps"):
            MazeEnv(max_steps=0)

    @pytest.mark.unit
    def test_max_steps_negative(self) -> None:
        """TC-24b：max_steps < 0 应抛出 ValueError。

        输入:  MazeEnv(max_steps=-1)
        期望:  ValueError
        实测:  pytest.raises
        """
        with pytest.raises(ValueError, match="max_steps"):
            MazeEnv(max_steps=-1)

    # ------------------------------------------------------------------ #
    # TC-25  action_space 越界                                             #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_invalid_action_raises(self, env_zero: MazeEnv) -> None:
        """TC-25：传入无效 action（如 4）应抛出异常。

        输入:  reset()，step(4)
        期望:  抛出 ValueError 或 AssertionError（实现可选）
        实测:  pytest.raises
        """
        env_zero.reset()
        with pytest.raises((ValueError, AssertionError)):
            env_zero.step(4)

    # ------------------------------------------------------------------ #
    # TC-26  render_mode 非法值                                            #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_invalid_render_mode(self) -> None:
        """TC-26：不支持的 render_mode 应抛出 ValueError。

        输入:  MazeEnv(render_mode='invalid')
        期望:  ValueError，消息含 'render_mode'
        实测:  pytest.raises
        """
        with pytest.raises(ValueError, match="render_mode"):
            MazeEnv(grid_size=6, render_mode="invalid")
