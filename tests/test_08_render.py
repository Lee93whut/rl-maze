"""
测试模块 08 —— render() 输出

需求覆盖
--------
* R11：render(mode='ansi') 返回含 A（agent）和 G（goal）标记的字符串
* R12：render 头部信息含 HitWalls 计数

对应用例
--------
TC-20, TC-21
"""

from __future__ import annotations

import pytest

from maze_env import MazeEnv


def _ansi_env() -> MazeEnv:
    """返回 render_mode='ansi' 的确定性零障碍环境（grid=6，seed=0）。"""
    return MazeEnv(grid_size=6, obstacle_density=0.0, seed=0, render_mode="ansi")


class TestRender:
    """验证 render('ansi') 输出格式与关键标记。"""

    # ------------------------------------------------------------------ #
    # TC-20  A / G 标记                                                    #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_render_contains_agent_marker(self) -> None:
        """TC-20a：render 输出包含 'A'（agent 标记）。

        输入:  render_mode='ansi'，reset()，render()
        期望:  'A' in rendered_str
        实测:  返回字符串内容
        """
        env = _ansi_env()
        env.reset()
        rendered = env.render()
        assert "A" in rendered, "render 输出应包含 agent 标记 'A'"

    @pytest.mark.unit
    def test_render_contains_goal_marker(self) -> None:
        """TC-20b：render 输出包含 'G'（goal 标记）。

        输入:  render_mode='ansi'，reset()，render()
        期望:  'G' in rendered_str
        实测:  返回字符串内容
        """
        env = _ansi_env()
        env.reset()
        rendered = env.render()
        assert "G" in rendered, "render 输出应包含 goal 标记 'G'"

    @pytest.mark.unit
    def test_render_returns_string(self) -> None:
        """TC-20c：render 返回值类型为 str（ansi 模式）。

        输入:  render_mode='ansi'，reset()，render()
        期望:  isinstance(rendered, str)
        实测:  type 检查
        """
        env = _ansi_env()
        env.reset()
        rendered = env.render()
        assert isinstance(rendered, str), "render 应返回字符串"

    # ------------------------------------------------------------------ #
    # TC-21  头部 HitWalls 信息                                            #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_render_header_contains_hitwalls(self) -> None:
        """TC-21a：render 头部含 'WallHits' 相关信息。

        输入:  render_mode='ansi'，reset()，撞墙 1 次，render()
        期望:  render 输出含 'wall'（大小写不敏感）
        实测:  rendered.lower()
        """
        env = _ansi_env()
        env.reset()
        env.step(0)   # 撞上边界墙
        rendered = env.render()
        assert "wall" in rendered.lower(), \
            "render 头部应包含 WallHits 相关信息"

    @pytest.mark.unit
    def test_render_header_hitwalls_count(self) -> None:
        """TC-21b：render 头部 WallHits 数值与实际撞墙次数一致。

        输入:  render_mode='ansi'，reset()，撞墙 2 次，render()
        期望:  render 输出首行含 '2'
        实测:  rendered 首行内容
        """
        env = _ansi_env()
        env.reset()
        env.step(0)
        env.step(0)
        rendered = env.render()
        first_line = rendered.split("\n")[0]
        assert "2" in first_line, \
            f"render 头部应显示 WallHits=2，实际首行：{first_line!r}"

    @pytest.mark.unit
    def test_render_after_move_a_moved(self) -> None:
        """TC-21c：移动后，'A' 标记仍存在于 render 输出中。

        输入:  render_mode='ansi'，reset()，右移一步到 (1,2)，render()
        期望:  'A' in rendered
        实测:  字符串包含检查
        """
        env = _ansi_env()
        env.reset()
        env.step(3)   # 右移
        rendered = env.render()
        assert "A" in rendered

    # ------------------------------------------------------------------ #
    # TC-22  human 模式与 None 模式                                        #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_render_human_returns_none(self, capsys) -> None:
        """TC-22a：render_mode='human' 时 render() 打印到 stdout 并返回 None。

        输入:  render_mode='human'，reset()，render()
        期望:  返回值为 None，stdout 非空
        实测:  返回值类型 + capsys 捕获
        """
        env = MazeEnv(grid_size=5, obstacle_density=0.0, seed=0, render_mode="human")
        env.reset()
        result = env.render()
        captured = capsys.readouterr()
        assert result is None, "human 模式 render() 应返回 None"
        assert len(captured.out) > 0, "human 模式应向 stdout 打印内容"

    @pytest.mark.unit
    def test_render_none_mode_returns_none(self) -> None:
        """TC-22b：render_mode=None 时 render() 直接返回 None，无输出。

        输入:  MazeEnv()（默认 render_mode=None），reset()，render()
        期望:  None
        实测:  返回值
        """
        env = MazeEnv(grid_size=5, obstacle_density=0.0, seed=0)
        env.reset()
        assert env.render() is None

    @pytest.mark.unit
    def test_step_triggers_human_render(self, capsys) -> None:
        """TC-22c：render_mode='human' 时，step() 自动调用 render() 打印输出。

        输入:  render_mode='human'，reset()，step(右移)
        期望:  stdout 非空（step 内部调用 render）
        实测:  capsys.readouterr().out
        """
        env = MazeEnv(grid_size=5, obstacle_density=0.0, seed=0, render_mode="human")
        env.reset()
        capsys.readouterr()   # 清空 reset 阶段的可能输出
        env.step(3)           # 右移；render_mode='human' 时 step 末尾自动 render
        captured = capsys.readouterr()
        assert len(captured.out) > 0, "human 模式 step() 应自动触发 render 输出"

    # ------------------------------------------------------------------ #
    # TC-23  close() 方法                                                  #
    # ------------------------------------------------------------------ #

    @pytest.mark.unit
    def test_close_does_not_raise(self) -> None:
        """TC-23：close() 调用不抛出任何异常。

        输入:  reset()，close()
        期望:  无异常
        实测:  直接调用
        """
        env = MazeEnv(grid_size=5, obstacle_density=0.0, seed=0)
        env.reset()
        env.close()   # 无外部资源，不应抛出
