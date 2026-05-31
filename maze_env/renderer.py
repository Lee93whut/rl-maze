"""迷宫 ASCII 渲染器。

职责
----
* ``render_frame`` — 将环境当前状态编码为 ASCII 字符串。

符号约定
--------
* ``A`` —— Agent 当前位置。
* ``G`` —— 终点位置。
* ``█`` —— 墙壁。
* ``·`` —— 可通行空地。
"""

from __future__ import annotations

import numpy as np


def render_frame(
    wall_map: np.ndarray,
    agent_pos: tuple[int, int],
    goal_pos: tuple[int, int],
    step_count: int,
    max_steps: int,
    hit_wall_count: int,
    episode_success: bool,
) -> str:
    """将当前环境状态渲染为 ASCII 字符串。

    Args:
        wall_map:        形状 ``(N, N)`` 的墙壁图，``1.0`` 为墙。
        agent_pos:       Agent 当前坐标 ``(row, col)``。
        goal_pos:        终点坐标 ``(row, col)``。
        step_count:      本幕已执行步数。
        max_steps:       单幕最大步数上限。
        hit_wall_count:  本幕累计撞墙次数。
        episode_success: 本幕是否已到达终点。

    Returns:
        含头部统计行与网格图的多行字符串。
    """
    N = wall_map.shape[0]
    ar, ac = agent_pos
    gr, gc = goal_pos

    rows: list[str] = []
    for r in range(N):
        cells: list[str] = []
        for c in range(N):
            if (r, c) == (ar, ac):
                cells.append("A")
            elif (r, c) == (gr, gc):
                cells.append("G")
            elif wall_map[r, c] == 1.0:
                cells.append("█")
            else:
                cells.append("·")
        rows.append(" ".join(cells))

    header = (
        f"Step {step_count}/{max_steps}  "
        f"WallHits {hit_wall_count}  "
        f"Success {episode_success}"
    )
    return header + "\n" + "\n".join(rows) + "\n"
