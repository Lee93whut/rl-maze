"""maze_env.bfs —— 广度优先搜索（BFS）寻路算法

职责
----
* 在二维迷宫网格上求解从 ``start`` 到 ``end`` 的**最短路径**。
* 作为自动化标注器，为强化学习训练生成"最短路径真值（Ground Truth）"。
* 作为 Baseline，与 RL Agent 的解路径在步数、成功率上进行对比。
* 被 :mod:`maze_env.generator` 复用，提供连通性验证（取代重复实现）。

归属说明
--------
BFS 是迷宫环境包的底层工具。它既服务于环境内部（连通性检查），
也服务于训练脚本（POR 计算）和 Web App（最短路展示）。
因此正确归属是 ``maze_env`` 包，而非上层的 ``src/`` 脚本目录。

输入约定
--------
* ``grid``  : 形状 ``(H, W)`` 的二维 NumPy 数组（int 或 float 均可）。
              ``0`` 表示可通行，非零表示墙壁。
* ``start`` : ``(row, col)`` 起点坐标。
* ``end``   : ``(row, col)`` 终点坐标。

输出约定
--------
返回标准字典，字段如下：

.. code-block:: python

    {
        "success":            bool,
        "path":               List[Tuple[int, int]],
        "steps":              int,          # len(path)-1；无解为 -1
        "execution_time_ms":  float,
    }

特殊情形处理
-----------
+-----------------------------------+----------+-------+-----------+
| 情形                               | success  | steps | path      |
+===================================+==========+=======+===========+
| 起点 == 终点                        | True     | 0     | [start]   |
+-----------------------------------+----------+-------+-----------+
| 起点或终点本身是墙（非法输入）        | False    | -1    | []        |
+-----------------------------------+----------+-------+-----------+
| 存在路径                            | True     | N>0   | 完整轨迹  |
+-----------------------------------+----------+-------+-----------+
| 无路径（被墙封死）                    | False    | -1    | []        |
+-----------------------------------+----------+-------+-----------+

Example::

    import numpy as np
    from maze_env.bfs import bfs

    grid = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])
    result = bfs(grid, start=(1, 1), end=(3, 3))
    print(result["success"])   # True
    print(result["steps"])     # 4
"""

from __future__ import annotations

import time
from collections import deque
from typing import TypedDict

import numpy as np

from maze_env.actions import DELTAS as _DELTAS

# 标准输出类型（TypedDict 让调用方的静态检查直接感知各字段类型）
class BFSResult(TypedDict):
    success:           bool
    path:              list[tuple[int, int]]
    steps:             int
    execution_time_ms: float


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def bfs(
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> BFSResult:
    """在二维迷宫网格上执行广度优先搜索，返回最短路径。

    BFS 保证：在无权图（等步代价）上找到的第一条路径即为最短路径。

    Args:
        grid:  形状 ``(H, W)`` 的二维 NumPy 数组。``0`` = 可通行，非零 = 墙壁。
        start: 起点坐标 ``(row, col)``。
        end:   终点坐标 ``(row, col)``。

    Returns:
        标准结果字典，含 ``success``、``path``、``steps``、
        ``execution_time_ms`` 四个字段，详见模块文档。

    Raises:
        TypeError:  若 ``grid`` 不是 ``numpy.ndarray``。
        ValueError: 若 ``grid`` 不是二维数组，或坐标越界。
    """
    t_start = time.perf_counter()

    _validate_inputs(grid, start, end)

    if _is_wall(grid, start) or _is_wall(grid, end):
        return _make_result(success=False, path=[], steps=-1,
                            elapsed_ms=_elapsed_ms(t_start))

    if start == end:
        return _make_result(success=True, path=[start], steps=0,
                            elapsed_ms=_elapsed_ms(t_start))

    H, W = grid.shape
    visited: set[tuple[int, int]] = {start}
    queue: deque[tuple[int, int]] = deque([start])
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    found = False
    while queue:
        cur = queue.popleft()
        for dr, dc in _DELTAS:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if grid[nr, nc] != 0:
                continue
            if nxt in visited:
                continue
            visited.add(nxt)
            came_from[nxt] = cur
            if nxt == end:
                found = True
                break
            queue.append(nxt)
        if found:
            break

    if not found:
        return _make_result(success=False, path=[], steps=-1,
                            elapsed_ms=_elapsed_ms(t_start))

    path = _reconstruct_path(came_from, start, end)
    return _make_result(success=True, path=path, steps=len(path) - 1,
                        elapsed_ms=_elapsed_ms(t_start))


# ---------------------------------------------------------------------------
# 私有辅助函数
# ---------------------------------------------------------------------------

def _validate_inputs(
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    if not isinstance(grid, np.ndarray):
        raise TypeError(f"grid 必须是 numpy.ndarray，实际类型：{type(grid)}")
    if grid.ndim != 2:
        raise ValueError(f"grid 必须是二维数组，实际维度：{grid.ndim}")
    H, W = grid.shape
    for name, (r, c) in (("start", start), ("end", end)):
        if not (0 <= r < H and 0 <= c < W):
            raise ValueError(f"{name}={( r, c)} 越界，grid 尺寸为 ({H}, {W})")


def _is_wall(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    return bool(grid[pos[0], pos[1]] != 0)


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    node = end
    while node != start:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path


def _elapsed_ms(t_start: float) -> float:
    return (time.perf_counter() - t_start) * 1_000.0


def _make_result(
    *,
    success: bool,
    path: list[tuple[int, int]],
    steps: int,
    elapsed_ms: float,
) -> BFSResult:
    return BFSResult(
        success=success,
        path=path,
        steps=steps,
        execution_time_ms=round(elapsed_ms, 6),
    )
