"""迷宫地图生成与连通性验证。

职责
----
* ``generate_maze``   — 随机采样墙壁图（不检查连通性）。
* ``bfs_reachable``   — BFS 验证起点→终点可达性（复用 maze_env.bfs）。

设计约束
--------
* 所有随机操作通过传入的 ``np_random`` 句柄完成，
  严禁在此模块内使用 ``numpy.random.*`` 全局函数或 ``random`` 标准库。
* 连通性检查统一使用 :func:`maze_env.bfs.bfs`，不重复实现 BFS 逻辑。
"""

from __future__ import annotations

import numpy as np

from maze_env.bfs import bfs as _bfs


def generate_maze(
    grid_size: int,
    obstacle_density: float,
    np_random: np.random.Generator,
) -> np.ndarray:
    """随机生成一张迷宫墙壁图。

    生成规则：

    * 四条边界全部设为墙壁（``1.0``）。
    * 内部格子按 ``obstacle_density`` 概率独立伯努利采样。
    * 起点 ``(1, 1)`` 与终点 ``(N-2, N-2)`` 强制清空，永不为墙。

    .. important::
        本函数仅负责采样，**不检查连通性**；连通性由调用方（``reset()``）
        中的 BFS 循环保证。

    Args:
        grid_size:        迷宫边长 N，生成 N×N 的网格。
        obstacle_density: 内部格子成为墙壁的概率，范围 ``[0.0, 1.0)``。
        np_random:        Gymnasium 注入的 ``numpy.random.Generator``，
                          唯一合法随机源。

    Returns:
        形状 ``(N, N)``、dtype ``float32`` 的数组：
        ``1.0`` 表示墙，``0.0`` 表示可通行。
    """
    N = grid_size
    wall_map = np.zeros((N, N), dtype=np.float32)

    # 边界墙（向量化赋值）
    wall_map[0, :]  = 1.0
    wall_map[-1, :] = 1.0
    wall_map[:, 0]  = 1.0
    wall_map[:, -1] = 1.0

    # 内部格子：伯努利采样
    inner_mask: np.ndarray = np_random.random((N - 2, N - 2)) < obstacle_density
    wall_map[1:N-1, 1:N-1] = inner_mask.astype(np.float32)

    # 强制保证起点与终点为空地
    wall_map[1, 1]         = 0.0
    wall_map[N - 2, N - 2] = 0.0

    return wall_map


def bfs_reachable(
    wall_map: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    """验证 ``start`` 到 ``goal`` 在给定墙壁图中是否可达。

    委托给 :func:`maze_env.bfs.bfs` 实现，不重复编写 BFS 逻辑。

    Args:
        wall_map: 形状 ``(N, N)`` 的墙壁图，非零为墙，``0.0`` 可通行。
        start:    搜索起点坐标 ``(row, col)``。
        goal:     搜索终点坐标 ``(row, col)``。

    Returns:
        若存在从 ``start`` 到 ``goal`` 的全空地路径则返回 ``True``，
        否则返回 ``False``。
    """
    return bool(_bfs(wall_map.astype(np.int32), start, goal)["success"])
