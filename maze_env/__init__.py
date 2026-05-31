"""maze_env —— 生产级二维迷宫强化学习环境（Gymnasium 标准接口）。

公开 API
--------
* :class:`MazeEnv`  — 环境主类。
* :class:`Action`   — 四方向动作枚举（``UP / DOWN / LEFT / RIGHT``）。
* :func:`bfs`       — BFS 最短路算法（供训练脚本与 Web App 直接使用）。

快速上手::

    from maze_env import MazeEnv, Action
    from maze_env.bfs import bfs

    env = MazeEnv(grid_size=10, obstacle_density=0.25, seed=42)
    obs, info = env.reset()

    obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
"""

from maze_env.env import MazeEnv
from maze_env.actions import Action
from maze_env.bfs import bfs

__all__ = ["MazeEnv", "Action", "bfs"]

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("maze-env")
except PackageNotFoundError:
    __version__ = "unknown"
