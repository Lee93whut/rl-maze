"""MazeEnv 环境核心 —— 符合 OpenAI Gymnasium 标准接口（生产级）。

设计要点
--------
* **唯一随机源**：所有随机操作统一使用 Gymnasium 注入的 ``self.np_random``
  句柄（``numpy.random.Generator``），通过 ``super().reset(seed=seed)``
  初始化，严格禁止 ``numpy.random.*`` 全局函数或标准库 ``random``。
* **连通性保证**：``reset()`` 内嵌 BFS，确保生成的迷宫起点→终点绝对
  可达，不可达时自动重新采样，直到满足条件。
* **终止语义区分**：

  * ``terminated = True`` —— Agent 到达终点（任务成功完成）。
  * ``truncated  = True`` —— 超出 ``max_steps`` 步数上限（时间截断）。
  * 二者严格互斥，不会同时为 ``True``。

* **奖励语义**：撞墙时同时扣除时间惩罚（``reward_step``）与撞墙惩罚
  （``reward_wall_hit``），体现每一步都有时间成本。

典型用法::

    from maze_env import MazeEnv, Action

    env = MazeEnv.from_yaml("config.yaml")
    obs, info = env.reset()

    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, SupportsFloat

import numpy as np
import yaml
import gymnasium as gym
from gymnasium import spaces

from maze_env.actions import Action, DELTAS
from maze_env.generator import bfs_reachable, generate_maze
from maze_env.renderer import render_frame


__all__ = ["MazeEnv"]


class MazeEnv(gym.Env):
    """二维迷宫环境，遵循 OpenAI Gymnasium ``Env`` 标准接口（生产级）。

    状态空间
    --------
    ``Box(0, 1, shape=(4, N, N), dtype=np.float32)``

    * 通道 0 —— 墙壁层：``1.0`` 表示墙，``0.0`` 表示可通行格子。
    * 通道 1 —— Agent 层：Agent 当前所在格子为 ``1.0``，其余为 ``0.0``。
    * 通道 2 —— 终点层：终点格子为 ``1.0``，其余为 ``0.0``。
    * 通道 3 —— 访问历史层：本 episode 内已访问过的格子为 ``1.0``，未访问为 ``0.0``。

    动作空间
    --------
    ``Discrete(4)``：使用 :class:`~maze_env.actions.Action` 枚举或整数均可。

    奖励设计
    --------
    * 到达终点：``+reward_goal``（默认 +100），``terminated=True``。
    * 撞墙/越界：``reward_step + reward_wall_hit``（默认 −11），位置保持不变。
    * 正常移动：``reward_step``（默认 −1）。

    Note:
        撞墙时**同时**扣除时间惩罚与撞墙惩罚（体现每步均有时间成本）。

    Example:
        >>> from maze_env import MazeEnv, Action
        >>> env = MazeEnv(grid_size=6, obstacle_density=0.0, seed=0)
        >>> obs, info = env.reset()
        >>> obs.shape
        (4, 6, 6)
        >>> obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
        >>> info["agent_pos"]
        (1, 2)
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # ------------------------------------------------------------------
    # 构造与参数校验
    # ------------------------------------------------------------------

    def __init__(
        self,
        grid_size: int = 10,
        obstacle_density: float = 0.25,
        max_steps: int = 200,
        seed: Optional[int] = None,
        reward_goal: float = 100.0,
        reward_wall_hit: float = -10.0,
        reward_step: float = -1.0,
        distance_shaping_alpha: float = 0.0,
        render_mode: Optional[str] = None,
    ) -> None:
        """初始化迷宫环境。

        Args:
            grid_size:        迷宫边长 N，最小值为 4。
            obstacle_density: 内部格子成为墙壁的概率，范围 ``[0.0, 1.0)``。
            max_steps:        单幕最大步数，超出后触发 ``truncated=True``。
            seed:             构造期随机种子；每次 ``reset()`` 也可独立传入。
            reward_goal:      到达终点的奖励（建议为正数）。
            reward_wall_hit:  撞墙惩罚（建议为负数）。
            reward_step:      每步时间惩罚（建议为负数）。
            distance_shaping_alpha: 距离 shaping 系数（默认 0.0 = 关闭）。
                每步额外奖励 = alpha × (移动前曼哈顿距离 − 移动后曼哈顿距离)，
                靠近目标为正，远离为负；撞墙步不计入（位置未变）。
            render_mode:      渲染模式，可选 ``"human"`` 或 ``"ansi"``。

        Raises:
            ValueError: 若 ``grid_size < 4``、``obstacle_density`` 越界、
                ``max_steps < 1``，或 ``render_mode`` 不在合法值列表中。
        """
        super().__init__()

        # ── 参数校验 ───────────────────────────────────────────────────
        if grid_size < 4:
            raise ValueError(f"grid_size 必须 >= 4，当前值：{grid_size}")
        if not (0.0 <= obstacle_density < 1.0):
            raise ValueError(
                f"obstacle_density 必须在 [0.0, 1.0) 内，当前值：{obstacle_density}"
            )
        if max_steps < 1:
            raise ValueError(f"max_steps 必须 >= 1，当前值：{max_steps}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"不支持的 render_mode '{render_mode}'，"
                f"可选值：{self.metadata['render_modes']}"
            )

        # ── 超参数（不可变） ───────────────────────────────────────────
        self.grid_size: int = grid_size
        self.obstacle_density: float = obstacle_density
        self.max_steps: int = max_steps
        self.init_seed: Optional[int] = seed
        self.reward_goal: float = reward_goal
        self.reward_wall_hit: float = reward_wall_hit
        self.reward_step: float = reward_step
        self.distance_shaping_alpha: float = distance_shaping_alpha
        self.render_mode: Optional[str] = render_mode

        # ── 空间声明 ───────────────────────────────────────────────────
        self.observation_space: spaces.Box = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4, grid_size, grid_size),   # ch0=wall, ch1=agent, ch2=goal, ch3=visited
            dtype=np.float32,
        )
        self.action_space: spaces.Discrete = spaces.Discrete(len(Action))

        # ── 运行时状态（占位，由 reset() 正式填充） ────────────────────
        self._wall_map: np.ndarray = np.zeros(
            (grid_size, grid_size), dtype=np.float32
        )
        self._visited_map: np.ndarray = np.zeros(
            (grid_size, grid_size), dtype=np.float32
        )
        self._agent_pos: tuple[int, int] = (1, 1)
        self._goal_pos: tuple[int, int] = (grid_size - 2, grid_size - 2)
        self._step_count: int = 0
        self._hit_wall_count: int = 0
        self._episode_success: bool = False

    # ------------------------------------------------------------------
    # 公开只读属性（封装内部状态，供训练脚本等外部代码合法访问）
    # ------------------------------------------------------------------

    @property
    def wall_map(self) -> np.ndarray:
        """当前幕的墙壁图，形状 ``(N, N)`` float32，1.0=墙，0.0=可通行。

        返回只读视图（zero-copy），防止外部意外篡改环境内部状态。
        若需要可写副本，请显式调用 ``.copy()``。
        """
        view = self._wall_map.view()
        view.flags.writeable = False
        return view

    @property
    def goal_pos(self) -> tuple[int, int]:
        """当前幕的终点坐标 ``(row, col)``，只读。"""
        return self._goal_pos

    @property
    def agent_pos(self) -> tuple[int, int]:
        """Agent 当前坐标 ``(row, col)``，只读。"""
        return self._agent_pos

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        render_mode: Optional[str] = None,
    ) -> "MazeEnv":
        """从已解析的配置字典创建环境实例。

        配置格式::

            maze:
              grid_size: 10
              obstacle_density: 0.25
              max_steps: 200
            rewards:
              goal: 100
              wall_hit: -10
              step: -1

        注：``maze.seed`` 不被此方法读取。需固定地图时，
        请在创建实例后显式调用 ``env.reset(seed=X)``。

        Args:
            config:      ``yaml.safe_load`` 等工具解析得到的字典。
            render_mode: 渲染模式。

        Returns:
            配置好的 ``MazeEnv`` 实例。
        """
        maze_cfg: dict[str, Any] = config["maze"]
        reward_cfg: dict[str, Any] = config.get("rewards", {})
        return cls(
            grid_size=int(maze_cfg.get("grid_size", 10)),
            obstacle_density=float(maze_cfg.get("obstacle_density", 0.25)),
            max_steps=int(maze_cfg.get("max_steps", 200)),
            # seed 不从 config 读取：调用方按需显式传入。
            # config.yaml 中 maze.seed 仅用于 overfit 调试节，
            # 透传此处会导致普通调用者意外锁死到同一张地图。
            reward_goal=float(reward_cfg.get("goal", 100.0)),
            reward_wall_hit=float(reward_cfg.get("wall_hit", -10.0)),
            reward_step=float(reward_cfg.get("step", -1.0)),
            distance_shaping_alpha=float(reward_cfg.get("distance_shaping_alpha", 0.0)),
            render_mode=render_mode,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = "config.yaml",
        render_mode: Optional[str] = None,
    ) -> "MazeEnv":
        """从 YAML 文件路径直接创建环境实例。

        Args:
            path:        YAML 配置文件路径，默认 ``"config.yaml"``。
            render_mode: 渲染模式。

        Returns:
            配置好的 ``MazeEnv`` 实例。
        """
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls.from_config(cfg, render_mode=render_mode)

    # ------------------------------------------------------------------
    # Gymnasium 核心接口
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境，生成新迷宫并将 Agent 放置到起点。

        Args:
            seed:    本幕随机种子。优先级：``reset(seed=X)`` > 构造期 ``seed``。
            options: 可选注入字典，支持以下键：

                * ``"wall_map"`` *(np.ndarray)*  — 直接使用外部提供的墙壁图，
                  跳过随机生成（形状须为 ``(N, N)``，非零为墙）。
                * ``"start"`` *(tuple[int,int])* — Agent 起点坐标，
                  默认 ``(1, 1)``。
                * ``"goal"`` *(tuple[int,int])*  — 终点坐标，
                  默认 ``(N-2, N-2)``。

                注入外部地图时，调用方须自行保证起点→终点连通。

        Returns:
            ``(observation, info)``：初始观测张量与 info 字典。
        """
        effective_seed = seed if seed is not None else self.init_seed
        super().reset(seed=effective_seed)

        opts: dict[str, Any] = options or {}

        # 重置幕级统计
        self._step_count = 0
        self._hit_wall_count = 0
        self._episode_success = False
        self._visited_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._agent_pos = opts.get("start", (1, 1))
        self._goal_pos  = opts.get("goal",  (self.grid_size - 2, self.grid_size - 2))
        # 起点标记为已访问
        ar, ac = self._agent_pos
        self._visited_map[ar, ac] = 1.0

        if "wall_map" in opts:
            # ── 外部注入地图（用于推理 / 可视化，跳过随机生成）────────────
            wall_map = np.asarray(opts["wall_map"], dtype=np.float32)
            expected = (self.grid_size, self.grid_size)
            if wall_map.shape != expected:
                raise ValueError(
                    f"注入的 wall_map 形状 {wall_map.shape} 与环境 "
                    f"grid_size={self.grid_size} 不匹配，期望 {expected}"
                )
            self._wall_map = wall_map
        else:
            # ── 随机生成，BFS 保证连通 ────────────────────────────────────
            while True:
                self._wall_map = generate_maze(
                    self.grid_size, self.obstacle_density, self.np_random
                )
                if bfs_reachable(self._wall_map, self._agent_pos, self._goal_pos):
                    break

        return self._build_observation(), self._build_info()

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """执行一步动作并返回转移结果。

        Args:
            action: 动作编号，合法值 ``{0,1,2,3}`` 或 :class:`Action` 枚举。

        Returns:
            ``(observation, reward, terminated, truncated, info)``

        Raises:
            AssertionError: 若 ``action`` 不在合法动作空间内。
        """
        assert self.action_space.contains(action), (
            f"非法动作 {action!r}，合法范围：{self.action_space}"
        )

        dr, dc = DELTAS[action]
        cur_row, cur_col = self._agent_pos
        new_row, new_col = cur_row + dr, cur_col + dc
        N = self.grid_size

        # 移动前的曼哈顿距离（用于距离 shaping）
        gr, gc = self._goal_pos
        dist_before: int = abs(cur_row - gr) + abs(cur_col - gc)

        # 碰撞检测（显式 bool() 强转，避免 numpy.bool_ 与 Python bool 不一致）
        hit_wall: bool = bool(
            new_row < 0 or new_row >= N
            or new_col < 0 or new_col >= N
            or self._wall_map[new_row, new_col] == 1.0
        )

        if hit_wall:
            self._hit_wall_count += 1
            # 撞墙：时间惩罚 + 撞墙惩罚（体现每步均有时间成本）；位置不变，不计入 shaping
            reward: float = self.reward_step + self.reward_wall_hit
        else:
            self._agent_pos = (new_row, new_col)
            reward = self.reward_step
            # 距离 shaping：靠近目标为正，远离为负（仅有效移动步计入）
            # 注：本项目 config 固定 distance_shaping_alpha=0.0，train.py 也未透传该字段，
            #     故此 if 分支在当前训练/评估流程中永不执行，保留作为参数设计的可扩展点。
            if self.distance_shaping_alpha != 0.0:
                dist_after: int = abs(new_row - gr) + abs(new_col - gc)
                reward += self.distance_shaping_alpha * (dist_before - dist_after)
            # 更新访问地图（有效移动后标记新格子）
            self._visited_map[new_row, new_col] = 1.0

        self._step_count += 1

        # 终止判断（terminated 与 truncated 严格互斥）
        terminated: bool = self._agent_pos == self._goal_pos
        if terminated:
            reward += self.reward_goal
            self._episode_success = True

        truncated: bool = (not terminated) and (self._step_count >= self.max_steps)

        info = self._build_info()
        info["hit_wall"] = hit_wall  # 本步撞墙标志（单步，非幕级）

        if self.render_mode == "human":
            self.render()

        return self._build_observation(), float(reward), terminated, truncated, info

    def render(self) -> Optional[str]:
        """渲染当前状态为 ASCII 网格。

        Returns:
            * ``"ansi"``  模式：返回字符串。
            * ``"human"`` 模式：打印到 stdout，返回 ``None``。
            * ``None``    模式：无操作，返回 ``None``。
        """
        if self.render_mode is None:
            return None

        output = render_frame(
            wall_map=self._wall_map,
            agent_pos=self._agent_pos,
            goal_pos=self._goal_pos,
            step_count=self._step_count,
            max_steps=self.max_steps,
            hit_wall_count=self._hit_wall_count,
            episode_success=self._episode_success,
        )

        if self.render_mode == "human":
            print(output)
            return None
        return output

    def close(self) -> None:
        """释放资源（当前无外部资源，保留以满足 Gymnasium 接口规范）。"""

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """将当前状态编码为四通道观测张量 ``(4, N, N)``。

        通道说明：
            ch0 — wall_map：墙壁位置（1=墙，0=通路）
            ch1 — agent_map：agent 当前位置（one-hot）
            ch2 — goal_map：终点位置（one-hot）
            ch3 — visited_map：本 episode 内已访问过的格子（二值，1=到达过，0=未到达）
        """
        N = self.grid_size
        obs = np.zeros((4, N, N), dtype=np.float32)
        obs[0] = self._wall_map
        ar, ac = self._agent_pos
        obs[1, ar, ac] = 1.0
        gr, gc = self._goal_pos
        obs[2, gr, gc] = 1.0
        obs[3] = self._visited_map
        return obs

    def _build_info(self) -> dict[str, Any]:
        """构建幕级统计 info 字典。

        Returns:
            包含 ``agent_pos``、``goal_pos``、``step_count``、
            ``hit_wall_count``、``success`` 五个字段的字典。

        Note:
            ``step()`` 会在此基础上额外追加 ``"hit_wall": bool``（单步标志）；
            ``reset()`` 返回的 info 不含该字段（初始无此概念），
            调用方需注意两处 info 结构的微小差异。
        """
        return {
            "agent_pos":      self._agent_pos,
            "goal_pos":       self._goal_pos,
            "step_count":     self._step_count,
            "hit_wall_count": self._hit_wall_count,
            "success":        self._episode_success,
        }
