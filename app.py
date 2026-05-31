"""app.py —— DQN 迷宫寻路可视化 Web App
Hugging Face Spaces (Docker SDK) 专用

部署清单（上传到 HF Space 的全部文件）
--------------------------------------
app.py                                    本文件
src/model.py                              神经网络架构
results/best_model_train_vanilla.pth      vanilla DQN 权重
results/best_model_train_double.pth       Double DQN 权重
results/best_model_train_dueling.pth      Dueling DQN 权重
results/best_model_train_double_dueling.pth  Double Dueling DQN 权重
config.yaml                               环境配置（grid_size / obstacle_density / max_steps）
requirements.txt                          依赖列表

导入策略
--------
* maze_env 通过 `pip install -e .` 安装（见 Dockerfile），直接 import。
* src/ 通过 pyproject.toml packages.find 配置，同样可安装，直接 import。
* 所有模块均通过标准 import 路径解析，无需 sys.path 手动注入。

端口说明
--------
HF Docker Space 固定使用 7860 端口（见 Dockerfile / README）。
本地调试：streamlit run app.py
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

# ── maze_env 包（已安装，直接导入）──────────────────────────────────────────
from maze_env import MazeEnv
from maze_env.bfs import bfs as bfs_solve

# ── src 包（pip install -e . 后可直接导入）───────────────────────────────────
import torch.nn as nn
from src.model import DQNNetwork, DuelingDQNNetwork

# ===========================================================================
# 常量 & 配置
# ===========================================================================
_CONFIG_PATH = Path(__file__).parent / "config.yaml"
if _CONFIG_PATH.exists():
    _cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
else:
    import warnings
    warnings.warn(
        f"config.yaml 未找到（{_CONFIG_PATH}），使用内置默认值。"
        "若训练时使用了非默认 grid_size，推理结果可能错误。",
        stacklevel=1,
    )
    _cfg = {}
_maze_cfg = _cfg.get("maze", {})

GRID_SIZE        = int(_maze_cfg.get("grid_size", 10))
OBSTACLE_DENSITY = float(_maze_cfg.get("obstacle_density", 0.25))  # 与 config.yaml maze.obstacle_density 保持一致，确保 Demo 与训练分布相同
MAX_STEPS        = int(_maze_cfg.get("max_steps", 200))  # 与训练保持一致，推理步数预算对齐

# 支持切换的四算法（顺序决定 UI 下拉框排列）
ALGO_OPTIONS: list[str] = ["double_dueling", "dueling", "double", "vanilla"]
ALGO_LABELS: dict[str, str] = {
    "vanilla":        "Vanilla DQN（基准）",
    "double":         "Double DQN（抑制高估）",
    "dueling":        "Dueling DQN（V+A 分解）",
    "double_dueling": "Double + Dueling（推荐）",
}
# 默认算法：优先读 config.yaml，fallback 到 double_dueling
_default_algo = str(_cfg.get("dqn", {}).get("algorithm", "double_dueling")).strip().lower()
DEFAULT_ALGO: str = _default_algo if _default_algo in ALGO_OPTIONS else "double_dueling"

def model_path_for(algo: str) -> Path:
    """根据算法名返回对应权重文件路径。"""
    return Path(__file__).parent / "results" / f"best_model_train_{algo}.pth"

# 首屏默认迷宫 seed。
# 固定值保证分享链接时双方看到相同地图；改为 None 可让每次刷新随机生成。
DEFAULT_MAZE_SEED: int = 42

# 动画帧间隔（秒）
ANIM_DELAY = 0.08

# 颜色映射（RGB 列表，供 Plotly heatmap）
COLOR_EMPTY     = "#F8F9FA"   # 白/浅灰 —— 可通行地板
COLOR_WALL      = "#2C3E50"   # 深蓝灰  —— 墙壁
COLOR_START     = "#27AE60"   # 绿色    —— 起点
COLOR_GOAL      = "#E74C3C"   # 红色    —— 终点
COLOR_DQN_PATH  = "#3498DB"   # 蓝色    —— DQN 轨迹
COLOR_BFS_PATH  = "#F39C12"   # 橙色    —— BFS 最短路
COLOR_AGENT     = "#9B59B6"   # 紫色    —— 当前 Agent 位置

# ===========================================================================
# 工具函数
# ===========================================================================

def generate_maze(seed: Optional[int] = None) -> np.ndarray:
    """生成 GRID_SIZE×GRID_SIZE 迷宫，保证起点 (1,1) 与终点 (N-2,N-2) 可达。

    委托给 :class:`MazeEnv` 的 ``reset()`` 方法，确保与训练环境完全一致
    （相同的边界墙、障碍密度、BFS 连通性保证，不重复造轮子）。

    Args:
        seed: 随机种子；``None`` 表示不固定随机性。

    Returns:
        wall_map: shape ``(N, N)``，dtype ``int32``，0=通路，1=墙壁。
    """
    env = MazeEnv(
        grid_size=GRID_SIZE,
        obstacle_density=OBSTACLE_DENSITY,
    )
    env.reset(seed=seed)
    return env.wall_map.astype(np.int32)


def generate_maze_with_random_sg(
    seed: Optional[int] = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """生成迷宫并从可通行内部格随机选取起点和终点，与训练分布完全一致。

    复现 train.py 中 ``random_start_goal=True`` 的逻辑：
    先生成迷宫，再用 ``env.np_random``（Gymnasium 注入的唯一随机源）
    从内部可通行格中不放回地抽取两个不同坐标，确保 Demo 与训练同分布。

    Args:
        seed: 随机种子；``None`` 表示不固定随机性。

    Returns:
        (wall_map, start, goal)：
        * wall_map: shape ``(N, N)``，dtype ``int32``。
        * start:    起点坐标 ``(row, col)``。
        * goal:     终点坐标 ``(row, col)``。
    """
    env = MazeEnv(
        grid_size=GRID_SIZE,
        obstacle_density=OBSTACLE_DENSITY,
    )
    env.reset(seed=seed)
    wall_map = env.wall_map.astype(np.int32)   # (N, N)

    # 收集内部（非边界）可通行格，与 train.py 过滤条件完全相同
    rows, cols = np.where(wall_map == 0)
    inner_cells: list[tuple[int, int]] = [
        (int(r), int(c))
        for r, c in zip(rows, cols)
        if 0 < r < GRID_SIZE - 1 and 0 < c < GRID_SIZE - 1
    ]

    if len(inner_cells) < 2:
        # 极端情况（障碍密度极高）：退回到固定起终点
        return wall_map, (1, 1), (GRID_SIZE - 2, GRID_SIZE - 2)

    # 使用 env.np_random（与训练逻辑完全一致，不污染全局随机状态）
    idxs = env.np_random.integers(0, len(inner_cells), size=2)
    while idxs[0] == idxs[1]:
        idxs = env.np_random.integers(0, len(inner_cells), size=2)

    start = inner_cells[int(idxs[0])]
    goal  = inner_cells[int(idxs[1])]
    return wall_map, start, goal


def load_model(algo: str = DEFAULT_ALGO, grid_size: int = GRID_SIZE) -> tuple[Optional[nn.Module], int]:
    """加载指定算法的 DQN 模型权重，返回 (net, saved_grid_size)。

    Args:
        algo:      算法名，须在 ALGO_OPTIONS 中。
        grid_size: 当前环境 grid_size，用于维度不一致时的 fallback 返回值。

    失败时返回 (None, grid_size)。saved_grid_size 供调用方检测维度是否与
    当前 GRID_SIZE 一致；不一致时推理输入维度会与网络期望不符，应提前告警。
    """
    path = model_path_for(algo)
    if not path.exists():
        return None, grid_size
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        saved_gs    = ckpt.get("grid_size", grid_size)
        algorithm   = ckpt.get("algorithm", "vanilla").strip().lower()
        NetClass    = DuelingDQNNetwork if "dueling" in algorithm else DQNNetwork
        net = NetClass(grid_size=saved_gs)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net, saved_gs
    except Exception as e:
        st.error(f"❌ 模型加载失败：{e}")
        return None, grid_size


def dqn_rollout(
    net: nn.Module,
    wall_map: np.ndarray,
    start: tuple,
    goal: tuple,
) -> list[tuple]:
    """纯推理（ε=0）运行 DQN Agent，返回完整轨迹坐标列表。

    委托给 :class:`MazeEnv` 的标准 ``reset()`` / ``step()`` 接口，
    保证观测编码与训练时完全一致，无需在 app.py 中重复实现碰撞检测。

    Args:
        net:      已加载权重、处于 eval 模式的 DQN 网络。
        wall_map: shape ``(N, N)``，dtype int32，0=通路，1=墙壁。
        start:    Agent 起点 ``(row, col)``。
        goal:     终点 ``(row, col)``。

    Returns:
        完整轨迹（含起点），每条为 ``(row, col)``。
    """
    env = MazeEnv(
        grid_size=wall_map.shape[0],
        obstacle_density=0.0,       # 密度无关，地图由外部注入
        max_steps=MAX_STEPS,
    )
    obs, _ = env.reset(options={
        "wall_map": wall_map.astype(np.float32),
        "start":    start,
        "goal":     goal,
    })

    path = [env.agent_pos]

    # 注：R4 起观测已包含 visited_map 第4通道（ch3），Agent 天然感知访问历史，
    # 无需在推理侧注入 Q 值惩罚。直接贪心执行即可。
    while True:
        s = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = net(s)[0]            # shape: (num_actions,)

        action = int(q_values.argmax().item())
        obs, _reward, terminated, truncated, info = env.step(action)

        # 只在实际移动时追加（撞墙时位置不变，避免重复坐标导致动画抖帧）
        if not info["hit_wall"]:
            path.append(env.agent_pos)

        if terminated or truncated:
            break

    return path


# ===========================================================================
# Plotly 迷宫绘制
# ===========================================================================

def build_maze_figure(
    wall_map:    np.ndarray,
    start:       tuple,
    goal:        tuple,
    dqn_path:    Optional[list] = None,
    bfs_path:    Optional[list] = None,
    agent_pos:   Optional[tuple] = None,
    highlight_dqn_step: int = -1,
) -> go.Figure:
    """构建 Plotly 迷宫图，支持叠加 DQN / BFS 路径与动态 Agent 标记。"""
    N = wall_map.shape[0]

    # ── 底层热力图（单 Heatmap trace，O(1) traces vs O(N²) shapes）─────────
    # 数值矩阵：0=通路, 1=墙, 2=起点, 3=终点
    z = wall_map.astype(float).copy()
    z[start[0], start[1]] = 2.0
    z[goal[0],  goal[1]]  = 3.0

    # 离散颜色映射：值 → 颜色
    colorscale = [
        [0.00, COLOR_EMPTY],   # 0 = 通路
        [0.25, COLOR_EMPTY],
        [0.25, COLOR_WALL],    # 1 = 墙
        [0.50, COLOR_WALL],
        [0.50, COLOR_START],   # 2 = 起点
        [0.75, COLOR_START],
        [0.75, COLOR_GOAL],    # 3 = 终点
        [1.00, COLOR_GOAL],
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z,
        colorscale=colorscale,
        zmin=0, zmax=3,
        showscale=False,
        xgap=1, ygap=1,
        hoverinfo="skip",
    ))

    # ── BFS 路径（橙色虚线）──────────────────────────────────────────────
    if bfs_path and len(bfs_path) > 1:
        bx = [c for r, c in bfs_path]
        by = [r for r, c in bfs_path]
        fig.add_trace(go.Scatter(
            x=bx, y=by,
            mode="lines+markers",
            name="BFS 最短路",
            line=dict(color=COLOR_BFS_PATH, width=3, dash="dot"),
            marker=dict(size=6, color=COLOR_BFS_PATH, opacity=0.7),
        ))

    # ── DQN 路径（蓝色实线）──────────────────────────────────────────────
    if dqn_path and len(dqn_path) > 1:
        # 截取到 highlight_dqn_step（动画用）
        end_idx = highlight_dqn_step + 1 if highlight_dqn_step >= 0 else len(dqn_path)
        sub_path = dqn_path[:end_idx]
        dx = [c for r, c in sub_path]
        dy = [r for r, c in sub_path]
        fig.add_trace(go.Scatter(
            x=dx, y=dy,
            mode="lines+markers",
            name="DQN 轨迹",
            line=dict(color=COLOR_DQN_PATH, width=3),
            marker=dict(size=7, color=COLOR_DQN_PATH),
        ))

    # ── 当前 Agent 位置（紫色大圆点）────────────────────────────────────
    ap = agent_pos if agent_pos else (start if not dqn_path else
                                      (dqn_path[min(highlight_dqn_step, len(dqn_path)-1)]
                                       if highlight_dqn_step >= 0 else start))
    fig.add_trace(go.Scatter(
        x=[ap[1]], y=[ap[0]],
        mode="markers",
        name="Agent",
        marker=dict(size=16, color=COLOR_AGENT, symbol="circle",
                    line=dict(color="white", width=2)),
        showlegend=True,
    ))

    # ── 起点 / 终点标签 ───────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[start[1], goal[1]],
        y=[start[0], goal[0]],
        mode="markers+text",
        text=["S", "G"],
        textposition="middle center",
        textfont=dict(size=13, color="white", family="Arial Black"),
        marker=dict(size=22, color=[COLOR_START, COLOR_GOAL],
                    symbol="square", opacity=0.0),  # 透明底，只显示字
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── 布局 ─────────────────────────────────────────────────────────────
    fig.update_layout(
        width=560, height=560,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            range=[-0.5, N - 0.5], tickvals=list(range(N)),
            showgrid=False, zeroline=False, title="列 (col)",
        ),
        yaxis=dict(
            range=[N - 0.5, -0.5],
            tickvals=list(range(N)),
            showgrid=False, zeroline=False, title="行 (row)",
        ),
        legend=dict(x=1.01, y=1, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#BDC3C7", borderwidth=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(text="🏁 DQN 迷宫寻路", x=0.5, font=dict(size=16)),
    )
    return fig


def _find_cell_index(free_cells: list[tuple], pos: tuple) -> int:
    """在 free_cells 列表中查找 pos 的索引；未找到时返回 0（安全回退）。"""
    try:
        return free_cells.index(pos)
    except ValueError:
        return 0


# ===========================================================================
# Session State 初始化
# ===========================================================================

def _init_state() -> None:
    if "wall_map" not in st.session_state:
        # 首屏使用随机起终点（与训练分布一致），固定 seed 保证可复现
        wm, sg_start, sg_goal = generate_maze_with_random_sg(seed=DEFAULT_MAZE_SEED)
        st.session_state.wall_map   = wm
        st.session_state.start      = sg_start
        st.session_state.goal       = sg_goal
    if "start" not in st.session_state:
        st.session_state.start      = (1, 1)
    if "goal" not in st.session_state:
        st.session_state.goal       = (GRID_SIZE - 2, GRID_SIZE - 2)
    if "dqn_path" not in st.session_state:
        st.session_state.dqn_path   = None
    if "bfs_path" not in st.session_state:
        st.session_state.bfs_path   = None
    if "metrics" not in st.session_state:
        st.session_state.metrics    = None
    if "selected_algo" not in st.session_state:
        st.session_state.selected_algo = DEFAULT_ALGO
    if "model" not in st.session_state:
        net, saved_gs = load_model(algo=DEFAULT_ALGO)
        st.session_state.model           = net
        st.session_state.model_grid_size = saved_gs
    if "maze_seed" not in st.session_state:
        st.session_state.maze_seed  = DEFAULT_MAZE_SEED
    if "anim_running" not in st.session_state:
        st.session_state.anim_running = False
    if "anim_step" not in st.session_state:
        st.session_state.anim_step = 0
    if "anim_path" not in st.session_state:
        st.session_state.anim_path = None


# ===========================================================================
# 主程序
# ===========================================================================

def main() -> None:
    st.set_page_config(
        page_title="DQN 迷宫寻路 Demo",
        page_icon="🤖",
        layout="wide",
    )

    # ── 全局样式注入 ────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 16px 20px; color: white;
        text-align: center; margin: 6px 0;
    }
    .metric-label { font-size: 13px; opacity: 0.85; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; }
    .por-perfect  { color: #2ECC71; font-weight: 800; }
    .por-good     { color: #F39C12; font-weight: 700; }
    .por-bad      { color: #E74C3C; font-weight: 600; }
    div[data-testid="stButton"] button {
        width: 100%; border-radius: 8px; font-weight: 600;
    }
    /* 迷宫按钮网格：每格紧凑正方形，无内边距 */
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
        padding: 0 !important;
        min-height: 40px !important;
        font-size: 15px !important;
        border-radius: 3px !important;
        border: 1px solid #ccc !important;
        line-height: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    _init_state()

    st.title("🤖 DQN 迷宫寻路 · 可视化 Demo")
    st.caption("Deep Q-Network × BFS Ground-Truth · Hugging Face Spaces")

    # ═══════════════════════════════════════════════════════════════════════
    # 正常双栏布局（点击模式在右栏内处理，不破坏整体布局）
    # ═══════════════════════════════════════════════════════════════════════
    left_col, right_col = st.columns([1, 2.2], gap="large")

    # ───────────────────────────────────────────────────────────────────────
    # 左栏：控制面板
    # ───────────────────────────────────────────────────────────────────────
    with left_col:
        st.subheader("⚙️ 控制面板")

        # ── 迷宫生成 ─────────────────────────────────────────────────────
        st.markdown("**① 迷宫地图**")
        col_seed, col_rand = st.columns([3, 1])
        with col_seed:
            input_seed = st.number_input(
                "迷宫 Seed",
                min_value=0,
                max_value=999999,
                value=st.session_state.maze_seed,
                step=1,
                help="固定数字可复现指定地图；点击右侧按钮随机生成新地图",
            )
        with col_rand:
            st.write("")   # 对齐占位
            if st.button("🎲 随机"):
                # 随机 seed：同时随机生成地图和起终点（与训练分布一致）
                new_seed = random.randint(0, 999999)
                wm, sg_start, sg_goal = generate_maze_with_random_sg(seed=new_seed)
                st.session_state.maze_seed = new_seed
                st.session_state.wall_map  = wm
                st.session_state.start     = sg_start
                st.session_state.goal      = sg_goal
                st.session_state.dqn_path  = None
                st.session_state.bfs_path  = None
                st.session_state.metrics   = None
                # 同步下拉框索引，避免 selectbox key 缓存旧值
                _fc = [(r,c) for r in range(1,GRID_SIZE-1) for c in range(1,GRID_SIZE-1) if wm[r,c]==0]
                st.session_state.start_select = _find_cell_index(_fc, sg_start)
                st.session_state.goal_select  = _find_cell_index(_fc, sg_goal)
                st.rerun()   # 立即终止当前脚本，下方 input_seed 检测不会执行

        # 手动修改 seed 输入框时触发（随机按钮已由上方 rerun 短路，不会重复）
        if input_seed != st.session_state.maze_seed:
            wm, sg_start, sg_goal = generate_maze_with_random_sg(seed=input_seed)
            st.session_state.maze_seed = input_seed
            st.session_state.wall_map  = wm
            st.session_state.start     = sg_start
            st.session_state.goal      = sg_goal
            st.session_state.dqn_path  = None
            st.session_state.bfs_path  = None
            st.session_state.metrics   = None
            _fc = [(r,c) for r in range(1,GRID_SIZE-1) for c in range(1,GRID_SIZE-1) if wm[r,c]==0]
            st.session_state.start_select = _find_cell_index(_fc, sg_start)
            st.session_state.goal_select  = _find_cell_index(_fc, sg_goal)
            st.rerun()

        st.divider()

        # ── 起点 / 终点选择 ────────────────────────────────────────────────
        st.markdown("**② 起点 & 终点**")

        # 「随机起终点」按钮：从当前地图的可通行格随机选取，与训练分布一致
        if st.button("🎲 随机起终点", use_container_width=True,
                     help="从当前地图可通行格随机选取起点和终点，与训练分布完全一致"):
            _wm = st.session_state.wall_map
            _rows, _cols = np.where(_wm == 0)
            _inner = [
                (int(r), int(c))
                for r, c in zip(_rows, _cols)
                if 0 < r < GRID_SIZE - 1 and 0 < c < GRID_SIZE - 1
            ]
            if len(_inner) >= 2:
                _i, _j = random.sample(range(len(_inner)), 2)
                st.session_state.start    = _inner[_i]
                st.session_state.goal     = _inner[_j]
                st.session_state.dqn_path = None
                st.session_state.bfs_path = None
                st.session_state.metrics  = None
                st.session_state.start_select = _find_cell_index(_inner, _inner[_i])
                st.session_state.goal_select  = _find_cell_index(_inner, _inner[_j])
                st.rerun()

        N = GRID_SIZE
        free_cells = [
            (r, c)
            for r in range(1, N - 1)
            for c in range(1, N - 1)
            if st.session_state.wall_map[r, c] == 0
        ]
        cell_labels = [f"({r},{c})" for r, c in free_cells]

        start_idx = st.selectbox(
            "起点 (row, col)",
            options=range(len(free_cells)),
            format_func=lambda i: cell_labels[i],
            index=_find_cell_index(free_cells, st.session_state.start),
            key="start_select",
        )
        goal_idx = st.selectbox(
            "终点 (row, col)",
            options=range(len(free_cells)),
            format_func=lambda i: cell_labels[i],
            index=_find_cell_index(free_cells, st.session_state.goal),
            key="goal_select",
        )
        new_start = free_cells[start_idx]
        new_goal  = free_cells[goal_idx]

        if new_start == new_goal:
            st.warning("⚠️  起点与终点不能相同，请重新选择。")
        elif new_start != st.session_state.start or new_goal != st.session_state.goal:
            st.session_state.start    = new_start
            st.session_state.goal     = new_goal
            st.session_state.dqn_path = None
            st.session_state.bfs_path = None
            st.session_state.metrics  = None

        st.divider()

        # ── 算法选择 & 寻路触发按钮 ───────────────────────────────────────
        st.markdown("**③ 寻路算法**")

        selected_algo = st.selectbox(
            "DQN 算法变体",
            options=ALGO_OPTIONS,
            format_func=lambda a: ALGO_LABELS[a],
            index=ALGO_OPTIONS.index(st.session_state.selected_algo),
            key="algo_select",
            help="切换算法后点击「DQN 寻路」按钮可对比不同算法在同一地图上的路径",
        )
        # 算法切换时重新加载对应模型，清空上次路径结果
        if selected_algo != st.session_state.selected_algo:
            st.session_state.selected_algo = selected_algo
            net, saved_gs = load_model(algo=selected_algo)
            st.session_state.model           = net
            st.session_state.model_grid_size = saved_gs
            st.session_state.dqn_path        = None
            st.session_state.metrics         = None
            st.rerun()

        run_dqn = st.button(
            "🤖 DQN 智能体寻路",
            use_container_width=True,
            type="primary",
        )
        run_bfs = st.button(
            "📐 BFS 专家寻路",
            use_container_width=True,
        )

        st.divider()

        # ── 图例说明 ────────────────────────────────────────────────────
        st.markdown("**图例**")
        legend_html = """
        <div style='font-size:13px; line-height:2'>
        🟩 <b>S</b> 起点 &nbsp;&nbsp;
        🟥 <b>G</b> 终点<br>
        ⬛ 墙壁 &nbsp;&nbsp;
        ⬜ 通路<br>
        🔵 DQN 轨迹 &nbsp;&nbsp;
        🟠 BFS 最短路<br>
        🟣 Agent 当前位置
        </div>
        """
        st.markdown(legend_html, unsafe_allow_html=True)

        # ── 模型状态 ────────────────────────────────────────────────────
        st.divider()
        _cur_algo     = st.session_state.get("selected_algo", DEFAULT_ALGO)
        _cur_path     = model_path_for(_cur_algo)
        if st.session_state.model is not None:
            st.success(f"✅ 模型已加载 ({_cur_path.name})")
            # 维度不一致时提前告警：网络期望 (3, saved_gs, saved_gs) 输入，
            # 而推理环境会生成 (3, GRID_SIZE, GRID_SIZE) 观测，两者不符会在
            # 网络 forward 时抛出张量尺寸异常。提前展示警告便于用户定位原因。
            _saved_gs = st.session_state.get("model_grid_size", GRID_SIZE)
            if _saved_gs != GRID_SIZE:
                st.warning(
                    f"⚠️ 模型训练于 {_saved_gs}×{_saved_gs} 迷宫，"
                    f"当前配置为 {GRID_SIZE}×{GRID_SIZE}。\n"
                    "推理时输入维度不匹配，将导致运行时错误。\n"
                    "请使用匹配 grid_size 的模型，或更新 config.yaml。"
                )
        else:
            st.error(f"❌ 未找到 {_cur_path.name}")
            st.info(f"请先运行 `python src/train.py --algorithm {_cur_algo}` 训练模型。")

    # ───────────────────────────────────────────────────────────────────────
    # 右栏：主画布
    # ───────────────────────────────────────────────────────────────────────
    # ───────────────────────────────────────────────────────────────────────
    # 右栏：主画布
    # ───────────────────────────────────────────────────────────────────────
    with right_col:
        wall_map   = st.session_state.wall_map
        start      = st.session_state.start
        goal       = st.session_state.goal

        status_placeholder = st.empty()

        # ── BFS 寻路 ─────────────────────────────────────────────────────
        if run_bfs:
            result = bfs_solve(wall_map.astype(np.int32), start, goal)
            if result["success"]:
                st.session_state.bfs_path = result["path"]
                status_placeholder.success(
                    f"✅ BFS 完成！最短步数 = **{result['steps']}**，"
                    f"耗时 {result['execution_time_ms']:.3f} ms"
                )
            else:
                st.session_state.bfs_path = None
                status_placeholder.error("❌ BFS：起点与终点之间无可达路径！")

        # ── DQN 寻路按钮触发 ──────────────────────────────────────────────
        if run_dqn:
            model = st.session_state.model
            if model is None:
                status_placeholder.error("❌ 模型未加载，无法推理。")
            elif st.session_state.get("model_grid_size", GRID_SIZE) != GRID_SIZE:
                _mgs = st.session_state.model_grid_size
                status_placeholder.error(
                    f"❌ 模型训练于 {_mgs}×{_mgs}，当前为 {GRID_SIZE}×{GRID_SIZE}，维度不匹配。"
                )
            else:
                bfs_result = bfs_solve(wall_map.astype(np.int32), start, goal)
                if not bfs_result["success"]:
                    status_placeholder.error("❌ 该迷宫配置无解，请换起终点。")
                else:
                    with st.spinner("🤖 DQN 推理中…"):
                        dqn_path = dqn_rollout(model, wall_map, start, goal)

                    ai_steps  = len(dqn_path) - 1
                    bfs_steps = bfs_result["steps"]
                    success   = (dqn_path[-1] == goal)
                    por       = round(bfs_steps / ai_steps, 4) if (success and ai_steps > 0) else 0.0

                    st.session_state.dqn_path    = dqn_path
                    st.session_state.bfs_path    = bfs_result["path"]
                    st.session_state.metrics     = {
                        "ai_steps": ai_steps, "bfs_steps": bfs_steps,
                        "success": success, "por": por,
                    }
                    # 启动帧动画
                    st.session_state.anim_running = True
                    st.session_state.anim_step    = 0
                    st.session_state.anim_path    = dqn_path
                    st.rerun()

        # ── 动画驱动（session_state 帧推进）──────────────────────────────
        if st.session_state.anim_running:
            step_i   = st.session_state.anim_step
            anim_p   = st.session_state.anim_path
            total    = len(anim_p)
            status_placeholder.info(f"🎬 动画播放中… {step_i + 1}/{total}")

            fig = build_maze_figure(
                wall_map, start, goal,
                dqn_path=anim_p,
                bfs_path=st.session_state.bfs_path,
                highlight_dqn_step=step_i,
            )
            st.plotly_chart(fig, use_container_width=False, key=f"anim_{step_i}")

            if step_i + 1 < total:
                time.sleep(ANIM_DELAY)
                st.session_state.anim_step += 1
                st.rerun()
            else:
                st.session_state.anim_running = False
                m = st.session_state.metrics
                ok = m["success"]
                status_placeholder.success(
                    f"{'✅' if ok else '❌'} DQN 寻路{'成功' if ok else '失败'}！"
                    f"  AI 步数 = **{m['ai_steps']}**  |  BFS 最短 = **{m['bfs_steps']}**"
                )

        # ── 静态迷宫图 ────────────────────────────────────────────────────
        elif not run_dqn:
            fig = build_maze_figure(
                wall_map, start, goal,
                dqn_path=st.session_state.dqn_path,
                bfs_path=st.session_state.bfs_path,
                highlight_dqn_step=-1,
            )
            st.plotly_chart(fig, use_container_width=False, key="maze_static")

        # ── 指标仪表盘 ───────────────────────────────────────────────────
        m = st.session_state.metrics
        if m:
            ai_s   = m["ai_steps"]
            bfs_s  = m["bfs_steps"]
            por    = m["por"]
            ok     = m["success"]

            # POR 分级颜色
            if ok and por >= 0.99:
                por_cls  = "por-perfect"
                por_text = f"{por:.2f} 🏆 100% Perfect"
            elif ok and por >= 0.75:
                por_cls  = "por-good"
                por_text = f"{por:.2f} 👍 Good"
            elif ok:
                por_cls  = "por-bad"
                por_text = f"{por:.2f} ⚠️ Sub-optimal"
            else:
                por_cls  = "por-bad"
                por_text = "N/A ❌ 未到达终点"

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-label'>🤖 AI 实际步数</div>
                  <div class='metric-value'>{ai_s}</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-label'>📐 BFS 理论最短</div>
                  <div class='metric-value'>{bfs_s}</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""
                <div class='metric-card' style='background:linear-gradient(135deg,#11998e,#38ef7d)'>
                  <div class='metric-label'>⚡ Path Optimality Ratio</div>
                  <div class='metric-value {por_cls}'>{por_text}</div>
                </div>""", unsafe_allow_html=True)

            with st.expander("📊 指标说明"):
                st.markdown("""
| 指标 | 含义 |
|------|------|
| **AI 实际步数** | DQN Agent 从起点走到终点（或超时）所用的总步数 |
| **BFS 理论最短** | BFS 算法计算的绝对最短路径步数（Ground Truth）|
| **Path Optimality Ratio** | `BFS步数 / AI步数`，越接近 **1.00** 越完美。等于 1.00 说明 AI 走出了与 BFS 完全相同的最短路！ |
                """)

    # ── 页脚 ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#95A5A6;font-size:12px'>"
        "DQN Maze Solver · PyTorch + Gymnasium + Streamlit · "
        "Hugging Face Spaces Demo"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
