"""train.py —— DQN 训练主循环（三解耦看板版）

TensorBoard 三栏目架构
----------------------
📂 Backend_Net/  （后台大脑训练指标）
    横坐标：global_update_steps（每次反向传播写入一次）
    指标：Loss、Avg_Q_Value、Grad_Norm

📂 Frontend_Env/  （前台游戏交互指标）
    横坐标：episode_count（每局结束写入一次）
    指标：Episode_Reward、Episode_Steps、Rollout_Success_Rate、Global_Epsilon

📂 Evaluation_Exam/  （盲测闭卷考试指标）
    横坐标：episode_count（每 100 局写入一次，config: eval_every）
    时机：暂停训练，model.eval()，ε=0，50 张独立测试迷宫（config: num_test_mazes）
    指标：Test_Success_Rate、SPL（Anderson et al. 2018）

Warmup 机制
-----------
前 warmup_episodes 局（默认 200）：纯随机探索（ε=1.0），不执行任何梯度更新。
第 warmup_episodes+1 局起：ε 开始衰减，buffer 足够时开始梯度更新。

用法
----
python src/train.py --config config.yaml
python src/train.py --config config.yaml --overfit
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter

# 小网络在 CPU 上多线程反而因调度开销变慢，限制为 4 线程性能最优
# benchmark 实测：8线程 13.6s vs 16线程 528s（0.03x），4线程约快 2-3x
torch.set_num_threads(4)

# ── 项目内部模块 ──────────────────────────────────────────────────────────────
# maze_env 通过 `pip install -e .` 安装，可直接 import。
# src/ 通过 pyproject.toml packages.find 配置，同样作为包安装，可直接 import。
from src.model import DQNNetwork, DuelingDQNNetwork
from src.replay_buffer import ReplayBuffer
from maze_env import MazeEnv
from maze_env.bfs import bfs as _bfs
from maze_env.generator import bfs_reachable as _bfs_reachable


# ===========================================================================
# 模块级常量
# ===========================================================================

VALID_ALGORITHMS: frozenset[str] = frozenset({"vanilla", "double", "dueling", "double_dueling"})
"""支持的 DQN 变体算法名称集合（供外部检查或测试引用）。"""


# ===========================================================================
# 1. 可复现性种子锁
# ===========================================================================

def set_seed(seed: int) -> None:
    """锁死所有随机源，确保实验可复现。

    覆盖范围：
    - ``random``  —— ε-greedy 探索、ReplayBuffer 采样顺序
    - ``torch``   —— 网络权重初始化、GPU 计算
    - cudnn 确定性模式

    注：maze_env 使用 Gymnasium 注入的 ``self.np_random``（独立对象），
    不读取 numpy 全局状态，因此无需调用 ``np.random.seed()``。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================================
# 2. ε-Greedy 动作选择
# ===========================================================================

def select_action(
    state: np.ndarray,
    policy_net: nn.Module,
    epsilon: float,
    num_actions: int,
    device: torch.device,
) -> int:
    """ε-Greedy 策略：以 ε 概率随机探索，否则选 Q 值最大动作。"""
    if random.random() < epsilon:
        return random.randrange(num_actions)
    with torch.no_grad():
        s = torch.from_numpy(state).unsqueeze(0).to(device)
        q_values = policy_net(s)
        return int(q_values.argmax(dim=1).item())


# ===========================================================================
# 3. 单步梯度更新
# ===========================================================================

def optimize_model(
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    use_double: bool = False,
) -> tuple[float, float, float]:
    """从回放池采样 mini-batch，执行一步 DQN 梯度更新。

    Args:
        use_double: 若 True 使用 Double DQN 目标，消除过估计偏差（默认 False）。

    Returns:
        (loss, avg_q_value, grad_norm)  三个 Backend_Net 指标。
    """
    batch = buffer.sample(batch_size, device)

    states      = batch["states"]       # (B, 4, N, N)
    actions     = batch["actions"]      # (B,)
    rewards     = batch["rewards"]      # (B,)
    next_states = batch["next_states"]  # (B, 4, N, N)
    terminated_mask = batch["dones"]    # (B,)  terminated=1，truncated 存为 0

    # ── 当前 Q 值：Q(s, a) ────────────────────────────────────────────────
    q_all     = policy_net(states)                                    # (B, 4)
    q_current = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)      # (B,)
    avg_q     = float(q_all.detach().mean().item())

    # ── 目标 Q 值：r + γ · Q_target(s', argmax_policy) · (1 - terminated) ──
    # truncated 不屏蔽 bootstrap；仅 terminated（自然结束）屏蔽
    with torch.no_grad():
        if use_double:
            # Double DQN：policy_net 选动作，target_net 估值
            # 解耦"选哪个动作"与"该动作值多少"，消除 max 算子的过估计偏差
            # (van Hasselt et al., 2015)
            next_acts  = policy_net(next_states).argmax(dim=1, keepdim=True)  # (B,1)
            q_next_max = target_net(next_states).gather(1, next_acts).squeeze(1)
        else:
            # Vanilla DQN：target_net 直接取 max Q 值 (Mnih et al., 2015)
            q_next_max = target_net(next_states).max(dim=1).values
        q_target = rewards + gamma * q_next_max * (1.0 - terminated_mask)

    # ── Huber Loss & 反向传播 ─────────────────────────────────────────────
    loss = nn.functional.smooth_l1_loss(q_current, q_target)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(
        nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0).item()
    )
    optimizer.step()

    return float(loss.item()), avg_q, grad_norm


# ===========================================================================
# 4-pre. 共用辅助：采样连通起终点（训练侧与评估侧统一调用）
# ===========================================================================

def _sample_connected_start_goal(
    wall_map: np.ndarray,
    grid_size: int,
    rng: np.random.Generator,
    default_start: tuple[int, int],
    default_goal:  tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """从 wall_map 的内圈自由格中随机采样一对 BFS 连通的起终点。

    采用有限重试 + fallback 设计，杜绝任何极端地图下的无限循环：

    * 先筛选内圈（去除边界外圈）自由格列表 ``inner``。
    * 至多重试 ``len(inner) ** 2`` 次（覆盖所有排列对数量级）；
      每次用 ``rng.choice(..., replace=False)`` 一行完成不重复采样，
      无需额外去重循环。
    * 若耗尽重试仍未找到连通对（极端高密度地图、所有自由格互不连通），
      安全回退到环境默认起终点，训练/评估进程不会挂死。

    Args:
        wall_map:      当前地图的墙图（0=自由，1=墙）。
        grid_size:     地图边长，用于过滤边界外圈。
        rng:           调用方传入的 ``np.random.Generator``，保证随机流可控。
        default_start: fallback 用的默认起点（通常为 ``env.agent_pos``）。
        default_goal:  fallback 用的默认终点（通常为 ``env.goal_pos``）。

    Returns:
        ``(start_pos, goal_pos)`` 元组，均为 ``(row, col)`` 格式。
    """
    rows_free, cols_free = np.where(wall_map == 0)
    inner: list[tuple[int, int]] = [
        (int(r), int(c)) for r, c in zip(rows_free, cols_free)
        if 0 < r < grid_size - 1 and 0 < c < grid_size - 1
    ]
    if len(inner) < 2:
        # 自由格不足，直接 fallback
        return default_start, default_goal

    max_retries = len(inner) ** 2
    for _ in range(max_retries):
        idxs = rng.choice(len(inner), size=2, replace=False)   # 天然不重复，无需去重循环
        start_pos = inner[idxs[0]]
        goal_pos  = inner[idxs[1]]
        if _bfs_reachable(wall_map, start_pos, goal_pos):
            return start_pos, goal_pos

    # 耗尽重试：极端地图（所有自由格互不连通），安全回退
    return default_start, default_goal


# ===========================================================================
# 4. Evaluation_Exam：盲测闭卷考试
# ===========================================================================

def run_evaluation(
    policy_net: nn.Module,
    grid_size: int,
    obstacle_density: float,
    max_steps: int,
    device: torch.device,
    test_seeds: list[int],
    reward_goal: float,
    reward_wall_hit: float,
    reward_step: float,
    random_start_goal: bool = False,
) -> tuple[float, float]:
    """在 test_seeds 指定的迷宫上盲测，返回 (success_rate, spl)。

    盲测规则
    --------
    * model.eval()，ε=0（完全贪心）。
    * 测试迷宫由调用方传入固定 seed 列表，整个训练期间测试集恒定，
      使 TensorBoard 曲线的波动能真实反映 AI 能力变化，而非地图难度变化。
    * random_start_goal=True 时，每张地图用派生种子从自由格中随机选取起终点，
      与训练分布保持一致，避免 train/test 分布偏差。
    * SPL（Success-weighted Path Length，Anderson et al. 2018）：
        SPL = (1/N) × Σ S_i × ℓ*_i / max(ℓ*_i, p_i)
      其中 S_i∈{0,1} 为成功标志，ℓ*_i 为 BFS 最短步数，p_i 为实际移动步数。
      失败局 S_i=0，整项贡献 0，与主流导航论文定义一致。
    """
    policy_net.eval()

    successes:  list[int]   = []
    spl_terms:  list[float] = []

    # 构造一次环境，循环内通过 reset(seed=...) 切换地图，符合 Gymnasium 惯用模式
    env = MazeEnv(
        grid_size=grid_size,
        obstacle_density=obstacle_density,
        max_steps=max_steps,
        reward_goal=reward_goal,
        reward_wall_hit=reward_wall_hit,
        reward_step=reward_step,
    )

    with torch.no_grad():
        for seed_i in test_seeds:
            # ── Step 1：用 seed 生成地图 ──────────────────────────────────
            obs, _ = env.reset(seed=seed_i)

            if random_start_goal:
                # ── Step 2：从自由格随机选起终点（派生种子，保证确定性）──
                wall_map_copy = env.wall_map.copy()
                rng = np.random.default_rng(seed_i ^ 0xABCD1234)
                # 采样连通起终点（有限重试 + fallback，防止极端地图挂死）
                start_pos, goal_pos = _sample_connected_start_goal(
                    wall_map_copy, grid_size, rng,
                    default_start=env.agent_pos,
                    default_goal=env.goal_pos,
                )
                # Step 3：注入 wall_map + 随机起终点重置
                obs, _ = env.reset(seed=seed_i, options={
                    "wall_map": wall_map_copy,
                    "start":    start_pos,
                    "goal":     goal_pos,
                })
            else:
                start_pos = env.agent_pos
                goal_pos  = env.goal_pos

            state    = obs.astype(np.float32)
            done     = False
            ai_steps = 0

            while not done:
                action = select_action(
                    state, policy_net, epsilon=0.0,
                    num_actions=env.action_space.n, device=device,
                )
                next_obs, _, terminated, truncated, info = env.step(action)
                state  = next_obs.astype(np.float32)
                done   = terminated or truncated
                ai_steps += 1

            success = int(info.get("success", False))
            successes.append(success)

            # ── SPL 计算（Anderson et al. 2018）─────────────────────────
            # 成功：S_i=1，贡献 ℓ*_i / max(ℓ*_i, p_i)
            # 失败：S_i=0，整项贡献 0.0
            hit_wall_count    = info.get("hit_wall_count", 0)
            actual_move_steps = ai_steps - hit_wall_count  # p_i：排除撞墙步

            if success and actual_move_steps > 0:
                bfs_result = _bfs(
                    env.wall_map.astype(np.int32),
                    start=start_pos,
                    end=goal_pos,
                )
                if bfs_result["success"] and bfs_result["steps"] > 0:
                    l_star   = bfs_result["steps"]
                    spl_term = l_star / max(l_star, actual_move_steps)
                    spl_terms.append(spl_term)
                else:
                    spl_terms.append(0.0)
            else:
                spl_terms.append(0.0)

    policy_net.train()

    success_rate = float(np.mean(successes)) * 100.0
    spl          = float(np.mean(spl_terms)) if spl_terms else 0.0
    return success_rate, spl


# ===========================================================================
# 5. 训练主函数
# ===========================================================================

def train(cfg: dict[str, Any], overfit_mode: bool = False) -> None:
    """DQN 训练主循环（三解耦看板 + Episode 级 Warmup）。

    Args:
        cfg:          完整的 YAML 配置字典。
        overfit_mode: 若为 True，使用 overfit 节参数运行 5×5 超小迷宫验收。
    """
    # ── 合并配置 ──────────────────────────────────────────────────────────
    maze_cfg   = dict(cfg.get("maze", {}))
    reward_cfg = dict(cfg.get("rewards", {}))
    dqn_cfg    = dict(cfg.get("dqn", {}))
    ov         = cfg.get("overfit", {})       # 提前定义，消除 possibly-undefined 警告

    # ── 算法变体解析（提前，run_tag 依赖此值）────────────────────────────────
    # overfit 节若配置了 algorithm 字段，用 ov 中的值覆盖；否则回落到 dqn 节
    _algo_src  = ov if (overfit_mode and "algorithm" in ov) else dqn_cfg
    algorithm  = str(_algo_src.get("algorithm", "vanilla")).strip().lower()
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(
            f"不支持的 algorithm='{algorithm}'，合法值：{sorted(VALID_ALGORITHMS)}"
        )
    use_double  = "double"  in algorithm   # double / double_dueling → True
    use_dueling = "dueling" in algorithm   # dueling / double_dueling → True

    if overfit_mode:
        maze_cfg.update({
            "grid_size":        ov.get("grid_size",        5),
            "obstacle_density": ov.get("obstacle_density", 0.0),
            "max_steps":        ov.get("max_steps",        50),
        })
        dqn_cfg.update({
            "num_episodes":      ov.get("num_episodes",       500),
            "epsilon_decay":     ov.get("epsilon_decay",      0.990),
            "warmup_episodes":   ov.get("warmup_episodes",    50),
            "batch_size":        ov.get("batch_size",         32),
            "target_update_freq":ov.get("target_update_freq", 100),
            "print_every":       ov.get("print_every",        50),
            "eval_every":        ov.get("eval_every",         50),
            "num_test_mazes":    ov.get("num_test_mazes",     10),
        })
        run_tag = f"overfit_5x5_{algorithm}"
        print("=" * 60)
        print("  [OVERFIT MODE] 5×5 超小迷宫过拟合调试")
        print("=" * 60)
    else:
        run_tag = f"train_{algorithm}"

    # ── 超参数解包 ─────────────────────────────────────────────────────────
    seed             = int(dqn_cfg.get("seed", 42))
    grid_size        = int(maze_cfg.get("grid_size", 10))
    obstacle_density = float(maze_cfg.get("obstacle_density", 0.25))
    max_steps        = int(maze_cfg.get("max_steps", 50))
    num_episodes     = int(dqn_cfg.get("num_episodes", 2000))
    buffer_capacity  = int(dqn_cfg.get("buffer_capacity", 20000))
    batch_size       = int(dqn_cfg.get("batch_size", 64))
    lr               = float(dqn_cfg.get("learning_rate", 5e-4))
    gamma            = float(dqn_cfg.get("gamma", 0.99))
    eps_start        = float(dqn_cfg.get("epsilon_start", 1.0))
    eps_end          = float(dqn_cfg.get("epsilon_end", 0.05))
    eps_decay        = float(dqn_cfg.get("epsilon_decay", 0.995))
    target_freq      = int(dqn_cfg.get("target_update_freq", 500))
    warmup_episodes  = int(dqn_cfg.get("warmup_episodes", 200))  # episode-based warmup
    log_dir          = str(dqn_cfg.get("log_dir", "runs"))
    save_dir         = str(dqn_cfg.get("save_dir", "results"))
    success_window   = int(dqn_cfg.get("success_window", 100))
    save_window      = int(dqn_cfg.get("save_window", 50))
    print_every        = int(dqn_cfg.get("print_every", 10))
    eval_every         = int(dqn_cfg.get("eval_every", 50))
    num_test_mazes     = int(dqn_cfg.get("num_test_mazes", 20))
    random_start_goal  = bool(dqn_cfg.get("random_start_goal", False))

    reward_goal      = float(reward_cfg.get("goal", 100.0))
    reward_wall_hit  = float(reward_cfg.get("wall_hit", -10.0))
    reward_step_r    = float(reward_cfg.get("step", -1.0))
    # 注：revisit_penalty 已移除——违反马尔可夫性（状态不含访问历史）。
    # 改为在环境观测中加入 visited_map 第4通道，从状态层面编码访问历史。

    # 固定测试集：训练开始前生成，整个训练期间测试地图恒定，
    # 确保 TensorBoard 曲线波动反映 AI 能力变化而非地图难度变化。
    eval_seed_base  = seed + 100000
    TEST_SEEDS: list[int] = [eval_seed_base + i for i in range(num_test_mazes)]

    # ── Seed Lock ────────────────────────────────────────────────────────
    set_seed(seed)

    # ── 设备 ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}  |  Grid {grid_size}×{grid_size}  |  "
          f"Episodes {num_episodes}  |  Seed {seed}")
    print(f"[Algorithm] {algorithm.upper()}  |  "
          f"Net={'Dueling' if use_dueling else 'Vanilla'}  |  "
          f"Target={'Double' if use_double else 'Vanilla'}")
    print(f"[Warmup] 前 {warmup_episodes} 局纯随机探索，不执行梯度更新")

    # ── 环境（训练用）──────────────────────────────────────────────────────
    # 正常训练：不传 seed，每局 reset() 使用 Gymnasium 内部 RNG 续进，
    # 每局生成不同地图，迫使 Agent 学习通用导航策略而非路线记忆。
    # 过拟合调试模式：传入固定 seed，锁定单张地图快速验证算法收敛性。
    env_seed = int(ov.get("seed", 0)) if overfit_mode else None
    env = MazeEnv(
        grid_size=grid_size,
        obstacle_density=obstacle_density,
        max_steps=max_steps,
        seed=env_seed,
        reward_goal=reward_goal,
        reward_wall_hit=reward_wall_hit,
        reward_step=reward_step_r,
    )

    # ── 网络 ───────────────────────────────────────────────────────────────
    NetClass   = DuelingDQNNetwork if use_dueling else DQNNetwork
    policy_net = NetClass(grid_size=grid_size).to(device)
    target_net = NetClass(grid_size=grid_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # ── 经验回放池 ─────────────────────────────────────────────────────────
    buffer = ReplayBuffer(capacity=buffer_capacity)

    # ── TensorBoard Writer ─────────────────────────────────────────────────
    timestamp  = time.strftime("%Y%m%d_%H%M%S")
    writer_dir = os.path.join(log_dir, f"{run_tag}_{timestamp}")
    writer     = SummaryWriter(log_dir=writer_dir)
    print(f"[TensorBoard] tensorboard --logdir={log_dir}")

    # ── 保存目录 ───────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_model_{run_tag}_{timestamp}.pth")

    # ── 滚动统计窗口 ───────────────────────────────────────────────────────
    reward_deque:  deque[float] = deque(maxlen=success_window)
    success_deque: deque[int]   = deque(maxlen=success_window)
    save_deque:    deque[float] = deque(maxlen=save_window)

    best_avg_reward   = float("-inf")
    best_eval_success = float("-inf")       # EVAL-based checkpoint 触发阈值
    epsilon           = eps_start           # warmup 期间固定为 1.0
    global_update_steps = 0                 # Backend_Net/ 横坐标
    total_env_steps     = 0                 # 全局环境交互步数（用于 Target Net 同步）

    print(f"\n{'─'*70}")
    print(f"{'Ep':>6} {'Reward':>8} {'Steps':>6} {'Eps':>7} "
          f"{'Loss':>8} {'AvgQ':>7} {'Suc%':>6} {'BestR':>8}")
    print(f"{'─'*70}")

    # =========================================================
    # 主训练循环
    # =========================================================
    for episode in range(1, num_episodes + 1):

        # ── Warmup 判断：前 warmup_episodes 局 ε 固定为 1.0 ──────────────
        in_warmup = (episode <= warmup_episodes)

        if random_start_goal and not overfit_mode:
            # 随机起终点训练：先 reset 取墙图，再从自由格随机选起终点重注入
            obs, _ = env.reset()
            _wall_map_train = env.wall_map.copy()
            # 采样连通起终点（有限重试 + fallback，与评估侧统一调用同一 helper）
            _train_rng = np.random.default_rng(int(env.np_random.integers(0, 2**31)))
            _start_t, _goal_t = _sample_connected_start_goal(
                _wall_map_train, grid_size, _train_rng,
                default_start=env.agent_pos,
                default_goal=env.goal_pos,
            )
            obs, _ = env.reset(options={
                "wall_map": _wall_map_train,
                "start":    _start_t,
                "goal":     _goal_t,
            })
        else:
            obs, _ = env.reset()
        state: np.ndarray = obs.astype(np.float32)

        ep_reward  = 0.0
        ep_steps   = 0
        ep_loss    = 0.0
        ep_avg_q   = 0.0
        ep_updates = 0
        done = False
        while not done:
            # ── 动作选择 ──────────────────────────────────────────────────
            cur_eps = 1.0 if in_warmup else epsilon
            action = select_action(
                state, policy_net, cur_eps,
                env.action_space.n, device,
            )

            # ── 环境交互 ──────────────────────────────────────────────────
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = next_obs.astype(np.float32)
            done = terminated or truncated

            # ── 存入回放池 ────────────────────────────────────────────────
            # 仅用 terminated 做 bootstrap mask：truncated 表示时间截断，
            # next_state 仍有价值，不应将 γ·Q(s') 归零（Gymnasium v0.26 语义）
            buffer.push(state, action, float(reward), next_state, terminated)

            state          = next_state
            ep_reward     += float(reward)
            ep_steps      += 1
            total_env_steps += 1

            # ── 梯度更新（Warmup 结束且 buffer 足够后才执行）──────────────
            if not in_warmup and buffer.is_ready(batch_size):
                loss, avg_q, grad_norm = optimize_model(
                    policy_net, target_net,
                    optimizer, buffer, batch_size, gamma, device,
                    use_double=use_double,
                )
                global_update_steps += 1
                ep_loss    += loss
                ep_avg_q   += avg_q
                ep_updates += 1

                # ── 📂 Backend_Net/ 每次梯度更新写入 ─────────────────────
                writer.add_scalar("Backend_Net/Loss",        loss,      global_update_steps)
                writer.add_scalar("Backend_Net/Avg_Q_Value", avg_q,     global_update_steps)
                writer.add_scalar("Backend_Net/Grad_Norm",   grad_norm, global_update_steps)

            # ── Target Net 硬拷贝 ─────────────────────────────────────────
            if (not in_warmup) and global_update_steps > 0 and \
               global_update_steps % target_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # ── 幕结束后统计 ──────────────────────────────────────────────────
        success = int(info.get("success", False))
        # warmup 期间为纯随机探索，不计入统计窗口，避免污染曲线
        if not in_warmup:
            reward_deque.append(ep_reward)
            success_deque.append(success)
            save_deque.append(ep_reward)

        avg_ep_loss    = ep_loss / ep_updates if ep_updates > 0 else 0.0
        avg_ep_q       = ep_avg_q / ep_updates if ep_updates > 0 else 0.0
        success_rate   = float(np.mean(success_deque)) * 100.0 if success_deque else 0.0
        avg_reward_win = float(np.mean(reward_deque)) if reward_deque else 0.0
        avg_save       = float(np.mean(save_deque)) if save_deque else float("-inf")

        # ── ε 衰减（Warmup 结束后才开始衰减）──────────────────────────────
        if not in_warmup:
            epsilon = max(eps_end, epsilon * eps_decay)

        # ── 📂 Frontend_Env/ 每局结束写入 ────────────────────────────────
        cur_eps_log = 1.0 if in_warmup else epsilon
        writer.add_scalar("Frontend_Env/Episode_Reward",        ep_reward,       episode)
        writer.add_scalar("Frontend_Env/Episode_Steps",         ep_steps,        episode)
        writer.add_scalar("Frontend_Env/Rollout_Success_Rate",  success_rate,    episode)
        writer.add_scalar("Frontend_Env/Global_Epsilon",        cur_eps_log,     episode)
        writer.add_scalar("Frontend_Env/Avg_Reward_Window",     avg_reward_win,  episode)
        # Sample Efficiency 视角：以环境交互步为横坐标，
        # 便于与其他算法在相同样本量下对比学习效率（Warmup 期间跳过，避免污染曲线）
        if not in_warmup:
            writer.add_scalar("SampleEfficiency/Success_Rate",   success_rate, total_env_steps)
            writer.add_scalar("SampleEfficiency/Episode_Reward", ep_reward,    total_env_steps)

        # ── 📂 Evaluation_Exam/ 每 eval_every 局盲测（Warmup 结束后、非 overfit 模式才触发）──
        if episode % eval_every == 0 and not in_warmup and not overfit_mode:
            test_success_rate, test_spl = run_evaluation(
                policy_net=policy_net,
                grid_size=grid_size,
                obstacle_density=obstacle_density,
                max_steps=max_steps,
                device=device,
                test_seeds=TEST_SEEDS,
                reward_goal=reward_goal,
                reward_wall_hit=reward_wall_hit,
                reward_step=reward_step_r,
                random_start_goal=random_start_goal,
            )
            writer.add_scalar("Evaluation_Exam/Test_Success_Rate", test_success_rate, episode)
            writer.add_scalar("Evaluation_Exam/SPL",               test_spl,          episode)
            print(f"  [EVAL ep={episode:4d}]  "
                  f"Test_Success={test_success_rate:.1f}%  "
                  f"SPL={test_spl:.3f}  "
                  f"(越接近 1.0 越好，失败局贡献 0)")

            # ── EVAL-based checkpoint（R4 核心改动）──────────────────────────
            # 每次盲测成功率创新高时保存，保证 Holdout 对应训练过程中最佳泛化点。
            # 比"训练滚动奖励最高"更准确：训练奖励受地图难度随机性影响，
            # 与泛化能力相关性弱；盲测成功率直接度量真实泛化能力。
            if not in_warmup and test_success_rate > best_eval_success:
                best_eval_success = test_success_rate
                torch.save(
                    {
                        "episode":      episode,
                        "grid_size":    grid_size,
                        "state_dict":   policy_net.state_dict(),
                        "epsilon":      epsilon,
                        "eval_success": best_eval_success,
                        "algorithm":    algorithm,
                    },
                    best_model_path,
                )
                print(f"  [EVAL SAVE] EVAL 新高 {best_eval_success:.1f}% → 已保存 {best_model_path}")

        # ── Best Model Save（训练奖励，仅用于控制台 ✓ 标记，不再保存权重）────
        # 权重保存已移至 EVAL-based checkpoint（见上方 EVAL 块）。
        # 保留 save_deque / best_avg_reward 逻辑仅为在控制台打印 ✓ 标记，
        # 方便对照训练奖励高点与 EVAL 高点的时序关系。
        model_saved = False
        if not in_warmup and len(save_deque) >= save_window and avg_save > best_avg_reward:
            best_avg_reward = avg_save
            model_saved = True

        # ── 控制台打印 ────────────────────────────────────────────────────
        if episode % print_every == 0 or episode == 1:
            # 每 20 行数据前重打一次表头，方便在长日志中快速定位列含义
            _rows_printed = (episode // print_every)
            if episode == 1 or _rows_printed % 20 == 0:
                print(f"{'─'*70}")
                print(f"{'Ep':>6} {'Reward':>8} {'Steps':>6} {'Eps':>7} "
                      f"{'Loss':>8} {'AvgQ':>7} {'Suc%':>6} {'BestR':>8}")
                print(f"{'─'*70}")
            warmup_flag = " [WARMUP]" if in_warmup else ""
            saved_flag  = " ✓" if model_saved else ""
            print(
                f"{episode:>6d}  "
                f"{ep_reward:>8.1f}  "
                f"{ep_steps:>6d}  "
                f"{cur_eps_log:>7.4f}  "
                f"{avg_ep_loss:>8.4f}  "
                f"{avg_ep_q:>7.3f}  "
                f"{success_rate:>5.1f}%"
                f"{saved_flag}{warmup_flag}"
            )

    # ── 训练结束 ──────────────────────────────────────────────────────────
    writer.close()
    print(f"\n{'═'*70}")
    print(f"  训练完成。共 {num_episodes} 个 Episode，{total_env_steps} 环境步，"
          f"{global_update_steps} 梯度步。")
    print(f"  Best Avg Reward（近{save_window}局）: {best_avg_reward:.2f}")
    print(f"  最终 ε = {epsilon:.4f}")
    print(f"  模型已保存至：{best_model_path}")
    print(f"  TensorBoard：tensorboard --logdir={log_dir}")
    print(f"{'═'*70}\n")

    # ── Holdout Test：训练后一次性最终评估（仅正常训练模式执行）─────────────
    # Holdout 地图（seed+200000）在整个训练过程中从未使用，
    # 是唯一可对外报告的无偏泛化性能数字。
    if not overfit_mode and os.path.exists(best_model_path):
        print("=" * 70)
        print("  [HOLDOUT TEST] 加载 best_model.pth，在 100 张全新地图上最终评估")
        print("=" * 70)
        holdout_seed_base = seed + 200000
        holdout_seeds     = [holdout_seed_base + i for i in range(100)]

        checkpoint  = torch.load(best_model_path, map_location=device, weights_only=True)
        HoldoutNet  = DuelingDQNNetwork if use_dueling else DQNNetwork
        holdout_net = HoldoutNet(grid_size=grid_size).to(device)
        holdout_net.load_state_dict(checkpoint["state_dict"])

        holdout_sr, holdout_spl = run_evaluation(
            policy_net=holdout_net,
            grid_size=grid_size,
            obstacle_density=obstacle_density,
            max_steps=max_steps,
            device=device,
            test_seeds=holdout_seeds,
            reward_goal=reward_goal,
            reward_wall_hit=reward_wall_hit,
            reward_step=reward_step_r,
            random_start_goal=random_start_goal,
        )
        print(f"  Holdout Success Rate : {holdout_sr:.1f}%  (100 张独立地图)")
        print(f"  Holdout SPL          : {holdout_spl:.3f}  (Success-weighted Path Length，越接近 1.0 越好)")
        print(f"  ← 此数字为唯一可信的最终泛化性能，可对外报告。")
        print("=" * 70 + "\n")

    # ── 过拟合模式验收断言 ─────────────────────────────────────────────────
    if overfit_mode:
        overfit_eval_seed  = int(ov.get("seed", 0))
        # 用固定 overfit 训练图重复 20 次，ε=0 冻结网络，给出干净的验收数字
        overfit_eval_seeds = [overfit_eval_seed] * 20
        final_success_rate, final_spl = run_evaluation(
            policy_net=policy_net,
            grid_size=grid_size,
            obstacle_density=obstacle_density,
            max_steps=max_steps,
            device=device,
            test_seeds=overfit_eval_seeds,
            reward_goal=reward_goal,
            reward_wall_hit=reward_wall_hit,
            reward_step=reward_step_r,
            random_start_goal=False,   # overfit 模式始终固定起终点
        )
        print(f"[OVERFIT 验收] 固定地图（seed={overfit_eval_seed}）成功率: "
              f"{final_success_rate:.1f}%  SPL={final_spl:.3f}")
        if final_success_rate >= 80.0:
            print("✅  过拟合测试通过：Agent 已在 5×5 迷宫上充分收敛。")
        else:
            print("⚠️  过拟合测试未达预期（成功率 < 80%），请检查超参数。")


# ===========================================================================
# 6. 入口
# ===========================================================================

def _parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="DQN 迷宫训练脚本（三解耦看板版）")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="YAML 配置文件路径（默认：config.yaml）",
    )
    parser.add_argument(
        "--overfit", action="store_true",
        help="启用 5×5 过拟合调试模式",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["vanilla", "double", "dueling", "double_dueling"],
        help="覆盖 config.yaml 中的 algorithm 字段（可选：vanilla/double/dueling/double_dueling）",
    )
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        candidates = [
            config_path,
            Path(__file__).resolve().parent.parent / config_path,
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    overfit_mode = args.overfit

    # CLI --algorithm 优先级最高，覆盖 config.yaml 中的对应节
    if args.algorithm is not None:
        key = "overfit" if overfit_mode else "dqn"
        cfg.setdefault(key, {})["algorithm"] = args.algorithm
        print(f"[CLI] --algorithm 覆盖 config.yaml：algorithm = {args.algorithm}")

    train(cfg, overfit_mode=overfit_mode)
