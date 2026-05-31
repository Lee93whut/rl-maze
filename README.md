---
title: DQN Maze Pathfinding Demo
emoji: 🧩
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# RL Maze Navigator

### Benchmarking DQN variants on procedurally-generated mazes · SPL evaluation · 74% Holdout success rate

[![CI](https://github.com/Lee93whut/rl-maze/actions/workflows/test.yml/badge.svg)](https://github.com/Lee93whut/rl-maze/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**在线 Demo**：[Hugging Face Spaces](https://huggingface.co/spaces/lil58/interview) · **代码**：[GitHub](https://github.com/Lee93whut/rl-maze) · **实验记录**：[docs/experiment_log.md](docs/experiment_log.md)

训练 4 种 DQN 变体在随机 10×10 迷宫中自主寻路。每局随机生成迷宫、随机选取起终点，测试智能体的导航**泛化**能力，而非记忆固定路径。

---

## 算法对比结果（Round 3，最终）

> Holdout 评估：100 张训练中**从未见过**的独立地图（seed+200000），ε=0 贪心推理。  
> 指标：[SPL](https://arxiv.org/abs/1807.06757)（Anderson et al. 2018，导航领域标准评估指标）。  
> Round 3 超参：`buffer=80000`、`target_update_freq=1500`、`distance_shaping_alpha=0.5`，Double DQN 单算法验证。

| 算法 | 成功率 | SPL | 峰值成功率 | 收敛 Episode |
|------|:------:|:---:|:---------:|:-----------:|
| **Double DQN** (R3) | **74.0%** | **0.735** | **84.0%** | 3750 |
| Double DQN (R2) | 64.0% | 0.633 | 74.0% | 3300 |
| Vanilla DQN (R1) | 56.0% | 0.559 | — | 1921 |
| Double DQN (R1) | 61.0% | 0.605 | — | 948 |
| Dueling DQN (R1) | 45.0% | 0.445 | — | 759 |
| Double + Dueling (R1) | 43.0% | 0.425 | — | 1843 |

> R1 为随机起终点初版超参（训练量不足，供参考）；R2/R3 为逐轮超参消融后的结果。完整演进见 [docs/experiment_log.md](docs/experiment_log.md)。

**R2 → R3 成功率对比（Double DQN）**：

![R2 vs R3 成功率对比](docs/assets/compare/cmp_eval_success_rate_r2_vs_r3.png)

---

## 项目亮点

### 1. TensorBoard 三解耦看板

训练过程写入三类指标，X 轴语义不同，互不混用：

| 看板前缀 | X 轴 | 记录频率 | 指标 |
|---|---|---|---|
| `Backend_Net/` | `global_update_steps` | 每次梯度更新 | Loss、Avg Q Value、Grad Norm |
| `Frontend_Env/` | `episode` | 每局结束 | Reward、Steps、Success Rate、Epsilon |
| `Evaluation_Exam/` | `episode` | 每 N 局 | 盲测成功率、SPL |

三类 X 轴对齐不同事件频率，避免将梯度步数与训练局数混在同一坐标轴产生误导。

### 2. Episode 级 Warmup 机制

前 `warmup_episodes` 局固定 ε=1.0 纯随机探索，不执行任何梯度更新，先填充回放池再开始学习：

```python
in_warmup = (episode <= warmup_episodes)
cur_eps   = 1.0 if in_warmup else epsilon          # warmup 期间不衰减

if not in_warmup and buffer.is_ready(batch_size):  # warmup 结束后才更新
    loss, avg_q, grad_norm = optimize_model(...)

if not in_warmup:                                  # warmup 结束后才衰减
    epsilon = max(eps_end, epsilon * eps_decay)
```

相比 step 级 warmup，episode 级保证每局数据完整收集，避免半局数据污染早期 Q 值估计。

### 3. SPL 评估指标（Anderson et al. 2018）

$$\text{SPL} = \frac{1}{N} \sum_{i=1}^{N} S_i \cdot \frac{\ell_{i}^{\ast}}{\max(\ell_{i}^{\ast},\; p_i)}$$

- $S_i$：第 i 局是否成功（0/1）
- $\ell_{i}^{\ast}$：BFS 最短路径步数（Ground Truth）
- $p_i$：Agent 实际移动步数（排除撞墙步）
- **失败局整项贡献 0**，比纯成功率更严格，同时惩罚绕路行为

与 HabitatAI、EmbodiedQA 等导航 Benchmark 使用相同评估体系。

### 4. 势函数距离 Shaping（Ng et al. 1999）

每步额外奖励 = `α × (移动前曼哈顿距离 − 移动后曼哈顿距离)`，缓解稀疏奖励问题：

```python
if self.distance_shaping_alpha != 0.0:
    reward += self.distance_shaping_alpha * (dist_before - dist_after)
```

撞墙步位置不变，不触发 shaping，避免撞墙获得零 shaping 奖励误导策略。`α=0.5` 使 shaping 幅度为基础奖励（-1）的 50%，提供方向感但不压过终点奖励（+100）。符合 Ng et al. (1999) 的势函数 shaping 理论，保证最优策略不变。

### 5. Anti-Loop 双层防护

**训练阶段**：对重复访问格子施加递进奖励惩罚，引导 Q 函数主动规避循环路径：

```python
if revisit_penalty != 0.0 and not info.get("hit_wall", False):
    visit_cnt = ep_visited.get(cur_pos, 0)
    if visit_cnt > 0:
        reward += revisit_penalty * visit_cnt  # 访问次数越多，惩罚越重
    ep_visited[cur_pos] = visit_cnt + 1
```

**推理阶段**（Demo）：对高频重复访问的格子施加 Q 值惩罚，作为额外安全网：

```python
visit_cnt = visited_count.get(cur_pos, 0)
if visit_cnt >= 2:
    q_values[action] -= 3.0 * visit_cnt
```

两层机制职责分离：训练层修改 reward shaping 使 Q 函数内化回避循环；推理层直接修正 Q 值作为兜底，不影响训练分布。

### 6. BFS 连通性保证

`reset()` 内嵌 BFS 验证，确保每张迷宫起点→终点绝对可达，排除"任务本身不可完成"对训练信号的干扰：

```python
while True:
    self._wall_map = generate_maze(self.grid_size, self.obstacle_density, self.np_random)
    if bfs_reachable(self._wall_map, self._agent_pos, self._goal_pos):
        break
```

### 7. 唯一随机源设计

**`seed` 固定的是评估集的地图分布**，使不同算法、不同 Round 的 Holdout 指标可以在相同测试条件下横向对比——而不是复现网络参数初始化，也不是复现训练轨迹。

Holdout 评估时固定 `seeds = range(200000, 200100)`，每次评估用完全相同的 100 张地图，排除"某次恰好抽到简单地图"对成功率/SPL 的干扰。训练过程本身不固定 seed，每局随机生成新地图是泛化训练的一部分。

`super().reset(seed=seed)` 初始化 `self.np_random`，迷宫生成和起终点采样全部通过这同一个 Generator 完成：

```python
super().reset(seed=effective_seed)       # 初始化 self.np_random

self._wall_map = generate_maze(
    self.grid_size, self.obstacle_density,
    self.np_random                       # 传入同一个 Generator
)
```

---

## 技术架构

```
config.yaml
    │
    ▼
src/train.py ──────────────────────────────────────────────────────┐
    │  读取超参数                                                    │
    ├── MazeEnv (maze_env/env.py)   Gymnasium 标准环境接口           │
    │       ├── generator.py        随机迷宫生成 + BFS 连通性校验     │
    │       ├── actions.py          Action 枚举 + DELTAS 方向向量     │
    │       └── renderer.py         ASCII 渲染器                     │
    ├── DQNNetwork / DuelingDQNNetwork (src/model.py)               │
    ├── ReplayBuffer (src/replay_buffer.py)                         │
    └── maze_env/bfs.py             BFS 最短路径 / SPL Ground Truth  │
                                                                    │
app.py (Streamlit Web Demo)  ◄──── results/*.pth (训练权重) ◄───────┘
    ├── MazeEnv                     迷宫生成与状态渲染
    ├── DQNNetwork / DuelingDQNNetwork  加载权重执行推理
    ├── bfs.py                      可视化对比最短路径
    └── Plotly go.Heatmap           交互式迷宫可视化
```

---

## 快速开始

```bash
git clone https://github.com/Lee93whut/rl-maze.git && cd rl-maze

# 安装（maze_env 包 + 训练依赖）
pip install -e ".[dev]"
pip install -r requirements.txt

# 下载训练权重（从 HF Spaces，约 4×4MB）
python scripts/download_weights.py

# 启动 Web Demo（本地）
streamlit run app.py

# Docker 本地验证
docker build -t maze-dqn-demo .
docker run --rm -p 7860:7860 maze-dqn-demo
```

> 权重文件不托管在 GitHub（二进制文件不适合 git），统一存放于  
> [HF Spaces lil58/interview](https://huggingface.co/spaces/lil58/interview)。  
> `download_weights.py` 会跳过已存在的文件，重复执行安全。

```bash
# 从头自行训练（无需下载权重）
python src/train.py --config config.yaml

# 调试模式（5×5 迷宫，快速验证代码正确性）
python src/train.py --config config.yaml --overfit

# 查看训练曲线
tensorboard --logdir runs/

# 生成算法对比报告
python src/report.py
```

---

## 项目结构

```
rl-maze/
├── app.py                      Web Demo 主程序（Streamlit + Plotly）
├── config.yaml                 训练与环境超参数配置
├── pyproject.toml              包元数据 + pytest/coverage 配置
├── requirements.txt            运行时依赖
├── Dockerfile                  Hugging Face Spaces Docker SDK 部署配置
│
├── src/
│   ├── train.py                DQN 训练主循环（Warmup + 三看板 + 盲测评估）
│   ├── model.py                DQNNetwork / DuelingDQNNetwork（3-Conv + 2-FC）
│   ├── replay_buffer.py        环形经验回放池
│   └── report.py               四算法 Holdout 横向对比报告生成器
│
├── maze_env/
│   ├── __init__.py             包入口（MazeEnv / Action / bfs）
│   ├── env.py                  Gymnasium 标准环境（连通性保证、唯一随机源）
│   ├── bfs.py                  BFS 最短路径算法
│   ├── generator.py            随机迷宫生成器
│   ├── actions.py              Action 枚举与 DELTAS 方向向量
│   └── renderer.py             ASCII 终端渲染器
│
├── docs/
│   ├── experiment_log.md       逐轮训练记录（配置 + 结果 + 诊断）
│   ├── hyperparameter_study.md 超参分析报告（含论文依据）
│   ├── technical_report.md     技术报告（算法原理 + 训练曲线解读）
│   └── assets/                 训练曲线截图（Round 1/2/3 + 对比图）
│
└── tests/                      pytest 测试套件（90%+ 覆盖率）
    ├── test_00_gymnasium_check.py
    ├── test_01_spaces.py ~ test_09_validation.py
    ├── test_bfs.py
    ├── test_model.py
    ├── test_replay_buffer.py
    └── test_train.py
```

---

## 超参演进

| 轮次 | 关键变更 | Holdout 成功率 | SPL |
|------|---------|:--------------:|:---:|
| Round 1 | 初版（`ep=2000`, `decay=0.995`） | 61.0% | 0.605 |
| Round 2 | `ep=6000`, `decay=0.9985` | 64.0% | 0.633 |
| Round 3 | `buffer=80k`, `target_freq=1500`, `shaping=0.5` | **74.0%** | **0.735** |

完整超参诊断与论文依据详见 [`docs/hyperparameter_study.md`](docs/hyperparameter_study.md)。

---

## 参考文献

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning*. **Nature**.
2. van Hasselt, Guez & Silver (2016). *Deep Reinforcement Learning with Double Q-learning*. **AAAI**.
3. Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. **ICML**.
4. Ng et al. (1999). *Policy invariance under reward transformations*. **ICML**.
5. Anderson et al. (2018). *On Evaluation of Embodied Navigation Agents*. arXiv:1807.06757.
6. Henderson et al. (2018). *Deep Reinforcement Learning that Matters*. **AAAI**.
