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

### 4 种 DQN 变体在随机 10×10 迷宫上的系统性消融实验 · SPL 评估 · Holdout 成功率 84%

[![CI](https://github.com/Lee93whut/rl-maze/actions/workflows/test.yml/badge.svg)](https://github.com/Lee93whut/rl-maze/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**在线 Demo**：[Hugging Face Spaces](https://huggingface.co/spaces/lil58/interview) · **代码**：[GitHub](https://github.com/Lee93whut/rl-maze) · **实验记录**：[docs/experiment_log.md](docs/experiment_log.md)

---

## 为什么看这个项目

随机起终点 + 随机地图将状态空间扩大约 40×，使 DQN 在此设定下初版只能达到 **61% 成功率**。
本项目通过 **4 轮系统性消融**（超参调优 → 算法横评），将 Holdout 成功率提升至 **84%**（+23pp：超参调优 +17pp，61%→78%；算法切换至 Dueling DQN +6pp，78%→84%）。

过程中发现并修复了一类常见的 RL 工程错误——**reward shaping 违反马尔可夫性**——并提供完整的诊断证据链与理论分析，给出了训练侧的根本解法。

> 这不只是"跑通了 DQN"，而是一次带完整方法论的 RL 实验：**有对照组、有问题诊断、有单变量消融、有算法归因**。

---

## 最终结果

> **Holdout 评估**：100 张训练中从未见过的独立地图（seed+200000），ε=0 贪心推理。
> ⚠️ n=100 单次跑点，95% CI ≈ ±5pp；#1 Dueling（84%）与 #2 Double+Dueling（81%）差距 3pp，处于 CI 边缘，排名供参考，严格对比需多次独立运行取均值±标准差（Henderson et al., 2018）。

### R4 四算法横向消融（固定最优超参，唯一变量 = 算法）

| 排名 | 算法 | Holdout 成功率 | SPL | EVAL→Holdout Gap |
|:---:|------|:-------------:|:---:|:----------------:|
| 🥇 | **Dueling DQN** | **84.0%** | **0.817** | −6pp（泛化最稳） |
| 🥈 | Double + Dueling | 81.0% | 0.793 | −9pp |
| 🥉 | Double DQN | 78.0% | 0.773 | −10pp |
| 4️⃣ | Vanilla DQN | 75.0% | 0.726 | −19pp |

> 排名基于单次跑点，相邻差距处于统计置信区间边缘，仅代表本次实验观测结果。

![R4 四算法 EVAL 成功率对比](docs/assets/round4/r4_eval_success_rate_all_algos.png)

### 超参演进纵向对比（Double DQN，保持算法一致）

| 轮次 | 核心变更 | Holdout 成功率 | SPL |
|------|---------|:-------------:|:---:|
| Round 1 | 初版超参（`ep=2000`, `decay=0.995`） | 61.0% | 0.605 |
| Round 2 | `ep=6000`, `decay=0.9985` | 64.0% | 0.633 |
| Round 3 | `buffer=80k`, `target=1500`, `shaping=0.5` | 74.0% | 0.735 |
| **Round 4** | visited_map 4通道 + EVAL checkpoint + BFS 连通性 | **78.0%** | **0.773** |

![R1→R4 超参演进纵向对比（Double DQN）](docs/assets/compare/cmp_eval_success_rate_r1_to_r4_double.png)

---

## 实验方法论

本项目遵循 Henderson et al. (2018) 的 RL 实验规范，分三个阶段执行：

```
阶段一：冒烟验证（Round 0）
    固定起终点，四算法均达 90%+
    → 验证训练流程无系统性 bug，建立对照组

阶段二：超参消融（Round 1–4）
    每轮只改一组变量，用问题诊断驱动下一轮变更
    R1 → R2：修复训练量 + 探索衰减（单变量）
    R2 → R3：修复 buffer + target + shaping（P3/P4/P5 同批修复）
    R3 → R4：修复 checkpoint 时序 + 连通性验证 + 状态编码

阶段三：算法横评（Round 4 续）
    固定 R4 最优超参，四算法串行训练
    唯一变量 = 网络架构与 Q 目标计算方式
```

**为什么先调超参再比算法**：先把超参调到最优，比的才是算法本身的能力差异；否则比的是"哪个算法对糟糕超参更鲁棒"。

完整逐轮记录见 [docs/experiment_log.md](docs/experiment_log.md)。

---

## 核心发现

### 发现一：reward shaping 的马尔可夫性陷阱（P9）

R4 首先尝试用 `revisit_penalty`（重复访问格子施加递进奖励惩罚）抑制推理时的循环行为，但训练至 ep=1000 时成功率仅 38%，持续低于基线 15pp，最终终止。

**根因**：Q-learning 的贝尔曼方程要求 $r(s,a,s')$ 仅依赖当前转移。`revisit_penalty` 使奖励依赖 episode 内访问历史（隐变量），相同的 $(s,a)$ 在不同时刻返回不同奖励，Q 函数在数学上无法收敛到唯一固定点。更严重的是训练/推理分布不一致——训练时奖励含惩罚项，推理时不含，策略崩溃不可修正。

**正确解法**：把历史信息从奖励空间移到状态空间。观测张量第四通道编码二值访问图（ch3=visited_map），Q 函数合法学习"已访问格价值低"的策略，马尔可夫性完整保持。

> 这一诊断具有通用性：**任何依赖 episode 内历史的奖励项都是这类错误的变体**，正确方向永远是状态编码。

### 发现二：Dueling 架构与本任务的结构适配性

随机起终点迷宫中存在大量"多动作等效"状态（死胡同、走廊段），Dueling 的 V(s)/A(s,a) 分解使 V(s) 流被所有动作共享——每次梯度更新中 V(s) 获得所有动作的梯度信号，更新频率是 A(s,a) 的 4 倍，估计更稳定、泛化更强。

实测证据：EVAL→Holdout Gap 仅 6pp（最小），而 Vanilla DQN 的 Gap 高达 19pp。

### 发现三：checkpoint 保存策略是系统性问题

R3 用训练滚动奖励触发 checkpoint，但训练奖励受随机地图难度影响，与泛化能力相关性弱（约 0.3–0.5）。实测 EVAL 峰值 84%、Holdout 仅 74%，差距 10pp 全部来自时序错位。

改为 EVAL-based checkpoint（每次评估若成功率创新高则保存）后，Holdout 直接对应训练过程出现过的最佳泛化能力，R4 Holdout 回升至 78%（+4pp）。

---

## 项目亮点

### 1. 系统性实验方法论：冒烟 → 超参消融 → 算法横评

三阶段渐进式实验设计，每步有明确终止条件，避免盲目调参。每轮变更有问题诊断、数据依据和验收标准，完整记录于 [docs/experiment_log.md](docs/experiment_log.md)。

### 2. 马尔可夫性违反的完整诊断链（P9）

发现 → 理论分析 → 实验验证 → 正确解法的完整闭环，是项目最有技术深度的发现。

### 3. EVAL-based Checkpoint（P7 修复）

```python
if eval_success_rate > best_eval_success:
    best_eval_success = eval_success_rate
    torch.save({"state_dict": policy_net.state_dict(), ...}, best_model_path)
```

三集严格分离（训练 buffer / EVAL 集 / Holdout 集），EVAL 集用于 checkpoint 选择，Holdout 集仅最终报告使用，保证评估数字无偏。

### 4. BFS 连通性保证 + 唯一随机源

`reset()` 内嵌 BFS 验证，确保每张迷宫起点→终点绝对可达，排除无解任务污染训练信号。所有随机操作使用 Gymnasium 注入的唯一 `self.np_random`，评估集可精确复现。

### 5. visited_map 第四通道（Markov-correct 状态编码）

观测张量 `(4, N, N)` 第四通道为二值访问图，将访问历史编码进状态而非奖励，保持马尔可夫性。相比错误方案（revisit_penalty 导致训练崩溃），此方案在 R4 中验证有效。

### 6. TensorBoard 三解耦看板

| 看板前缀 | X 轴 | 指标 |
|---|---|---|
| `Backend_Net/` | `global_update_steps` | Loss、Avg Q Value、Grad Norm |
| `Frontend_Env/` | `episode` | Reward、Steps、Success Rate、Epsilon |
| `Evaluation_Exam/` | `episode` | 盲测成功率、SPL |

三类 X 轴对齐不同事件频率，避免梯度步数与训练局数混用产生视觉误导。

### 7. SPL 评估指标（Anderson et al. 2018 变体）

$$\text{SPL} = \frac{1}{N} \sum_{i=1}^{N} S_i \cdot \frac{\ell_{i}^{\ast}}{\max(\ell_{i}^{\ast},\; p_i)}$$

失败局整项贡献 0，比纯成功率更严格。本项目使用 Grid-SPL 变体（$p_i$ 排除撞墙步，数值系统性偏高），不可与 HabitatAI 等连续导航 Benchmark 直接比较。

### 8. 势函数距离 Shaping（Ng et al. 1999）

每步额外奖励 = `α × (移动前曼哈顿距离 − 移动后曼哈顿距离)`，缓解稀疏奖励问题，早期学习速度显著提升（R3 vs R2 同期约 +6pp）。代码省略 γ（标准形式 F = γΦ(s')−Φ(s)），策略不变性定理（Ng et al., 1999）严格意义上不再成立，但 γ=0.99≈1，实践影响可忽略，属理论上不严格、实践可行的工程简化。

### 9. Episode 级 Warmup + CI（90%+ 覆盖率）

前 `warmup_episodes` 局固定 ε=1.0 纯随机探索，保证回放池多样性再开始学习。GitHub Actions + pytest-cov 保障环境包关键逻辑有测试覆盖。

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

> **注**：`app.py` 推理时叠加了计数惩罚兜底（访问次数 ≥ 2 时对高频格施加递进 Q 值惩罚），仅影响 Demo 视觉体验。所有 Holdout/EVAL 数字均来自 `run_evaluation()`，使用裸 argmax，不受此影响。根本解法是将 ch3 改为归一化计数图重新训练，因时间限制未实施。

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
│   ├── experiment_log.md       逐轮训练记录（配置 + 结果 + 诊断 + 后续优化项）
│   ├── hyperparameter_study.md 超参分析报告（含论文依据）
│   ├── technical_report.md     技术报告（算法原理 + 评估指标 + 结果分析）
│   └── assets/                 训练曲线截图（Round 1/2/3 + 对比图）
│
├── reports/
│   └── comparison.md           R4 四算法最终对比报告（含 SPL 共线性分析）
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

## 参考文献

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning*. **Nature**.
2. van Hasselt, Guez & Silver (2016). *Deep Reinforcement Learning with Double Q-learning*. **AAAI**.
3. Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. **ICML**.
4. Ng et al. (1999). *Policy invariance under reward transformations*. **ICML**.
5. Anderson et al. (2018). *On Evaluation of Embodied Navigation Agents*. arXiv:1807.06757.
6. Henderson et al. (2018). *Deep Reinforcement Learning that Matters*. **AAAI**.
7. Lin (1992). *Self-improving reactive agents based on reinforcement learning, planning and teaching*. **Machine Learning**, 8(3–4).
8. Schaul et al. (2016). *Prioritized Experience Replay*. **ICLR**.
