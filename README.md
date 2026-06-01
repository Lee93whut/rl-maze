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

## 项目背景

10×10 随机迷宫本身足够简单（原始 DQN 在固定起终点可达 90%+），但**随机起终点 + 随机地图**让状态空间扩大约 40 倍，把"走通迷宫"变成"学一种导航策略"——初版 DQN 在此设定下仅 **61% 成功率**。

本项目的目标不是再提升一两个百分点，而是**走完一遍完整的方法论**：先做冒烟验证排除工程 bug，再做 4 轮超参消融（每轮只改一组变量，用问题诊断驱动下一轮变更），最后固定最优超参做四算法横向消融。整个过程把 Holdout 成功率从 61% 推到 84%（+23pp：超参调优 +17pp，算法切换 +6pp）。

途中还发现并修复了一类常见的 RL 工程错误——**奖励依赖 episode 内历史，违反马尔可夫性**（详见"核心发现"）。

---

## 最终结果

> **Holdout 评估**：100 张训练中从未见过的独立地图（seed+200000），ε=0 贪心推理，裸 argmax。
> ⚠️ n=100 单次跑点，95% CI ≈ ±5pp；相邻算法差距 3–6pp 处于 CI 边缘，严格对比需多次独立运行（Henderson et al., 2018）。

### 训练结果呈现顺序

本节按"训练过程的时间顺序"组织：先看超参演进的纵向消融（同一算法、固定数据集、不同超参），再看 R4 算法横向消融（固定最优超参、不同算法）——这样才能把"哪个提升来自超参、哪个来自算法"分清。

#### 第一阶段：超参演进纵向对比（Double DQN，保持算法一致）

每轮变更都不是"再调一组数字试试"，而是**先看曲线、定位问题、再改超参**：

| 轮次 | 核心变更 | Holdout | SPL | 上一轮 → 本轮：问题诊断与决策 |
|------|---------|:-------:|:---:|--------------------------|
| Round 1 | 初版超参（`ep=2000`, `decay=0.995`） | 61.0% | 0.605 | 起点：随机起终点初次尝试 |
| Round 2 | `ep=6000`（×3）, `decay=0.9985` | 64.0% | 0.633 | R1 训练曲线 ep≈800 即触底、ep≈2000 仍未收敛 → 训练量不足 + 探索衰减过快 |
| Round 3 | `buffer=80k`（×4）, `target=1500`（×3） | 74.0% | 0.735 | R2 EVAL 出现 400–500 ep 周期的剧烈振荡 → buffer 太小、target 更新过频（Q 目标漂移） |
| **Round 4** | EVAL checkpoint + BFS 连通性 + visited_map 4通道 | **78.0%** | **0.773** | R3 EVAL 峰值 84% / Holdout 74% 差 10pp → checkpoint 时序错位（保存的是训练奖励最高的而非 EVAL 最佳的）；训练出现"循环"失败模式 → 修复 checkpoint 策略 + BFS 保证训练数据无死锁 + visited_map 状态编码抑制循环 |

四轮演进曲线（同一 Double DQN 算法，仅超参不同）：

![R1→R4 超参演进纵向对比（Double DQN）](docs/assets/compare/cmp_eval_success_rate_r1_to_r4_double.png)

**这一阶段说明**：在算法不变的前提下，靠"消融 + 诊断"已经把成功率从 61% 推到 78%。接下来的 6pp 提升不能继续靠调超参——超参空间已饱和，曲线形状说明 R4 已接近该算法的能力上限。

#### 第二阶段：R4 算法横向消融（固定 R4 最优超参，唯一变量 = 算法）

R4 既然已到超参极限，再换算法就是"把相同的训练数据 + 同样的训练步数给四种网络架构，看谁能在 Holdout 上多拿 6pp"：

| 排名 | 算法 | Holdout | SPL | EVAL→Holdout Gap |
|:---:|------|:-------:|:---:|:----------------:|
| 🥇 | **Dueling DQN** | **84.0%** | **0.817** | −6pp（泛化最稳） |
| 🥈 | Double + Dueling | 81.0% | 0.793 | −9pp |
| 🥉 | Double DQN | 78.0% | 0.773 | −10pp |
| 4️⃣ | Vanilla DQN | 75.0% | 0.726 | −19pp |

四算法 EVAL 训练曲线：

![R4 四算法 EVAL 成功率对比](docs/assets/round4/r4_eval_success_rate_all_algos.png)

**这一阶段说明**：Dueling 比 Vanilla 在 Holdout 上多 9pp、EVAL→Holdout Gap 小 13pp——这是结构性的，不是统计噪声。Dueling 适配本任务的具体机制见"核心发现"。

---

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
    R2 → R3：修复 buffer + target（Q 目标漂移）
    R3 → R4：修复 checkpoint 时序 + 连通性验证 + 状态编码

阶段三：算法横评（Round 4 续）
    固定 R4 最优超参，四算法串行训练
    唯一变量 = 网络架构与 Q 目标计算方式
```

**为什么先调超参再比算法**：先把超参调到最优，比的才是算法本身的能力差异；否则比的是"哪个算法对糟糕超参更鲁棒"。

### 三个评价指标在不同阶段的作用

本项目主要用三个指标，每个指标在不同阶段回答不同的问题：

| 指标 | 表征什么 | 主要用在哪一阶段 | 怎么驱动决策 |
|------|---------|----------------|------------|
| **训练奖励（Frontend_Env/Episode_Reward）** | 当前策略在训练地图上的即时表现 | 阶段二（超参消融）粗筛 | 看是否"在涨"——R1 ep≈800 触底 → 触发"R2 延长训练"的决策 |
| **EVAL 成功率（Evaluation_Exam/Test_Success_Rate）** | 当前策略在固定 EVAL 集上的泛化表现 | 阶段二精筛 + 阶段三对比 + 阶段一/二的 checkpoint 选择 | R2 出现 400–500 ep 振荡 → 触发"R3 调 buffer/target"；R4 峰值 88% 但 Dueling 末段 90% → 触发"阶段三切 Dueling" |
| **Holdout 成功率 + SPL** | 训练全程未见的 100 张独立地图上的最终泛化能力 | 仅阶段三最终一次性报告 | 不参与中途决策（防止"对着 Holdout 调超参"导致的间接过拟合），只用于回答"项目最终能拿多少分" |

**三套指标配合使用的关键约束**：
- **阶段一看训练奖励就够了**——固定起终点下，奖励能涨就说明训练代码无 bug
- **阶段二必须看 EVAL 成功率**——训练奖励在随机地图下噪声太大（与泛化能力相关性仅 0.3–0.5），用它调超参会选出"训练集噪声"模型
- **阶段三对比算法时同时看 EVAL 峰值 + Holdout + Gap**——EVAL 峰值反映"训练过程见过的最佳泛化"，Holdout 反映"训练过程完全没见过的泛化"，两者差距（Gap）反映模型稳定程度
- **Holdout 不允许中途观察**——一旦中途用 Holdout 调参，就等于把 Holdout 间接泄露进训练过程，最终数字就不可信

完整逐轮记录见 [docs/experiment_log.md](docs/experiment_log.md)。

---

## 核心发现

### 发现一：奖励依赖 episode 内历史会破坏马尔可夫性（P9）

R4 首先尝试用 `revisit_penalty`（重复访问格子施加递进奖励惩罚）抑制推理时的循环行为，但训练至 ep=1000 时成功率仅 38%，持续低于基线 15pp，最终终止。

**根因**：Q-learning 的收敛定理（Watkins & Dayan, 1992）要求奖励函数 $r(s,a,s')$ 仅依赖当前转移，贝尔曼最优算子在此条件下是压缩映射。`revisit_penalty` 使奖励依赖 episode 内访问历史（隐变量），相同的 $(s,a,s')$ 在 episode 内的不同时刻返回不同奖励，破坏马尔可夫性，**收敛性保证失效**——这不意味着"必然不收敛"，而是理论不再保证收敛，实际行为取决于具体问题结构。**本任务中真正致命的次生问题是训练/推理分布不一致**——训练时奖励含惩罚项，推理时不含，导致训练期学到的 Q 函数与推理期使用的环境动力学失配，策略崩溃不可修正。

**正确解法**：把历史信息从奖励空间移到状态空间。观测张量第四通道编码二值访问图（ch3=visited_map），Q 函数合法学习"已访问格价值低"的策略，马尔可夫性完整保持。

> 这一诊断具有通用性：**任何依赖 episode 内历史的奖励项都是这类错误的变体**，正确方向永远是状态编码。

### 发现二：Dueling 架构与本任务的结构适配性

随机起终点迷宫中存在大量"多动作等效"状态（死胡同、走廊段），Dueling 的 V(s)/A(s,a) 分解使价值流与优势流解耦——V(s) 通过所有动作路径的反向传播获得梯度信号，样本效率上等价于用同一个状态 batch 下的所有动作共同监督 V(s) 的估计，而每个 A(s,a) 仅在该动作对应路径上获得梯度，因此 V(s) 的有效更新样本量是 A(s,a) 分支的 `num_actions` 倍（4 个动作时为 4×），估计更稳定，泛化更强。

实测证据：EVAL→Holdout Gap 仅 6pp（最小），而 Vanilla DQN 的 Gap 高达 19pp。

### 发现三：checkpoint 保存策略是系统性问题

R3 用训练滚动奖励触发 checkpoint，但训练奖励受随机地图难度影响，与泛化能力相关性弱（约 0.3–0.5）。实测 EVAL 峰值 84%、Holdout 仅 74%，差距 10pp 全部来自时序错位。

改为 EVAL-based checkpoint（每次评估若成功率创新高则保存）后，Holdout 直接对应训练过程出现过的最佳泛化能力，R4 Holdout 回升至 78%（+4pp）。

---

## 项目亮点

### 1. 训练-评估-报告三集严格分离 + EVAL-based Checkpoint

```
训练 buffer（Replay）── 梯度更新用，分布漂移无所谓
       │
       ▼
   EVAL 集（固定 100 张地图，固定 seed）── 训练期每 N 局评估一次，触发 checkpoint
       │
       ▼
 Holdout 集（独立 100 张地图，seed+200000）── 训练全程不接触，仅最终一次性报告
```

Checkpoint 触发：EVAL 集成功率创新高时保存，确保 checkpoint 永远对应"训练过程中 EVAL 表现最好的一刻"——而不是训练奖励偶然高的一刻。

```python
if eval_success_rate > best_eval_success:
    best_eval_success = eval_success_rate
    torch.save({"state_dict": policy_net.state_dict(), ...}, best_model_path)
```

**直接效果**：避免 R3 的 EVAL 峰值 84% / Holdout 仅 74% 的 10pp 错位，R4 Holdout 回升至 78%（+4pp）。马尔可夫性修复（visited_map 第四通道）和 Dueling 适配详见"核心发现"。

### 2. 训练信号侧的"三项叠加"（ep/buffer/target 调优 + BFS 连通性）

- `ep=6000` + `decay=0.9985`：训练量翻 3 倍、ε 衰减 50× 慢，给策略充分时间收敛
- `buffer=80k`（×4）、`target_update=1500`（×3）：buffer 太小会让成功样本被冲掉、target 更新太频会让 Q 目标变成"移动靶"，两者都是 R2 振荡的直接原因
- BFS 连通性保证：`reset()` 内嵌 BFS，无解迷宫直接重采样——不修这个，训练信号会被"无法完成任务"污染，Q 值学出"迷宫无解"的错误估计

**效果**：仅这三项把成功率从 61% 推到 74%（+13pp）。

### 3. Episode 级 Warmup + BFS Ground Truth

- Warmup：前 N 局 ε=1.0 纯随机，不做任何梯度更新——防止早期 Q 值被少量样本"冲偏"
- BFS Ground Truth：每次评估计算当前迷宫的最短路径步数作为 SPL 的 $\ell^*$——不依赖 Q 值估计，避免"用 Q 值评估 Q 值"的循环

**作用**：提效（减少无效探索 + 评估指标无偏）。

### 4. TensorBoard 三解耦看板

| 看板 | X 轴 | 记录时机 | 避免的问题 |
|---|---|---|---|
| `Backend_Net/` | `global_update_steps` | 每次 `backward()` 后 | — |
| `Frontend_Env/` | `episode` | 每局结束后 | 避免"每局更新步数不同"导致 Loss 曲线横向压缩 |
| `Evaluation_Exam/` | `episode` | 每 N 局暂停训练、`model.eval()` | 训练/评估指标分轴对比，避免视觉误导 |

**作用**：纯提效——诊断问题时不用猜"这条陡降到底是 eval 触发了还是 loss 飙升了"。

### 5. 唯一随机源 + 评估集可复现

`maze_env/env.py` 所有随机操作统一走 Gymnasium 注入的 `self.np_random`。

- 训练时 `env.reset()` 不传 seed → 训练地图每局不同，强制学泛化
- 评估时 `env.reset(seed=X)` 固定 → 同一 seed 永远生成同一张地图，Holdout 100 张地图可精确复现，论文级可对比

**作用**：纯提效 + 评估可信度。如果评估地图每次都变，"成功率 78%"这个数字就没有任何可重复性可言。

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
