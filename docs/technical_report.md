# 技术报告：DQN 迷宫寻路

> 作者：lil58  
> 日期：2026-05-31

---

## 一、问题定义

**任务**：在随机生成的 10×10 迷宫中，训练智能体从随机起点导航至随机终点。

| 挑战因素 | 说明 |
|---------|------|
| **Sparse Reward** | 仅到达终点时奖励 +100，其余步骤 −1、撞墙 −10，随机探索触碰终点概率极低 |
| **随机起终点** | 状态空间从 $O(N^2)$ 扩展至 $O(N^2 \times \binom{K}{2})$（K≈40 个可通行格），模型须学习泛化导航策略而非记忆固定路径 |
| **随机地图** | 每局生成新地图，进一步要求泛化而非过拟合特定迷宫结构 |

**为什么选 DQN 系列而不是 PPO/A3C**：
- 离散动作空间（4 个动作）是 DQN 最适用场景
- Off-policy + Experience Replay 对 sparse reward 的样本效率高于 on-policy 方法
- 4 种变体形成清晰的 ablation study，便于分析各项改进的独立贡献

---

## 二、四种算法实现

### 2.1 算法关系

```
Vanilla DQN (Mnih et al., 2015)
    │
    ├── + Double Q-Learning (van Hasselt et al., 2016) → Double DQN
    │       解耦 action selection 与 value estimation，减少 max 算子的过估计偏差
    │
    ├── + Dueling Architecture (Wang et al., 2016) → Dueling DQN
    │       将 Q(s,a) 分解为 V(s) + A(s,a)，对动作无关的状态价值估计更准确
    │
    └── + Double + Dueling → Double Dueling DQN（两项改进正交叠加）
```

### 2.2 关键代码区别

**Vanilla vs Double DQN**（TD 目标计算）：

```python
# Vanilla DQN：同一个 target_net 同时选 action 和估值（overestimation bias）
next_q = target_net(next_states).max(dim=1).values

# Double DQN：policy_net 选 action，target_net 估值（去偏）
next_actions = policy_net(next_states).argmax(dim=1)
next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
```

**DQNNetwork vs DuelingDQNNetwork**（网络结构）：

```python
# DQNNetwork：直接输出 Q(s,a)
Q(s,a) = FC( Conv(s) )

# DuelingDQNNetwork：分支输出 V(s) 和 A(s,a)，再合并
V(s)   = value_stream( Conv(s) )          # 标量
A(s,a) = advantage_stream( Conv(s) )      # |A| 维向量
Q(s,a) = V(s) + A(s,a) - mean(A(s,·))    # 减均值防止 identifiability 问题
```

---

## 三、训练流程设计

### 3.1 TensorBoard 三解耦看板

```
Backend_Net/     X轴：梯度步数   每次 backward() 后记录
    └── Loss / Avg_Q_Value / Grad_Norm

Frontend_Env/    X轴：episode    每局结束后记录
    └── Episode_Reward / Episode_Steps / Rollout_Success_Rate / Global_Epsilon

Evaluation_Exam/ X轴：episode    每 N 局暂停训练，model.eval()，ε=0
    └── Test_Success_Rate / SPL
```

三类 X 轴对齐不同事件频率。若将 Loss 和 Success_Rate 放在同一 X 轴（episode），Loss 曲线会因每局更新步数不同而产生横向压缩偏差，导致视觉误导。

### 3.2 Episode 级 Warmup

前 N 局固定 ε=1.0 纯随机探索，不做任何梯度更新，先把回放池填充至足够多样。

Step 级 warmup 可能在一局中途切换为学习模式，导致同一局内前半段随机、后半段贪心，破坏 episode 的完整性，污染早期 TD 目标估计。

### 3.3 BFS Ground Truth

每次评估用 BFS 计算当前迷宫的最优路径步数，作为 SPL 的 $\ell^{*}$。`reset()` 内嵌 BFS 连通性验证——不可达则重新采样，确保训练信号不被"无解任务"污染。

---

## 四、评估指标

### 4.1 指标选择逻辑

本项目使用两个互补指标，各自回答不同的问题：

| 指标 | 回答的问题 | 指导下一步方向 |
|------|----------|--------------|
| **成功率** | 策略是否可用？ | 若低 → 改算法、调超参、修训练信号 |
| **Grid-SPL** | 成功时路径是否高效？ | 若 SPL/成功率比值低 → 优化路径规划（延长训练等） |

**本项目实测结论**：四算法 Holdout SPL/成功率比值为 0.968–0.991（均值 0.978），全程 EVAL 逐点比值稳定在 0.986 ± 0.014。这说明**成功时的路径效率已不是瓶颈**——agent 要么高效成功，要么完全失败，路径规划本身已接近最优。后续提升空间集中在成功率本身，而非路径效率。

### 4.2 成功率

最直接的任务完成度指标。Holdout 评估在 100 张固定地图（seed+200000）上以 ε=0 贪心推理，裸 argmax，无任何推理时修正，保证数字反映纯网络能力。

### 4.3 Grid-SPL（Anderson et al. 2018 变体）

$$\text{SPL} = \frac{1}{N} \sum_{i=1}^{N} S_i \cdot \frac{\ell^{*}_{i}}{\max(\ell^{*}_{i},\, p_{i})}$$

- $S_i$：第 i 局成功标志（0/1）
- $\ell^{*}_i$：BFS 最短路径步数
- $p_i$：Agent **实际移动步数**（排除撞墙步，见下方说明）

失败局整项贡献 0，同时惩罚绕路行为，比纯成功率更严格。

**为何排除撞墙步（Grid-SPL 与标准 SPL 的差异）**：

标准 SPL（HabitatAI）的 $p_i$ 是 agent 总动作数。在连续导航场景中这没有问题，因为"碰墙"不是一个显式的离散动作。但在网格迷宫中，撞墙是一个明确的失败动作——若计入 $p_i$，SPL 变成"路径质量 × 撞墙规避能力"的混合指标，两种不同能力的 agent 无法被区分。排除撞墙步后，$p_i$ 只反映真实移动路径的长度，SPL 纯粹度量**路径规划质量**，两种能力解耦。

> ⚠️ 此变体使 $p_i$ 偏小、SPL 数值系统性偏高，**不可与 HabitatAI、EmbodiedQA 等连续导航 Benchmark 的 SPL 直接比较**。

**Holdout 防泄漏**：训练地图每局随机生成（seed 随机）；评估地图固定 100 张（seed+200000），seed 空间完全隔离，确保曲线波动反映 Q 函数能力而非地图难度变化。

### 4.4 各阶段用哪个指标

三套指标配合，按阶段分工：

| 阶段 | 主要指标 | 次要指标 | 决策依据 |
|------|---------|---------|---------|
| 阶段一（冒烟） | 训练奖励 | — | 固定起终点下奖励能涨 → 训练代码无 bug |
| 阶段二（超参消融 R1–R4） | EVAL 成功率 | 训练奖励 | 训练奖励噪声大（与泛化能力相关性 0.3–0.5），仅用于粗筛；EVAL 用于精筛超参与选 checkpoint |
| 阶段三（算法横评） | EVAL 峰值 + Holdout + Gap | — | EVAL 峰值反映"训练中见过的最佳泛化"，Holdout 反映"训练完全没见过的泛化"，Gap 反映稳定性 |
| **全程约束** | — | — | **Holdout 不允许中途观察**，仅最终一次性报告，防止间接过拟合 |

---

## 五、工程设计亮点

| 亮点 | 实现 | 意义 |
|------|------|------|
| 唯一随机源 | 所有随机操作使用 Gymnasium 注入的 `self.np_random` | `env.reset(seed=X)` 固定评估集地图分布，不影响训练随机流 |
| BFS 连通性保证 | `reset()` 内嵌 BFS，不可达则重新采样 | 排除无解迷宫污染训练信号 |
| 三解耦 TensorBoard | Backend/Frontend/Evaluation 三类 X 轴独立 | 避免梯度步数与幕数混用产生视觉误导 |
| Episode 级 Warmup | 前 N 局不做任何梯度更新 | 保证回放池多样性，避免早期 Q 值估计崩溃 |
| Holdout 防泄漏 | 评估 seed 与训练 seed 空间隔离 | 确保评估曲线反映真实泛化能力 |
| CI + 90% 覆盖率 | GitHub Actions + pytest-cov | 环境包关键逻辑有测试保障 |

---

## 六、实验结果

### 6.1 四轮超参演进（Double DQN）

| 轮次 | 核心变更 | Holdout 成功率 | SPL | 关键问题诊断 |
|------|---------|:-------------:|:---:|------------|
| R1 | 随机起终点初版 | 61.0% | 0.605 | 训练量不足、探索过早终止 |
| R2 | `ep↑` + `decay↓` | 64.0% | 0.633 | buffer 过小导致振荡 |
| R3 | `buffer×4` + `target×3` | 74.0% | 0.735 | checkpoint 时序偏差 10pp |
| R4 | EVAL checkpoint + BFS + visited_map | **78.0%** | **0.773** | — |

![R1→R4 超参演进 EVAL 成功率对比（Double DQN）](assets/compare/cmp_eval_success_rate_r1_to_r4_double.png)

### 6.2 R4 四算法横向消融

固定 R4 最优超参，唯一变量为算法：

| 排名 | 算法 | Holdout 成功率 | SPL | EVAL→Holdout Gap |
|:---:|------|:-------------:|:---:|:----------------:|
| 🥇 | Dueling DQN | **84.0%** | **0.817** | −6pp（最小） |
| 🥈 | Double + Dueling | 81.0% | 0.793 | −9pp |
| 🥉 | Double DQN | 78.0% | 0.773 | −10pp |
| 4️⃣ | Vanilla DQN | 75.0% | 0.726 | −19pp |

![R4 四算法 EVAL 成功率对比](assets/round4/r4_eval_success_rate_all_algos.png)

**核心结论**：Dueling 架构的 V(s)/A(s,a) 分解与本任务高度适配（大量多动作等效状态），泛化最稳定（Gap 最小）。Double DQN 的主要收益在训练早中期（加速危机恢复），充分训练后与 Vanilla 差距收敛。完整分析见 [experiment_log.md](experiment_log.md)。

**统计显著性说明**：n=100 单次跑点，二项分布标准差 ≈ √(p·(1−p)/100)。Dueling（84%）vs Double+Dueling（81%）差距 3pp，而 σ ≈ 3.6pp，差异不具统计显著性（p > 0.2）；Dueling（84%）vs Double DQN（78%）差距 6pp ≈ 1.7σ，边缘显著。排名仅反映本次单次实验的观测点，严格结论需多次独立运行取均值±标准差。

**Double+Dueling 低于单独 Dueling 的反直觉结论**：两项改进理论上正交，但实测 Double+Dueling（81%）低于单独 Dueling（84%）。可能原因：Double DQN 通过 policy_net 选动作、target_net 估值来降低 Q 值高估，而 Dueling 的 V/A 分解本身已通过"减去均值"使优势估计更保守，两者同时作用可能导致 Q 值被过度低估（underestimation bias），在稀疏奖励迷宫中使策略趋于保守，抑制了 Dueling 的泛化优势。该假设未经消融验证，3pp 差距处于统计 CI 内，亦不排除随机因素。

### 6.3 失败模式分析

EVAL 期逐局步数（R3 / R4 best_model，Holdout 100 局，ε=0 贪心）：

| 分类 | R3 | R4 |
|------|---:|---:|
| 快速成功（≤30 步） | 75 | 78 |
| 失败·截断（步数=200） | 25 | 22 |
| 失败·近截断（150-199）/ 早夭（<150） | 0 / 0 | 0 / 0 |
| 成功率 | 75% | 78% |
| **失败局中步数=200 截断占比** | **25/25 = 100%** | **22/22 = 100%** |
| 失败局平均撞墙数 | 0.0 | 0.0 |

**R3 / R4 失败局 100% 是步数=200 截断且撞墙=0**——10×10 迷宫最优路径仅 15-25 步，200 步远超合理上限；撞墙=0 排除"撞墙堵死"；**唯一合理解释是自由格间反复震荡**（Demo 中 A→B→A→B 模式肉眼可见）。0% 近截断、0% 早夭、成功-失败 0/1 离散分布（成功≤30 步，失败=200 步），无"路径规划差但仍在前进"或"撞墙堵死"的中间情形。

**R3 → R4 变化**：25 → 22 截断局（绝对 −3pp；n=100 下 σ ≈ 4pp，**不具统计显著性**），但**截断率 100% 这一结构特征未变**——4 通道减少循环地图数量，循环机制仍是 R4 EVAL 期失败主因（详见 §7 Web Demo 兜底机制说明）。

**训练期 vs EVAL 期**（验证 ε 探索对循环的边际贡献）：

| 阶段 | 截断占比 |
|------|---------:|
| R3 训练期 580 局（ep 200+，含 5% ε 探索） | 15.5% |
| R3 EVAL 期 100 局（ε=0 贪心） | 25% |
| R4 训练期 480 局（ep 200+，三算法均值，含 5% ε） | 8-12% |
| R4 EVAL 期 100 局（ε=0 贪心） | 22% |

训练期截断率系统性低于 EVAL 期，与"推理时纯贪心放大循环失败"判断一致。R3 训练期 15.5% 是 R4 各算法（8-12%）的 1.3-1.9 倍——4 通道在训练期也已部分抑制循环（visited_map 使网络对"重访格子"敏感）。

R4 引入 4 通道的完整因果链（训练期 15.5% → EVAL 期 25% → 100% 截断且撞墙=0 → R4 引入 visited_map）：见 [experiment_log.md §R3 总结与 R4 决策依据](experiment_log.md#r3-总结与-r4-决策依据) 第四节。

---

## 七、Web Demo 兜底机制说明

### 7.1 评估侧 vs Demo 侧

| 路径 | anti-loop 修正 | 说明 |
|------|:-------------:|------|
| Holdout / EVAL（`run_evaluation`） | **无**，裸 argmax | 所有报告数字均为纯网络能力 |
| Web Demo（`app.py`） | **有**，计数惩罚 | 仅影响 Demo 视觉体验 |

### 7.2 兜底的必要性

visited_map 第四通道将访问历史编码为二值图（去过/没去过），解决了马尔可夫性问题。但二值信息使网络无法感知"访问了几次"，对两格死循环这一边缘状态的 Q 值估计不稳定（训练中此类样本覆盖密度极低）。失败模式分析（6.3节）证实循环是主要失败原因，因此在 Demo 中加入计数惩罚作为安全网是有针对性的工程决策。

### 7.3 根本解法（后续优化项）

将 ch3 从二值图改为**归一化计数图**（cap=3）：

```python
obs[3, r, c] = min(visit_count[r, c], 3) / 3.0
# 0次→0.0, 1次→0.33, 2次→0.67, ≥3次→1.0
```

网络训练后直接内化"高频重访格应规避"的策略，届时推理时的 Q 值修正可完全移除。因时间限制未重新训练，当前 Demo 保留推理时兜底作为临时替代。

---

## 八、参考文献

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning*. **Nature**, 518, 529–533.
2. van Hasselt, Guez & Silver (2016). *Deep Reinforcement Learning with Double Q-learning*. **AAAI**.
3. Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. **ICML**.
4. Ng et al. (1999). *Policy invariance under reward transformations*. **ICML**.
5. Anderson et al. (2018). *On Evaluation of Embodied Navigation Agents*. arXiv:1807.06757.
6. Henderson et al. (2018). *Deep Reinforcement Learning that Matters*. **AAAI**.
7. Lin (1992). *Self-improving reactive agents based on reinforcement learning, planning and teaching*. **Machine Learning**, 8(3–4).
