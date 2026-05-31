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

### SPL（Anderson et al. 2018）

$$\text{SPL} = \frac{1}{N} \sum_{i=1}^{N} S_i \cdot \frac{\ell^{*}_i}{\max(\ell^{*}_i,\ p_i)}$$

- $S_i$：第 i 局成功标志（0/1）
- $\ell^{*}_i$：BFS 最短路径步数
- $p_i$：Agent 实际移动步数（排除撞墙步）

失败局整项贡献 0，避免"成功但绕远路"的高分；与 HabitatAI、EmbodiedQA 等主流导航 Benchmark 评估体系一致。

**Holdout 防泄漏**：训练地图每局随机生成（seed 随机）；评估地图固定 100 张（seed+200000），seed 空间完全隔离，确保曲线波动反映 Q 函数能力而非地图难度变化。

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

## 六、参考文献

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning*. **Nature**, 518, 529–533.
2. van Hasselt, Guez & Silver (2016). *Deep Reinforcement Learning with Double Q-learning*. **AAAI**.
3. Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. **ICML**.
4. Ng et al. (1999). *Policy invariance under reward transformations*. **ICML**.
5. Anderson et al. (2018). *On Evaluation of Embodied Navigation Agents*. arXiv:1807.06757.
6. Henderson et al. (2018). *Deep Reinforcement Learning that Matters*. **AAAI**.
7. Lin (1992). *Self-improving reactive agents based on reinforcement learning, planning and teaching*. **Machine Learning**, 8(3–4).
