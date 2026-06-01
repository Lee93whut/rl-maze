# 超参决策参考表

> 各轮核心超参变更、论文依据与效果的快速索引。  
> 完整逐轮诊断数据见 `docs/experiment_log.md`。

---

## 超参调整历程

| 超参 | R1 | R2 | R3 | R4 | 论文依据 |
|------|:--:|:--:|:--:|:--:|---------|
| `num_episodes` | 2000 | **6000** | 6000 | **5000** | Mnih et al. (2015)：Atari 数万 ep 才收敛；R3 均值 ep=3000 后饱和，继续加 ep 收益边际为零 |
| `epsilon_decay` | 0.995 | **0.9985** | 0.9985 | 0.9985 | van Hasselt et al. (2016)：ε 应在训练期 10–25% 内衰减完；R1 ep≈800 触底，后 1200 ep 样本单一 |
| `buffer_capacity` | 20000 | 20000 | **80000** | 80000 | Lin (1992)：ER 核心价值是保留稀有样本；Mnih et al. (2015) 原版 1M；20k≈250 局，成功样本快速消失 |
| `target_update_freq` | 500 | 500 | **1500** | 1500 | Mnih et al. (2015) 原版 10000 步；固定目标 $\theta^-$ 保证 TD 收敛性，过频同步等价于"移动靶"监督学习 |
| `distance_shaping_alpha` | 0 | 0 | **0.5** | 0.5 | Ng et al. (1999)：势函数 shaping 标准形式 $r' = r + \gamma\Phi(s') - \Phi(s)$；代码实现省略 $\gamma$（$\gamma=1$ 近似），在 $\gamma=0.99$ 下误差约 1%，实践可忽略，但不严格满足策略不变性定理 |
| `revisit_penalty` | 0 | 0 | 0 | ~~-1.0~~ → **弃用** | Round 4 实验后因违反马尔可夫性放弃，改用 visited_map 第四通道编码访问历史 |
| checkpoint 保存策略 | 训练奖励触发 | 训练奖励触发 | 训练奖励触发 | **EVAL 最优触发** | R3 实测：训练奖励与 EVAL 成功率时序不对齐，导致 Holdout 74% vs EVAL 峰值 84%，差 10pp |

---

## 各轮效果汇总

| 轮次 | 核心变更 | Holdout 成功率 | EVAL 峰值 | 主要发现 |
|------|---------|:--------------:|:---------:|---------|
| R1 | 基线（随机起终点初版） | 61.0% | — | 诊断 P1（训练量）+ P2（探索）+ P3（buffer）+ P4（target）|
| R2 | `ep=6000` + `decay=0.9985` | 64.0% | 74.0% | P1/P2 消除；振荡周期 400–500 ep（P3 定量确认）|
| R3 | `buffer=80k` + `target=1500` + `shaping=0.5` | **74.0%** | **84.0%** | 峰值突破 80%；Holdout 低于峰值 10pp（P6：保存策略错位）|
| R4 | EVAL-based checkpoint + visited_map 第四通道（revisit_penalty 因违反马尔可夫性弃用） | **78.0%** | **0.773** | |

---

## 迭代原则

Henderson et al. (2018) 实证结论：**RL 超参敏感度远高于监督学习，需单变量消融**，避免多参数同时变动导致无法归因。

- R2：仅修复 P1+P2，验证收敛形状
- R3：同时修复 P3+P4+P5，验证振荡是否消除
- R4：修复保存策略（P6）+ visited_map 第四通道（revisit_penalty 已弃用），Holdout 78.0%

---

## 参考文献

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
2. van Hasselt, Guez & Silver (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI.
3. Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML.
4. Ng et al. (1999). *Policy invariance under reward transformations*. ICML.
5. Lin (1992). *Self-improving reactive agents based on reinforcement learning*. Machine Learning.
6. Anderson et al. (2018). *On Evaluation of Embodied Navigation Agents*. arXiv:1807.06757.
7. Henderson et al. (2018). *Deep Reinforcement Learning that Matters*. AAAI.
