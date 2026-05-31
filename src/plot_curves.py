"""
plot_curves.py — 从 TensorBoard event 文件生成对比图
用法：python src/plot_curves.py
输出：docs/assets/round2/ 和 docs/assets/compare/ 下的 PNG 文件
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ── 配置 ──────────────────────────────────────────────────────────────────────
RUNS = {
    "Round1\n(ε_decay=0.995, ep=2000)": "runs/Round1_double_epsilon0.995_ep2000",
    "Round2\n(ε_decay=0.9985, ep=6000)": "runs/Round2_double_epsilon0.9985_ep6000",
}
COLORS = {
    "Round1\n(ε_decay=0.995, ep=2000)":  "#E07B39",   # 橙
    "Round2\n(ε_decay=0.9985, ep=6000)": "#4878CF",   # 蓝
}
OUT_CMP   = "docs/assets/compare"
OUT_R2    = "docs/assets/round2"
os.makedirs(OUT_CMP, exist_ok=True)
os.makedirs(OUT_R2,  exist_ok=True)

# ── 读取 event 文件 ───────────────────────────────────────────────────────────
def load(run_path: str, tag: str):
    ea = event_accumulator.EventAccumulator(
        run_path, size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values

def smooth(values, weight=0.6):
    """指数移动平均平滑"""
    s, out = values[0], []
    for v in values:
        s = weight * s + (1 - weight) * v
        out.append(s)
    return np.array(out)

# ── 通用绘图函数 ──────────────────────────────────────────────────────────────
def plot_compare(tag, ylabel, title, filename,
                 run_keys=None, ylim=None, smooth_w=0.0,
                 hlines=None, outdir=OUT_CMP):
    """绘制多条曲线对比图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    keys = run_keys or list(RUNS.keys())
    for label in keys:
        path = RUNS[label]
        try:
            steps, vals = load(path, tag)
        except Exception as e:
            print(f"  [SKIP] {label}: {e}")
            continue
        if smooth_w > 0:
            vals = smooth(vals, smooth_w)
        ax.plot(steps, vals, label=label, color=COLORS[label],
                linewidth=1.8, alpha=0.9)

    if hlines:
        for y, color, text in hlines:
            ax.axhline(y, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
            ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 6000,
                    y + 0.5, text, color=color, fontsize=9, va="bottom")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(outdir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅  {out_path}")

# ── 1. 对比图：Test_Success_Rate R1 vs R2 ────────────────────────────────────
print("生成对比图...")
plot_compare(
    tag="Evaluation_Exam/Test_Success_Rate",
    ylabel="Blind Test Success Rate",
    title="Evaluation: Test Success Rate — Round 1 vs Round 2",
    filename="cmp_eval_success_rate_r1_vs_r2.png",
    ylim=(0, 100),
    smooth_w=0.0,
    hlines=[(74, "#4878CF", "R2 峰值 74%"), (54, "#E07B39", "R1 峰值 54%")],
)

# ── 2. 对比图：Global_Epsilon R1 vs R2 ───────────────────────────────────────
plot_compare(
    tag="Frontend_Env/Global_Epsilon",
    ylabel="ε (Epsilon)",
    title="Exploration Decay — Round 1 vs Round 2",
    filename="cmp_frontend_epsilon_r1_vs_r2.png",
    ylim=(0, 1.05),
    smooth_w=0.0,
    hlines=[(0.05, "gray", "ε_min=0.05")],
)

# ── 3. R2 单独：Test_Success_Rate ────────────────────────────────────────────
print("生成 Round 2 单独图...")
R2_KEY = "Round2\n(ε_decay=0.9985, ep=6000)"

def plot_single(tag, ylabel, title, filename, ylim=None, smooth_w=0.0,
                hlines=None, vlines=None, outdir=OUT_R2):
    fig, ax = plt.subplots(figsize=(10, 5))
    steps, vals = load(RUNS[R2_KEY], tag)
    color = COLORS[R2_KEY]
    if smooth_w > 0:
        ax.plot(steps, vals, color=color, alpha=0.25, linewidth=1.0)
        ax.plot(steps, smooth(vals, smooth_w), color=color,
                linewidth=2.0, label="Round 2（平滑）")
    else:
        ax.plot(steps, vals, color=color, linewidth=1.8,
                label="Round 2\n(ε_decay=0.9985, ep=6000)")
    if hlines:
        for y, c, text in hlines:
            ax.axhline(y, color=c, linestyle="--", linewidth=1.2, alpha=0.7)
            ax.text(6100, y + 0.5, text, color=c, fontsize=9, va="bottom")
    if vlines:
        for x, c, text in vlines:
            ax.axvline(x, color=c, linestyle=":", linewidth=1.2, alpha=0.8)
            ax.text(x + 30, ax.get_ylim()[0] + 2 if ylim is None else ylim[0] + 2,
                    text, color=c, fontsize=8, rotation=90, va="bottom")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(outdir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✅  {out_path}")

plot_single(
    tag="Evaluation_Exam/Test_Success_Rate",
    ylabel="Blind Test Success Rate (%)",
    title="Round 2: Blind Test Success Rate（全程振荡，无收敛平台）",
    filename="r2_eval_success_rate.png",
    ylim=(0, 100),
    hlines=[(74, "#2ca02c", "峰值 74%（ep=3300, ep=4250）"),
            (52, "#d62728", "低谷 52%（ep=5300, ep=5900）")],
    vlines=[(3300, "#2ca02c", "ep=3300"), (4250, "#2ca02c", "ep=4250"),
            (2189, "#9467bd", "ε触底 ep≈2189")],
)

# ── 4. R2 单独：SPL ───────────────────────────────────────────────────────────
plot_single(
    tag="Evaluation_Exam/SPL",
    ylabel="SPL",
    title="Round 2: SPL（Success-weighted Path Length）",
    filename="r2_eval_spl.png",
    ylim=(0, 1.0),
)

# ── 5. R2 单独：Global_Epsilon ────────────────────────────────────────────────
plot_single(
    tag="Frontend_Env/Global_Epsilon",
    ylabel="ε (Epsilon)",
    title="Round 2: Exploration Decay（ep≈2189 触底）",
    filename="r2_frontend_epsilon.png",
    ylim=(0, 1.05),
    hlines=[(0.05, "gray", "ε_min=0.05")],
    vlines=[(2189, "#9467bd", "ε触底 ep≈2189")],
)

# ── 6. R2 单独：Avg_Reward_Window ─────────────────────────────────────────────
plot_single(
    tag="Frontend_Env/Avg_Reward_Window",
    ylabel="Avg Reward (rolling 100 ep)",
    title="Round 2: Rolling Average Reward",
    filename="r2_frontend_avg_reward.png",
    smooth_w=0.7,
)

# ── 7. R2 单独：Loss ──────────────────────────────────────────────────────────
plot_single(
    tag="Backend_Net/Loss",
    ylabel="TD Loss",
    title="Round 2: Training Loss",
    filename="r2_backend_loss.png",
    smooth_w=0.85,
)

# ── 8. R2 单独：Avg_Q_Value ───────────────────────────────────────────────────
plot_single(
    tag="Backend_Net/Avg_Q_Value",
    ylabel="Avg Q Value",
    title="Round 2: Average Q Value",
    filename="r2_backend_avg_q.png",
    smooth_w=0.7,
)

print("\n全部完成！")
