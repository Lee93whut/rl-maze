"""report.py —— DQN 算法横向对比报告生成器

职责
----
扫描 ``results/`` 目录下所有 ``best_model_train_*.pth``，对每个算法：
1. 加载 checkpoint（自动识别 DQNNetwork / DuelingDQNNetwork）
2. 在 Holdout 集（seed + 200000，共 100 张从未参与训练的地图）上运行评估
3. 汇总成功率、SPL、保存 Episode、训练 AvgReward，输出对比表格

输出
----
* 终端打印对比表格
* 保存 ``reports/comparison.md``

用法
----
python src/report.py                        # 使用默认 config.yaml
python src/report.py --config config.yaml   # 显式指定配置文件
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

# ── 项目内部模块 ────────────────────────────────────────────────────────────
from src.model import DQNNetwork, DuelingDQNNetwork
from src.train import run_evaluation


# ===========================================================================
# 工具：从文件名提取算法标签
# ===========================================================================

def _algo_from_path(pth: Path) -> str:
    """从 best_model_train_<algo>.pth 中提取 <algo>。"""
    stem = pth.stem                          # e.g. "best_model_train_double_dueling"
    prefix = "best_model_train_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem                              # 兜底：原始文件名去扩展名


# ===========================================================================
# 主逻辑
# ===========================================================================

def build_report(config_path: str = "config.yaml") -> None:
    """扫描 results/，评估所有算法，输出对比报告。"""

    # ── 读取配置 ────────────────────────────────────────────────────────────
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"[WARN] 配置文件未找到：{cfg_file}，使用内置默认值。")
        cfg = {}
    else:
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))

    maze_cfg   = cfg.get("maze", {})
    reward_cfg = cfg.get("rewards", {})
    dqn_cfg    = cfg.get("dqn", {})

    grid_size        = int(maze_cfg.get("grid_size", 10))
    obstacle_density = float(maze_cfg.get("obstacle_density", 0.25))
    max_steps        = int(maze_cfg.get("max_steps", 200))
    reward_goal      = float(reward_cfg.get("goal", 100.0))
    reward_wall_hit  = float(reward_cfg.get("wall_hit", -10.0))
    reward_step      = float(reward_cfg.get("step", -1.0))
    seed             = int(dqn_cfg.get("seed", 42))
    save_dir         = str(dqn_cfg.get("save_dir", "results"))

    # Holdout 集：seed+200000，100 张在整个训练过程中从未出现的地图
    holdout_seed_base = seed + 200000
    holdout_seeds     = [holdout_seed_base + i for i in range(100)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 扫描 results/ ───────────────────────────────────────────────────────
    results_dir = Path(save_dir)
    pth_files   = sorted(results_dir.glob("best_model_train_*.pth"))
    if not pth_files:
        print(f"[ERROR] 在 {results_dir.resolve()} 中未找到任何 best_model_train_*.pth。")
        print("  请先运行 python src/train.py 或 ./pipeline.sh 完成训练。")
        sys.exit(1)

    # ── 逐算法评估 ──────────────────────────────────────────────────────────
    rows: list[dict] = []
    for pth in pth_files:
        algo = _algo_from_path(pth)
        print(f"  评估 [{algo}]  {pth.name} …", end=" ", flush=True)

        try:
            ckpt      = torch.load(pth, map_location=device, weights_only=True)
            saved_gs  = ckpt.get("grid_size", grid_size)
            ckpt_algo = ckpt.get("algorithm", algo).strip().lower()
            NetClass  = DuelingDQNNetwork if "dueling" in ckpt_algo else DQNNetwork
            net       = NetClass(grid_size=saved_gs).to(device)
            net.load_state_dict(ckpt["state_dict"])

            success_rate, spl = run_evaluation(
                policy_net=net,
                grid_size=saved_gs,
                obstacle_density=obstacle_density,
                max_steps=max_steps,
                device=device,
                test_seeds=holdout_seeds,
                reward_goal=reward_goal,
                reward_wall_hit=reward_wall_hit,
                reward_step=reward_step,
                random_start_goal=False,   # 横向对比：四算法统一固定起终点评估，消除起终点随机噪声，确保比较公平
            )
            rows.append({
                "algo":        algo,
                "success":     success_rate,
                "spl":         spl,
                "episode":     ckpt.get("episode", -1),
                "avg_reward":  ckpt.get("avg_reward", float("nan")),
            })
            print(f"Success={success_rate:.1f}%  SPL={spl:.3f}")
        except Exception as exc:
            print(f"[SKIP] 加载失败：{exc}")

    if not rows:
        print("[ERROR] 没有成功加载任何模型，报告生成中止。")
        sys.exit(1)

    # ── 排序：Holdout 成功率降序 ────────────────────────────────────────────
    rows.sort(key=lambda r: r["success"], reverse=True)
    best = rows[0]

    # ── 格式化表格 ──────────────────────────────────────────────────────────
    SEP   = "=" * 62
    HDR   = f"{'算法':<18} {'成功率':>6}  {'SPL':>6}  {'保存Episode':>11}  {'训练AvgReward':>13}"
    lines = [
        SEP,
        "  DQN 算法对比报告（Holdout Test，100 张独立地图）",
        SEP,
        HDR,
    ]
    for r in rows:
        lines.append(
            f"{r['algo']:<18} {r['success']:>5.1f}%  {r['spl']:>6.3f}"
            f"  {r['episode']:>11d}  {r['avg_reward']:>13.1f}"
        )
    lines += [
        SEP,
        f"最优算法：{best['algo']}（Holdout 成功率 {best['success']:.1f}%）",
    ]

    # ── 终端输出 ────────────────────────────────────────────────────────────
    print()
    for line in lines:
        print(line)

    # ── Markdown 报告 ───────────────────────────────────────────────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    md_path = reports_dir / "comparison.md"

    md_rows_header = "| 算法 | 成功率 | SPL | 保存Episode | 训练AvgReward |"
    md_rows_sep    = "|------|-------:|----:|------------:|--------------:|"
    md_data_rows   = [
        f"| {r['algo']} | {r['success']:.1f}% | {r['spl']:.3f}"
        f" | {r['episode']} | {r['avg_reward']:.1f} |"
        for r in rows
    ]
    md_content = "\n".join([
        "# DQN 算法对比报告",
        "",
        "> Holdout Test：100 张独立地图（seed+200000），整个训练过程中从未使用。",
        "",
        md_rows_header,
        md_rows_sep,
        *md_data_rows,
        "",
        f"**最优算法：{best['algo']}**（Holdout 成功率 {best['success']:.1f}%）",
        "",
    ])
    md_path.write_text(md_content, encoding="utf-8")
    print(f"报告已保存至：{md_path.resolve()}")
    print(SEP + "\n")


# ===========================================================================
# 入口
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN 算法对比报告生成器")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="YAML 配置文件路径（默认：config.yaml）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # 支持从项目根目录或 src/ 目录调用时都能找到 config.yaml
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        candidates = [cfg_path, Path(__file__).resolve().parent.parent / cfg_path]
        for c in candidates:
            if c.exists():
                cfg_path = c
                break
    build_report(config_path=str(cfg_path))
