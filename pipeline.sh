#!/usr/bin/env bash
# pipeline.sh —— DQN 迷宫训练全自动流水线
#
# 流程
# ----
#   Step 1  overfit 冒烟测试（double_dueling，5×5 迷宫，验证算法可收敛）
#           通过 → Step 2；不通过 → exit 1
#
#   Step 2  四算法串行训练（vanilla → double → dueling → double_dueling）
#           每个算法训练结束后自动执行 Holdout Test（train.py 内置）
#           日志写入 logs/train_<algo>.log
#
#   Step 3  运行 src/report.py 汇总四算法 Holdout 对比报告
#           输出 reports/comparison.md
#
# 用法
# ----
#   bash pipeline.sh              # 从项目根目录执行
#   chmod +x pipeline.sh && ./pipeline.sh
#
# 依赖
# ----
#   python src/train.py --help 可用（maze_env 已安装，依赖已满足）
#   无需 tmux；串行执行，日志清晰，适合 CI / 无人值守环境
# ---------------------------------------------------------------------------

set -euo pipefail   # 任何命令失败立即终止；未定义变量报错；管道错误传播

# ── 目录准备 ────────────────────────────────────────────────────────────────
mkdir -p logs results reports

# ── 颜色输出（如果终端支持）────────────────────────────────────────────────
_bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
_green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
_red()   { printf '\033[0;31m%s\033[0m\n' "$*"; }
_info()  { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

_bold "================================================================"
_bold "  DQN 迷宫训练流水线  (pipeline.sh)"
_bold "================================================================"

# ===========================================================================
# Step 1：overfit 冒烟测试
# ===========================================================================
_info "[1/3] 运行 overfit 冒烟测试 (double_dueling, 5×5 迷宫)..."

python -u src/train.py --overfit --algorithm double_dueling \
    2>&1 | tee logs/overfit.log

if grep -q "✅  过拟合测试通过" logs/overfit.log; then
    _green "[PASS] overfit 验收通过，进入正式训练阶段。"
else
    _red   "[FAIL] overfit 验收未通过，终止流程。"
    _red   "       请检查 logs/overfit.log 排查原因。"
    exit 1
fi

echo ""

# ===========================================================================
# Step 2：四算法串行训练
# ===========================================================================
_info "[2/3] 开始四算法串行训练..."

for ALGO in vanilla double dueling double_dueling; do
    _info "  训练算法：$ALGO"
    LOG_FILE="logs/train_${ALGO}.log"

    python -u src/train.py --algorithm "$ALGO" \
        2>&1 | tee "$LOG_FILE"

    # 检查训练是否正常完成（train.py 成功退出即可，Holdout 已内置打印）
    if grep -q "训练完成" "$LOG_FILE"; then
        _green "  [DONE] $ALGO 训练完成，日志：$LOG_FILE"
    else
        _red   "  [WARN] $ALGO 训练日志中未找到'训练完成'标志，请检查 $LOG_FILE"
    fi
    echo ""
done

# ===========================================================================
# Step 3：生成对比报告
# ===========================================================================
_info "[3/3] 生成四算法对比报告..."

python src/report.py

_bold "================================================================"
_green "  流水线全部完成！"
_bold "  对比报告：reports/comparison.md"
_bold "  TensorBoard：tensorboard --logdir=runs"
_bold "================================================================"
