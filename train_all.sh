#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_all.sh  —  一键并行启动四组 DQN 算法对比实验（tmux 多窗口）
#
# 用法：
#   chmod +x train_all.sh && ./train_all.sh
#
# 依赖：
#   - tmux（brew install tmux / apt install tmux）
#   - 已激活的 Python 虚拟环境（内含 torch / gymnasium / tensorboard 等）
#
# 产物：
#   logs/train_<algo>.log   —— 各算法完整训练日志
#   runs/<algo>_<ts>/       —— TensorBoard 事件文件
#   results/best_model_<algo>.pth  —— 各算法最优权重
#
# 查看进度：
#   tmux attach -t maze_train        # 在各 pane 之间 Ctrl+b → 方向键切换
#   tensorboard --logdir=runs        # http://localhost:6006
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SESSION="maze_train"
ALGOS=("vanilla" "double" "dueling" "double_dueling")
SCRIPT="src/train.py"
CONFIG="config.yaml"
LOG_DIR="logs"

# ── 检查依赖 ──────────────────────────────────────────────────────────────────
command -v tmux  >/dev/null 2>&1 || { echo "[ERROR] tmux 未安装"; exit 1; }
command -v python >/dev/null 2>&1 || { echo "[ERROR] python 未找到"; exit 1; }
[[ -f "$SCRIPT"  ]] || { echo "[ERROR] 找不到 $SCRIPT，请在项目根目录执行本脚本"; exit 1; }
[[ -f "$CONFIG"  ]] || { echo "[ERROR] 找不到 $CONFIG"; exit 1; }

mkdir -p "$LOG_DIR" results runs

# ── 杀死同名旧 session（避免端口/文件冲突）────────────────────────────────────
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── 创建新 session（后台，无附加）─────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -x 220 -y 50

for i in "${!ALGOS[@]}"; do
    ALGO="${ALGOS[$i]}"
    LOG_FILE="${LOG_DIR}/train_${ALGO}.log"

    CMD="python ${SCRIPT} --config ${CONFIG} --algorithm ${ALGO} \
2>&1 | tee ${LOG_FILE}; echo '[DONE] ${ALGO} 训练结束'; exec bash"

    if [[ $i -eq 0 ]]; then
        # 第一个算法复用初始 pane（window 0）
        tmux send-keys -t "${SESSION}:0" "$CMD" Enter
    else
        # 后续算法各开一个新 window
        tmux new-window -t "${SESSION}"
        tmux send-keys -t "${SESSION}:${i}" "$CMD" Enter
    fi

    # 给每个 window 取有意义的名字
    tmux rename-window -t "${SESSION}:${i}" "$ALGO"
done

echo "============================================================"
echo "  ✅  已在 tmux session「${SESSION}」中启动 ${#ALGOS[@]} 组实验"
echo ""
echo "  查看进度：tmux attach -t ${SESSION}"
echo "    窗口切换：Ctrl+b → 数字键（0~$((${#ALGOS[@]}-1))）"
echo "    分离会话：Ctrl+b d"
echo ""
echo "  实时日志：tail -f logs/train_<algo>.log"
echo "  TensorBoard：tensorboard --logdir=runs"
echo "============================================================"
