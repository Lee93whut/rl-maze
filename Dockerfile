# Dockerfile — Hugging Face Spaces (Docker SDK) 专用
#
# 规范遵守
# --------
# 1. 暴露端口 7860（HF 固定前端端口，禁止使用 8501）
# 2. 非 root 用户 `user`（UID=1000），符合 HF 安全策略
# 3. 启动命令：streamlit run app.py --server.port=7860 --server.address=0.0.0.0
#
# 镜像体积优化
# ------------
# torch CPU-only wheel (~700MB) 代替完整 GPU 版本 (~3GB)
# Demo 仅做推理，不需要 CUDA 支持
#
# 本地构建测试
# ------------
# docker build -t maze-dqn-demo .
# docker run -p 7860:7860 maze-dqn-demo
# 浏览器访问：http://localhost:7860

FROM python:3.10.14-slim

# ── 系统基础依赖 ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── 创建非 root 用户（HF 安全要求：UID=1000）──────────────────────────────
RUN useradd -m -u 1000 user
WORKDIR /app

# ── 先安装依赖（利用 Docker layer cache，代码变更不触发重装）──────────────
COPY --chown=user:user requirements.txt .
# 先单独安装 torch CPU-only wheel，避免默认拉取 GPU 版本（~3GB → ~700MB）
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# ── 拷贝应用核心文件 ──────────────────────────────────────────────────────────
#   app.py                          Web 前端主程序
#   maze_env/                       Gymnasium 环境包（含 bfs.py，通过 pip install 安装）
#   src/model.py                    神经网络架构（sys.path 注入 src/ 后导入）
#   config.yaml                     超参数配置文件（app.py 启动时读取）
#   results/best_model_train_*.pth  四算法训练权重（app.py 按 algorithm 名动态加载）
#   pyproject.toml                  包安装描述符（pip install -e . 需要）
COPY --chown=user:user app.py                          .
COPY --chown=user:user src/model.py                    src/model.py
COPY --chown=user:user maze_env/                       maze_env/
COPY --chown=user:user pyproject.toml                  .
COPY --chown=user:user config.yaml                     .
COPY --chown=user:user results/best_model_train_vanilla.pth        results/best_model_train_vanilla.pth
COPY --chown=user:user results/best_model_train_double.pth         results/best_model_train_double.pth
COPY --chown=user:user results/best_model_train_dueling.pth        results/best_model_train_dueling.pth
COPY --chown=user:user results/best_model_train_double_dueling.pth results/best_model_train_double_dueling.pth

# ── 安装 maze_env 包（使 `from maze_env import ...` 生效）──────────────────
RUN pip install --no-cache-dir -e .

# ── 切换到非 root 用户 ────────────────────────────────────────────────────
USER user

# ── 暴露 HF 专用端口 ──────────────────────────────────────────────────────
EXPOSE 7860

# ── 启动命令（HF 规范：port=7860, address=0.0.0.0）──────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
