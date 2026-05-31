"""scripts/download_weights.py — 从 Hugging Face Spaces 下载推理权重

用法
----
    python scripts/download_weights.py

说明
----
训练权重不放在 GitHub（二进制文件不适合 git），统一存放在 HF Spaces。
本脚本将四个算法的最优权重下载到 results/ 目录，下载后可直接运行：

    streamlit run app.py       # 本地 Demo
    python src/train.py ...    # 继续训练（可选）

依赖
----
    pip install huggingface_hub   # requirements.txt 已包含
"""

from __future__ import annotations

from pathlib import Path
from huggingface_hub import hf_hub_download

# HF Spaces repo（存放权重的 Space）
REPO_ID   = "lil58/interview"
REPO_TYPE = "space"

# 需要下载的权重文件（HF 上的路径 → 本地保存路径）
WEIGHTS = {
    "results/best_model_train_vanilla.pth":        "results/best_model_train_vanilla.pth",
    "results/best_model_train_double.pth":         "results/best_model_train_double.pth",
    "results/best_model_train_dueling.pth":        "results/best_model_train_dueling.pth",
    "results/best_model_train_double_dueling.pth": "results/best_model_train_double_dueling.pth",
}

def main() -> None:
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"Downloading weights from HF Space: {REPO_ID}\n")

    for remote_path, local_rel in WEIGHTS.items():
        local_path = Path(__file__).parent.parent / local_rel
        if local_path.exists():
            print(f"  [skip]  {local_rel}  (already exists)")
            continue

        print(f"  [down]  {local_rel} ...", end=" ", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=remote_path,
            local_dir=str(Path(__file__).parent.parent),
        )
        print("done")

    print("\nAll weights ready. Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
