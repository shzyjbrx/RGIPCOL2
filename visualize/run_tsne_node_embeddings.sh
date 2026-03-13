#!/bin/bash
#SBATCH --job-name=tsne_vis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/visualize/tsne_%j.out
#SBATCH --error=logs/visualize/tsne_%j.err

# ─────────────────────────────────────────────
# 节点嵌入 t-SNE / UMAP 可视化
# ─────────────────────────────────────────────

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

# 安装可视化依赖（首次运行时取消注释）
# pip install scikit-learn matplotlib seaborn umap-learn --break-system-packages

mkdir -p logs/visualize

# ─── 配置 ───
DATASET="mit-states"
CONFIG="configs/mit_states_config.yaml"
CHECKPOINT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/RGIPCOL2/saved_models/mit-states/bs256_lr1e-4_20260313_1156850/best_model.pt"
SAVE_DIR="./visualize/outputs/t_SNE"
METHOD="both"        # tsne / umap / both
PERPLEXITY=30
MAX_NODES=400        # 每类最多可视化节点数
DPI=200

echo "======================================================"
echo "  节点嵌入可视化"
echo "  数据集     : ${DATASET}"
echo "  降维方法   : ${METHOD}"
echo "  模型权重   : ${CHECKPOINT}"
echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "======================================================"

python -u visualize/tsne_node_embeddings.py \
    --config      ${CONFIG}     \
    --checkpoint  ${CHECKPOINT} \
    --method      ${METHOD}     \
    --perplexity  ${PERPLEXITY} \
    --max_nodes   ${MAX_NODES}  \
    --save_dir    ${SAVE_DIR}   \
    --dpi         ${DPI}        \
    --seed        42

echo "======================================================"
echo "  可视化完成，结果保存在: ${SAVE_DIR}"
echo "======"