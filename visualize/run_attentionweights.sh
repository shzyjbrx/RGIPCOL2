#!/bin/bash
#SBATCH --job-name=attn_vis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/visualize/attn_%j.out
#SBATCH --error=logs/visualize/attn_%j.err

# ─────────────────────────────────────────────
# 软提示注意力权重可视化
# ─────────────────────────────────────────────

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38


# ─── 配置 ───
CONFIG="configs/mit_states_config.yaml"
CHECKPOINT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/RGIPCOL2/saved_models/mit-states/bs256_lr1e-4_20260313_1156850/best_model.pt"
SAVE_DIR="./visualize/outputs/attention"
TOP_LAYERS=4
DPI=200

# ─── 选取有代表性的组合 ───
# 建议选：1个seen seen的典型组合 + 2-3个unseen组合 + 1个物理上不太合理的组合
COMPS=(
    "sliced apple"
    "wet dog"
    "broken glass"
    "ripe tomato"
    "melted chocolate"
)

echo "======================================================"
echo "  软提示注意力权重可视化"
echo "  配置文件   : ${CONFIG}"
echo "  输出目录   : ${SAVE_DIR}"
echo "  可视化层数 : 最后 ${TOP_LAYERS} 层"
echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "======================================================"

python -u visualize/attention_weights.py \
    --config      ${CONFIG}                      \
    --checkpoint  ${CHECKPOINT}                  \
    --comps       "${COMPS[@]}"                  \
    --top_layers  ${TOP_LAYERS}                  \
    --save_dir    ${SAVE_DIR}                    \
    --dpi         ${DPI}

echo "======================================================"
echo "  可视化完成，结果保存在: ${SAVE_DIR}"
echo "======================================================"