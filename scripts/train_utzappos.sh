#!/bin/bash
#SBATCH --job-name=rgipcol-zappos       # 作业名称
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00                 # Zappos 较小，12h 足够
#SBATCH --output=logs/ut-zappos/train/%x-%j.out
#SBATCH --error=logs/ut-zappos/train/%x-%j.err

# ─────────────────────────────────────────────
# RGIPCOL  |  训练脚本  |  UT-Zappos
# ─────────────────────────────────────────────

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

CLIP_PATH=/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/checkpoints/ViT-L-14.pt
DATA_ROOT=/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/data
DATASET=ut-zappos

LR=1e-4
LR_PREFIX=5e-5
LR_RGCN=1e-4
BATCH_SIZE=256
EPOCHS=10
NUM_WORKERS=4
SEED=215

# UT-Zappos 节点极少（16 attr + 12 obj），无需基函数分解
RGCN_LAYERS=2
RGCN_BASES=-1
RGCN_DROPOUT=0.2
PREFIX_LEN=3

TIMESTAMP=$(date +%Y%m%d)
JOB_ID=${SLURM_JOB_ID:-manual}
SAVE_PATH=./saved_models/${DATASET}/bs${BATCH_SIZE}_lr${LR}_${TIMESTAMP}_${JOB_ID}

mkdir -p ${SAVE_PATH}
mkdir -p logs/ut-zappos/train

echo "========================================"
echo "  RGIPCOL Training  |  ${DATASET}"
echo "  SLURM Job ID : ${JOB_ID}"
echo "  Save Path    : ${SAVE_PATH}"
echo "  LR=${LR}  BS=${BATCH_SIZE}  Epochs=${EPOCHS}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "========================================"

python -u train.py \
    --config              configs/ut_zappos_config.yaml   \
    --clip_model_path     ${CLIP_PATH}                    \
    --dataset_path        ${DATA_ROOT}/${DATASET}         \
    --save_path           ${SAVE_PATH}                    \
    --lr                  ${LR}                           \
    --lr_prefix           ${LR_PREFIX}                    \
    --lr_rgcn             ${LR_RGCN}                      \
    --batch_size          ${BATCH_SIZE}                   \
    --epochs              ${EPOCHS}                       \
    --num_workers         ${NUM_WORKERS}                  \
    --seed                ${SEED}                         \
    --rgcn_num_layers     ${RGCN_LAYERS}                  \
    --rgcn_num_bases      ${RGCN_BASES}                   \
    --rgcn_dropout        ${RGCN_DROPOUT}                 \
    --prefix_length       ${PREFIX_LEN}

echo "========================================"
echo "  训练完成！结果保存在 ${SAVE_PATH}"
echo "========================================"