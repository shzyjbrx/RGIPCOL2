#!/bin/bash
#SBATCH --job-name=rgipcol-mit          # 作业名称
#SBATCH --partition=gpu_mem             # 分区
#SBATCH --gres=gpu:1                    # 申请 1 块 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # CPU 核数
#SBATCH --time=24:00:00
#SBATCH --output=logs/mit-states/train/%x-%j.out
#SBATCH --error=logs/mit-states/train/%x-%j.err

# ─────────────────────────────────────────────
# RGIPCOL  |  训练脚本  |  MIT-States
# ─────────────────────────────────────────────

# 1. 加载系统模块
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

# 2. 激活 Conda 环境
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com

echo $HF_ENDPOINT

export http_proxy=http://172.16.54.201:8888

# 3. 路径与超参数定义
CLIP_PATH=/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/checkpoints/ViT-L-14.pt
DATA_ROOT=/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/data
DATASET=mit-states

LR=1e-4
LR_PREFIX=5e-5
LR_RGCN=1e-4
BATCH_SIZE=64
EPOCHS=10
NUM_WORKERS=4
SEED=0

RGCN_LAYERS=3
RGCN_BASES=-1          # MIT-States 节点少，不需要基函数分解
RGCN_DROPOUT=0.3
PREFIX_LEN=3

TIMESTAMP=$(date +%Y%m%d)
JOB_ID=${SLURM_JOB_ID:-manual}
SAVE_PATH=./saved_models/${DATASET}/bs${BATCH_SIZE}_lr${LR}_${TIMESTAMP}_${JOB_ID}

# 4. 创建必要目录
mkdir -p ${SAVE_PATH}
mkdir -p logs/mit-states/train

# [新增] 打印当前 GPU 资源与状态信息
echo -e "\n>>> 当前 GPU 硬件分配状态 <<<"
nvidia-smi

# 5. 打印环境信息
echo "========================================"
echo "  RGIPCOL Training  |  ${DATASET}"
echo "  SLURM Job ID : ${JOB_ID}"
echo "  Save Path    : ${SAVE_PATH}"
echo "  LR=${LR}  BS=${BATCH_SIZE}  Epochs=${EPOCHS}"
echo "  RGCN: layers=${RGCN_LAYERS}  bases=${RGCN_BASES}  dropout=${RGCN_DROPOUT}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "========================================"

# 6. 运行训练（-u 实时输出日志）
python -u train.py \
    --config              configs/mit_states_config.yaml  \
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