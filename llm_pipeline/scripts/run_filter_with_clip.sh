#!/bin/bash
#SBATCH --job-name=clip_filter          # 作业名称
#SBATCH --partition=gpu                 # 申请 GPU 分区
#SBATCH --gres=gpu:1                    # 申请 1 块 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00                  # CLIP 筛选速度快，2 小时足够
#SBATCH --output=logs/llm/clip_filter_%j.out
#SBATCH --error=logs/llm/clip_filter_%j.err

set -e

# 1. 加载系统模块
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

# 2. 激活 Conda 环境
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

# 3. 核心参数配置
DATASET="mit-states"
CANDIDATE_PATH="./LLM/${DATASET}_candidate_descriptions.json"
FILTERED_PATH="./LLM/${DATASET}_filtered_descriptions.json"
CLIP_PATH="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/checkpoints/ViT-L-14.pt"

echo "============================================================"
echo "  🚀 [Step 2] 启动 CLIP 质量筛选"
echo "  SLURM Job ID : ${SLURM_JOB_ID:-manual}"
echo "  候选文件     : ${CANDIDATE_PATH}"
echo "  CLIP 路径    : ${CLIP_PATH}"
echo "  GPU节点      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# 执行筛选脚本 sbatch llm_pipeline/scripts/run_filter_with_clip.sh
python -u llm_pipeline/filter_with_clip.py \
    --candidate_path ${CANDIDATE_PATH} \
    --save_path ${FILTERED_PATH} \
    --clip_path ${CLIP_PATH}

echo "============================================================"
echo "  ✅ [Step 2 完成] 高质量描述已保存至: ${FILTERED_PATH}"
echo "============================================================"