#!/bin/bash
#SBATCH --job-name=extract_feats        # 作业名称
#SBATCH --partition=gpu                 # 申请 GPU 分区
#SBATCH --gres=gpu:1                    # 申请 1 块 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # CPU 核数，提取特征不需要太多CPU
#SBATCH --time=2:00:00                  # 离线提特征通常很快，2小时非常充足
#SBATCH --output=logs/ut-zappos/test/extract_feats_%j.out
#SBATCH --error=logs/ut-zappos/test/extract_feats_%j.err

# ─────────────────────────────────────────────
# RGIPCOL2  |  离线 LLM 特征提取脚本
# ─────────────────────────────────────────────

# 1. 加载系统模块
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

# 2. 激活 Conda 环境
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

# 3. 路径与参数配置
JSON_PATH="./LLM/ut-zappos_node_descriptions.json"
SAVE_PATH="./LLM/ut-zappos_node_features.pt"
CLIP_PATH="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/checkpoints/ViT-L-14.pt"
BATCH_SIZE=256

echo "========================================"
echo "  启动离线特征提取 (CLIP Text Encoder)"
echo "  SLURM Job ID : ${SLURM_JOB_ID:-manual}"
echo "  JSON 输入    : ${JSON_PATH}"
echo "  PT 输出      : ${SAVE_PATH}"
echo "  CLIP 权重    : ${CLIP_PATH}"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "========================================"

# 4. 执行 Python 脚本 (-u 保证日志实时输出)
python -u llm_pipeline/extract_llm_features.py \
    --json_path   ${JSON_PATH} \
    --save_path   ${SAVE_PATH} \
    --clip_path   ${CLIP_PATH} \
    --batch_size  ${BATCH_SIZE}

echo "========================================"
echo "  特征提取完成！结果保存在 ${SAVE_PATH}"
echo "========================================"