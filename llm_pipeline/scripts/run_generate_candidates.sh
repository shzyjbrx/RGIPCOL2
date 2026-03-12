#!/bin/bash
#SBATCH --job-name=gen_cands            # 作业名称
#SBATCH --partition=gpu                 # 申请 GPU 分区 (本地 Qwen 模型需要 GPU)
#SBATCH --gres=gpu:1                    # 申请 1 块 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00                 # 本地模型推理可能较慢，申请 12 小时
#SBATCH --output=logs/llm/gen_cands_%j.out
#SBATCH --error=logs/llm/gen_cands_%j.err

set -e

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

# 3. 核心参数配置
DATASET="mit-states"
DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/data/${DATASET}"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
K_CANDIDATES=5
CANDIDATE_PATH="./LLM/${DATASET}_candidate_descriptions_Qwen2.json"


echo "============================================================"
echo "  🚀 [Step 1] 启动本地大模型生成候选描述"
echo "  SLURM Job ID : ${SLURM_JOB_ID:-manual}"
echo "  数据集       : ${DATASET}"
echo "  本地模型     : ${MODEL_ID}"
echo "  候选数量(K)  : ${K_CANDIDATES}"
echo "  GPU节点      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# 执行生成脚本 sbatch llm_pipeline/scripts/run_generate_candidates.sh
python -u llm_pipeline/generate_candidates.py \
    --data_root ${DATA_ROOT} \
    --save_path ${CANDIDATE_PATH} \
    --model_id ${MODEL_ID} \
    --k_candidates ${K_CANDIDATES}

echo "============================================================"
echo "  ✅ [Step 1 完成] 候选文件已保存至: ${CANDIDATE_PATH}"
echo "============================================================"