#!/bin/bash
#SBATCH --job-name=qwen_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1          # 必须申请 1 块 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/ut-zappos/test/llm_gen_%j.out
#SBATCH --error=logs/ut-zappos/test/llm_gen_%j.err

# 激活环境
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com

echo $HF_ENDPOINT

echo ">>> 启动 Qwen 节点描述生成 <<<"
python -u llm_pipeline/generate_node_descriptions.py \
    --data_root /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/data/ut-zappos \
    --output ./LLM/ut-zappos_node_descriptions.json \
    --model_id Qwen/Qwen2.5-7B-Instruct