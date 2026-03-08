#!/bin/bash
#SBATCH --job-name=qwen_edges         # 任务名称
#SBATCH --partition=gpu               # 使用 GPU 分区
#SBATCH --gres=gpu:1                  # 申请 1 块 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4             # 申请 4 个 CPU 核心
#SBATCH --time=2:00:00                # 预计运行时间
#SBATCH --output=logs/mit-states/test/llm_edges_%j.out
#SBATCH --error=logs/mit-states/test/llm_edges_%j.err

# 加载环境
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com

echo $HF_ENDPOINT

echo "=================================================="
echo " >>> 启动 Qwen 边权重 (Edge Weights) 生成 <<< "
echo "=================================================="

# 运行 Python 脚本 (-u 参数保证日志实时输出)
python -u generate_edge_weights.py \
    --data_root /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/data/mit-states \
    --output ./LLM/mit_edge_weights.json \
    --model_id Qwen/Qwen2.5-7B-Instruct

echo "=================================================="
echo " 边权重生成任务结束！"
echo "=================================================="