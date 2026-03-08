#!/bin/bash
#SBATCH --job-name=test_cgqa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00             # C-GQA 规模较大，建议给 2 小时以防开放世界计算超时
#SBATCH --output=logs/cgqa/test/%j.out
#SBATCH --error=logs/cgqa/test/%j.err

# 1. 加载必要的系统模块 (保持与训练环境一致)
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

# 2. 激活 Conda 环境
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

# ================= 配置区域 =================
# 1. 配置文件路径 (务必确认 configs/cgqa_config.yaml 存在)
CONFIG="configs/cgqa_config.yaml"

# 2. 模型权重路径 
# 引用自 Job 1152807 的保存目录 
CHECKPOINT="./saved_models/cgqa/bs64_lr1e-4_20260308_1152807/best_model.pt"

# 3. 测试集划分
SPLIT="test"

# ================= 运行命令 =================
echo "=================================================="
echo " 启动 RGIPCOL2 测试流水线 (C-GQA)"
echo " 配置文件: ${CONFIG}"
echo " 模型权重: ${CHECKPOINT}"
echo "=================================================="

# 1. 执行闭合世界 (Closed-World) 测试
# 候选空间: 5592 (Seen) + 923 (Unseen) 
echo -e "\n>>> 运行闭合世界 (Closed-World) 测试 <<<"
python -u test.py \
  --config ${CONFIG} \
  --checkpoint ${CHECKPOINT} \
  --split ${SPLIT}

# 2. 执行开放世界 (Open-World) 测试
# 候选空间: 278,775 个全空间组合 
echo -e "\n>>> 运行开放世界 (Open-World) 测试 <<<"
python -u test.py \
  --config ${CONFIG} \
  --checkpoint ${CHECKPOINT} \
  --split ${SPLIT} \
  --open_world

echo "Test finished."