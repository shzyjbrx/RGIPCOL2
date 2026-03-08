#!/bin/bash
#SBATCH --job-name=test_rgipcol2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/ut-zappos/test/%j.out
#SBATCH --error=logs/ut-zappos/test/%j.err

# 1. 加载必要的系统模块 (ARM 架构环境)
# 这里的模块加载保持你之前编译 PyTorch 时的环境，防止运行时库缺失
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6     # 对应你安装 PyTorch 2.1 时用的 CUDA 版本

# 2. 激活 Conda 环境
source ~/.bashrc
# 刚才辛苦配置好的环境名
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

# ================= 配置区域 =================
# 1. 配置文件路径 (包含了数据集路径、CLIP路径、超参数等)
CONFIG="configs/ut_zappos_config.yaml"

# 2. 【关键】模型权重路径
# 请将下面的路径修改为您想要测试的那个实验生成的 best_model.pt
# (参考您之前的训练日志，例如 ./saved_models/ut-zappos/bs32_lr1e-4_20260308_1152345/best_model.pt)
CHECKPOINT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/RGIPCOL2/saved_models/ut-zappos/bs128_lr1e-4_20260308_1152365/best_model.pt"

# 3. 测试集划分 (通常选 test，如果想查验验证集可以改 val)
SPLIT="test"

# ================= 运行命令 =================
echo "=================================================="
echo " 启动 RGIPCOL2 测试流水线"
echo " 配置文件: ${CONFIG}"
echo " 模型权重: ${CHECKPOINT}"
echo "=================================================="

# 1. 执行闭合世界 (Closed-World) 测试
echo -e "\n>>> 运行闭合世界 (Closed-World) 测试 <<<"
python -u test.py \
  --config ${CONFIG} \
  --checkpoint ${CHECKPOINT} \
  --split ${SPLIT}

# 2. 执行开放世界 (Open-World) 测试
echo -e "\n>>> 运行开放世界 (Open-World) 测试 <<<"
python -u test.py \
  --config ${CONFIG} \
  --checkpoint ${CHECKPOINT} \
  --split ${SPLIT} \
  --open_world

echo "Test finished."