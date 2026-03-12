#!/bin/bash
#SBATCH --job-name=print-env
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/env_info_%j.out
#SBATCH --error=logs/env_info_%j.err

# ─────────────────────────────────────────────
# 环境信息打印脚本
# ─────────────────────────────────────────────

# 1. 加载系统模块
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

# 2. 激活 Conda 环境
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Troika/code/miniconda3/envs/recAtk/xuan-czsl-py38

mkdir -p logs

echo "========================================"
echo "       实验环境信息汇总"
echo "========================================"

# ──── 硬件环境 ────

echo ""
echo "[ 硬件环境 ]"

# GPU 信息
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
           --format=csv,noheader | \
while IFS=',' read -r idx name mem drv; do
    echo "  GPU ${idx}   : ${name}"
    echo "  显存容量  : ${mem}"
    echo "  驱动版本  : ${drv}"
done

echo ""
echo "--- GPU 完整状态 ---"
nvidia-smi

# CPU 信息
echo ""
echo "--- CPU ---"
CPU_MODEL=$(lscpu | grep "Model name" | awk -F: '{print $2}' | xargs)
CPU_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
CPU_THREADS=$(lscpu | grep "Thread(s) per core" | awk '{print $4}')
echo "  型号     : ${CPU_MODEL}"
echo "  逻辑核数 : ${CPU_CORES}"
echo "  每核线程 : ${CPU_THREADS}"

# 内存信息
echo ""
echo "--- 内存 ---"
MEM_TOTAL=$(free -h | grep "^Mem:" | awk '{print $2}')
MEM_TYPE=$(sudo dmidecode -t memory 2>/dev/null | grep "Type:" | grep -v "Unknown" | head -1 | awk '{print $2}' || echo "DDR（需root权限获取型号）")
echo "  总容量   : ${MEM_TOTAL}"
echo "  内存类型 : ${MEM_TYPE}"

# ──── 软件环境 ────

echo ""
echo "[ 软件环境 ]"

# 操作系统
echo ""
echo "--- 操作系统 ---"
OS_NAME=$(lsb_release -d 2>/dev/null | awk -F: '{print $2}' | xargs || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')
KERNEL=$(uname -r)
echo "  发行版   : ${OS_NAME}"
echo "  内核版本 : ${KERNEL}"

# Python 版本
echo ""
echo "--- Python ---"
PYTHON_VER=$(python --version 2>&1)
PYTHON_PATH=$(which python)
echo "  版本     : ${PYTHON_VER}"
echo "  路径     : ${PYTHON_PATH}"

# PyTorch 与 CUDA 版本
echo ""
echo "--- PyTorch / CUDA ---"
python - << 'EOF'
import torch
print(f"  PyTorch 版本 : {torch.__version__}")
print(f"  CUDA 可用    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 版本    : {torch.version.cuda}")
    print(f"  cuDNN 版本   : {torch.backends.cudnn.version()}")
    print(f"  GPU 数量     : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i} 名称  : {props.name}")
        print(f"  GPU {i} 显存  : {props.total_memory / 1024**3:.1f} GB")
EOF

# 关键 Python 包版本
echo ""
echo "--- 关键依赖包 ---"
python - << 'EOF'
packages = [
    "torch", "torchvision", "numpy", "clip",
    "torch_geometric", "transformers", "PIL", "yaml"
]
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "版本未知")
        print(f"  {pkg:<20} : {ver}")
    except ImportError:
        print(f"  {pkg:<20} : 未安装")
EOF

# CUDA 工具链版本
echo ""
echo "--- CUDA 工具链 ---"
if command -v nvcc &> /dev/null; then
    NVCC_VER=$(nvcc --version | grep "release" | awk '{print $6}' | tr -d ',')
    echo "  nvcc 版本    : ${NVCC_VER}"
else
    echo "  nvcc         : 未找到（模块可能未加载）"
fi

echo ""
echo "========================================"
echo "  环境信息打印完成"
echo "  输出文件: logs/env_info_${SLURM_JOB_ID}.out"
echo "========================================"