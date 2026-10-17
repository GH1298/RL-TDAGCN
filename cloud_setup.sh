#!/bin/bash
# ============================================================
# RL-TDAGCN 云GPU环境一键配置脚本
# 在 AutoDL 实例上运行: bash cloud_setup.sh
# ============================================================

set -e  # 任何命令失败立即退出

echo "============================================================"
echo "  RL-TDAGCN 云GPU环境配置"
echo "============================================================"

# 1. 检查GPU
echo ""
echo "[1/6] 检查GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "  GPU 可用"
else
    echo "  [警告] 未检测到GPU，将使用CPU（速度会很慢）"
fi

# 2. 检查Python版本
echo ""
echo "[2/6] 检查Python..."
python3 --version
echo "  Python 版本正常"

# 3. 检查PyTorch和CUDA
echo ""
echo "[3/6] 检查PyTorch和CUDA..."
python3 -c "
import torch
print(f'  PyTorch版本: {torch.__version__}')
print(f'  CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA版本: {torch.version.cuda}')
    print(f'  GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 4. 安装依赖
echo ""
echo "[4/6] 安装Python依赖..."
pip install numpy pandas scikit-learn matplotlib -q
echo "  依赖安装完成"

# 5. 验证项目结构
echo ""
echo "[5/6] 验证项目结构..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查关键文件
for f in train.py models/rlt_dagcn.py experiments/run_experiments.py \
         configs/default.py utils/dataloader.py data/PEMS04/PEMS04_inner.npz \
         data/PEMS04/PEMS04.csv; do
    if [ -f "$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [缺失] $f"
        echo "  请确保所有项目文件已上传到: $SCRIPT_DIR"
        exit 1
    fi
done

# 6. 运行模块验证
echo ""
echo "[6/6] 运行模块验证..."
python3 verify_all.py

echo ""
echo "============================================================"
echo "  环境配置完成! 可以开始运行实验"
echo "============================================================"
echo ""
echo "  快速验证 (5个epoch, CPU):"
echo "    python3 quick_validate.py"
echo ""
echo "  单组实验 (100 epochs, GPU):"
echo "    python3 -m experiments.run_experiments --experiment 1 --device cuda --epochs 100"
echo ""
echo "  全部实验 (5组, GPU):"
echo "    python3 -m experiments.run_experiments --all --device cuda --epochs 100"
echo ""
echo "  或使用一键运行脚本:"
echo "    bash run_all.sh"
echo ""
