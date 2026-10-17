#!/bin/bash
# ============================================================
# RL-TDAGCN 修复后全部实验重跑脚本
#
# 修复内容:
#   Bug1: RL经验从未push到buffer -> PPO从未更新 (RL空转)
#   Bug2: use_rl=False时访问rl_buffer -> AttributeError崩溃
#
# 使用方法:
#   cd /root/RL-TDAGCN
#   python3 fix_rl_buffer.py          # 先打补丁
#   nohup bash run_all_fixed.sh > experiment_log_fixed.txt 2>&1 &
#   tail -f experiment_log_fixed.txt
# ============================================================

set -e  # 任何命令失败立即停止

echo "============================================================"
echo "  RL-TDAGCN 修复后全部实验重跑"
echo "  数据集: PEMS04 (307节点)"
echo "  设备: CUDA (GPU)"
echo "  Epochs: 100"
echo "  修复: RL经验收集 + use_rl检查"
echo "  预计总耗时: 2-3小时 (RTX 4090D)"
echo "============================================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 进入项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 创建结果目录
mkdir -p experiments/results

# 备份旧结果
if ls experiments/results/*.json 1> /dev/null 2>&1; then
    mkdir -p experiments/results/backup_old
    cp experiments/results/*.json experiments/results/backup_old/ 2>/dev/null || true
    echo "旧结果已备份到 experiments/results/backup_old/"
    echo ""
fi

# ========== 实验1: 基础模型对比 ==========
echo "============================================================"
echo "  [1/5] 实验1: 基础模型对比 (修复后RL-TDAGCN)"
echo "  模型: LSTM, STGCN, TGCN, DCRNN, GWN, RL-TDAGCN"
echo "============================================================"
python3 -m experiments.run_experiments \
    --experiment 1 \
    --device cuda \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 64 \
    --dataset ./data/PEMS04 \
    --save_dir experiments/results

echo ""
echo "  实验1完成"
echo ""

# ========== 实验2: RL有效性消融 ==========
echo "============================================================"
echo "  [2/5] 实验2: RL有效性消融"
echo "  对比: DAGGCN vs DAGGCN+RL"
echo "============================================================"
python3 -m experiments.run_experiments \
    --experiment 2 \
    --device cuda \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 64 \
    --dataset ./data/PEMS04 \
    --save_dir experiments/results

echo ""
echo "  实验2完成"
echo ""

# ========== 实验3: Topology有效性消融 ==========
echo "============================================================"
echo "  [3/5] 实验3: Topology有效性消融"
echo "  对比: DAGGCN vs DAGGCN+Topo"
echo "============================================================"
python3 -m experiments.run_experiments \
    --experiment 3 \
    --device cuda \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 64 \
    --dataset ./data/PEMS04 \
    --save_dir experiments/results

echo ""
echo "  实验3完成"
echo ""

# ========== 实验4: 联合效果 ==========
echo "============================================================"
echo "  [4/5] 实验4: 联合效果"
echo "  对比: Base / +RL / +Topo / +RL+Topo"
echo "============================================================"
python3 -m experiments.run_experiments \
    --experiment 4 \
    --device cuda \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 64 \
    --dataset ./data/PEMS04 \
    --save_dir experiments/results

echo ""
echo "  实验4完成"
echo ""

# ========== 实验5: 多模态消融 ==========
echo "============================================================"
echo "  [5/5] 实验5: 多模态消融 (修复后RL)"
echo "  对比: Traffic only vs +Weather"
echo "============================================================"
python3 -m experiments.run_experiments \
    --experiment 5 \
    --device cuda \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 64 \
    --dataset ./data/PEMS04 \
    --save_dir experiments/results

echo ""
echo "  实验5完成"
echo ""

# ========== 汇总 ==========
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))

echo "============================================================"
echo "  全部实验完成!"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  总耗时: ${HOURS}小时${MINS}分钟"
echo "============================================================"
echo ""
echo "  结果文件:"
ls -la experiments/results/*.json
echo ""
echo "  结果汇总:"
echo "  -----------------------------------------------------------------"
for i in 1 2 3 4 5; do
    if [ -f "experiments/results/exp${i}_results.json" ]; then
        echo "  实验${i}:"
        python3 -c "
import json
with open('experiments/results/exp${i}_results.json') as f:
    r = json.load(f)
print(f'    {\"模型\":<15} {\"MAE\":<10} {\"RMSE\":<10} {\"SMAPE\":<10}')
for name, m in r.items():
    print(f'    {name:<15} {m[\"MAE\"]:<10.4f} {m[\"RMSE\"]:<10.4f} {m[\"SMAPE\"]*100:<9.2f}%')
"
        echo ""
    fi
done
echo "  -----------------------------------------------------------------"
echo ""
echo "  提示: 使用以下命令将结果下载到本地:"
echo "    scp -P <端口号> root@<IP地址>:/root/RL-TDAGCN/experiments/results/*.json ./"
echo ""
