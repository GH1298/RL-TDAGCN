#!/bin/bash
# ============================================================
# RL-TDAGCN v4 并行实验脚本
#
# v4 改进 (相比v3):
#   1. batch_size: 32 -> 48 (显存利用率 58% -> 75%)
#   2. AMP混合精度: 自动启用 (~1.5x加速)
#   3. DataLoader: num_workers=4 + pin_memory (消除I/O瓶颈)
#   4. 断点续训: 每5 epoch保存checkpoint, 崩溃可恢复
#   5. 实验级跳过: 已完成的variant自动跳过
#   6. --resume: 支持从中断处继续
#
# 预计耗时: 2.5-3小时 (RTX 4090D, vs v3的5.5小时)
# 预计费用: 3.0-3.4元 (vs v3的6.2元)
#
# 使用方法:
#   首次运行:  nohup bash run_v4_parallel.sh > v4_log.txt 2>&1 &
#   断点续训:  nohup bash run_v4_parallel.sh --resume > v4_resume_log.txt 2>&1 &
#   查看进度:  tail -f v4_log.txt
#   监控GPU:   watch -n 5 nvidia-smi
# ============================================================

set -e  # 任何命令失败立即停止

# 解析 --resume 参数
RESUME_FLAG=""
if [ "$1" == "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "[v4] 断点续训模式: 将跳过已完成的实验和variant"
fi

echo "============================================================"
echo "  RL-TDAGCN v4 并行实验"
echo "  数据集: PEMS04 (307节点, 3特征)"
echo "  batch_size: 48 (v3为32, 显存利用率提升)"
echo "  AMP混合精度: 自动启用"
echo "  断点续训: 每5 epoch保存checkpoint"
echo "  结果保存: experiments/results_v4/"
echo "  预计耗时: 2.5-3小时 (RTX 4090D)"
echo "  预计费用: 3.0-3.4元 (学生75折)"
echo "============================================================"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd /root/RL-TDAGCN

# 独立结果目录 (不覆盖v3结果)
V4_DIR="experiments/results_v4"
mkdir -p $V4_DIR
mkdir -p logs/v4_parallel

echo ">>> 启动5组v4并行实验..."
echo ""

# ========== 实验1: 基础模型对比 (6个模型) ==========
echo "  [启动] 实验1: 基础模型对比 (LSTM/STGCN/TGCN/DCRNN/GWN/RL-TDAGCN)"
python3 -m experiments.run_experiments \
    --experiment 1 --device cuda --epochs 100 --batch_size 48 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $V4_DIR $RESUME_FLAG \
    > logs/v4_parallel/exp1.log 2>&1 &
PID1=$!
echo "    PID=$PID1"

# ========== 实验2: RL有效性消融 ==========
echo "  [启动] 实验2: RL消融 (DAGGCN vs DAGGCN+RL)"
python3 -m experiments.run_experiments \
    --experiment 2 --device cuda --epochs 100 --batch_size 48 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $V4_DIR $RESUME_FLAG \
    > logs/v4_parallel/exp2.log 2>&1 &
PID2=$!
echo "    PID=$PID2"

# ========== 实验3: Topology有效性消融 ==========
echo "  [启动] 实验3: Topo消融 (DAGGCN vs DAGGCN+Topo)"
python3 -m experiments.run_experiments \
    --experiment 3 --device cuda --epochs 100 --batch_size 48 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $V4_DIR $RESUME_FLAG \
    > logs/v4_parallel/exp3.log 2>&1 &
PID3=$!
echo "    PID=$PID3"

# ========== 实验4: 联合效果 ==========
echo "  [启动] 实验4: 联合消融 (Base/+RL/+Topo/+RL+Topo)"
python3 -m experiments.run_experiments \
    --experiment 4 --device cuda --epochs 100 --batch_size 48 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $V4_DIR $RESUME_FLAG \
    > logs/v4_parallel/exp4.log 2>&1 &
PID4=$!
echo "    PID=$PID4"

# ========== 实验5: 多模态消融 ==========
echo "  [启动] 实验5: 多模态消融 (Traffic vs +Weather)"
python3 -m experiments.run_experiments \
    --experiment 5 --device cuda --epochs 100 --batch_size 48 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $V4_DIR $RESUME_FLAG \
    > logs/v4_parallel/exp5.log 2>&1 &
PID5=$!
echo "    PID=$PID5"

echo ""
echo ">>> 5组实验已全部启动 (并行运行中)"
echo ">>> 监控进度: tail -f logs/v4_parallel/exp*.log"
echo ">>> GPU使用: watch -n 5 nvidia-smi"
echo ">>> v4优化: AMP加速 + batch48 + DataLoader多进程 + 断点续训"
echo ""

# 等待所有实验完成
FAIL=0
for i in 1 2 3 4 5; do
    PID=$(eval echo "\$PID$i")
    if wait $PID; then
        echo "  [OK] 实验$i 完成 (PID=$PID) $(date '+%H:%M:%S')"
    else
        echo "  [FAIL] 实验$i 失败 (PID=$PID) $(date '+%H:%M:%S')"
        echo "    查看日志: tail -30 logs/v4_parallel/exp${i}.log"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "============================================================"
echo "  v4并行实验完成!"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  失败数: $FAIL/5"
echo "============================================================"
echo ""

# ========== 结果汇总 ==========
echo ""
echo "  v4结果文件 ($V4_DIR):"
ls -la $V4_DIR/*.json 2>/dev/null
echo ""

echo "  ============================================================"
echo "  v4 实验结果汇总"
echo "  ============================================================"
echo ""
for i in 1 2 3 4 5; do
    if [ -f "$V4_DIR/exp${i}_results.json" ]; then
        echo "  --- 实验${i} ---"
        python3 -c "
import json
with open('$V4_DIR/exp${i}_results.json') as f:
    r = json.load(f)
print(f'    {\"Model\":<18} {\"MAE\":<10} {\"RMSE\":<10} {\"SMAPE\":<10}')
for name, m in r.items():
    print(f'    {name:<18} {m[\"MAE\"]:<10.4f} {m[\"RMSE\"]:<10.4f} {m[\"SMAPE\"]*100:<9.2f}%')
"
        echo ""
    else
        echo "  --- 实验${i}: [结果文件缺失, 使用 --resume 重跑] ---"
        echo "    nohup bash run_v4_parallel.sh --resume > v4_retry.log 2>&1 &"
        echo ""
    fi
done

# ========== v3 vs v4 对比 (如果v3结果存在) ==========
V3_DIR="experiments/results_v3"
if [ -d "$V3_DIR" ] && ls $V3_DIR/*.json 2>/dev/null | grep -q .; then
    echo "  ============================================================"
    echo "  v3 vs v4 对比"
    echo "  ============================================================"
    echo ""
    for i in 1 2 3 4 5; do
        V3_FILE="$V3_DIR/exp${i}_results.json"
        V4_FILE="$V4_DIR/exp${i}_results.json"
        if [ -f "$V3_FILE" ] && [ -f "$V4_FILE" ]; then
            echo "  --- 实验${i} ---"
            python3 -c "
import json
with open('$V3_FILE') as f: v3 = json.load(f)
with open('$V4_FILE') as f: v4 = json.load(f)
print(f'    {\"Model\":<18} {\"v3 MAE\":<10} {\"v4 MAE\":<10} {\"Change\":<10}')
for name in v3:
    if name in v4:
        old = v3[name]['MAE']
        new = v4[name]['MAE']
        diff = (new - old) / old * 100
        sign = '+' if diff > 0 else ''
        print(f'    {name:<18} {old:<10.4f} {new:<10.4f} {sign}{diff:.2f}%')
"
            echo ""
        fi
    done
fi

echo "  ============================================================"
echo ""
echo "  v4 优化效果总结:"
echo "    batch_size: 32 -> 48"
echo "    AMP混合精度: 已启用"
echo "    DataLoader: num_workers=4 + pin_memory"
echo "    断点续训: 每5 epoch保存checkpoint"
echo "    预计加速: ~2.3x (v3: 5.5h -> v4: 2.5-3h)"
echo "    预计费用: ~3.0-3.4元 (v3: 6.2元)"
echo ""
echo "  下载结果到本地:"
echo "    scp -P 16583 root@connect.cqa1.seetacloud.com:/root/RL-TDAGCN/experiments/results_v4/*.json ./"
echo ""
echo "  下载日志:"
echo "    scp -P 16583 root@connect.cqa1.seetacloud.com:/root/RL-TDAGCN/logs/v4_parallel/*.log ./"
echo ""
echo "  如有实验失败, 使用断点续训重跑:"
echo "    nohup bash run_v4_parallel.sh --resume > v4_retry.log 2>&1 &"
echo ""
