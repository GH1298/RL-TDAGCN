#!/bin/bash
# ============================================================
# RL-TDAGCN 并行实验脚本 (5组实验同时运行)
#
# 利用 RTX 4090D 24GB 显存，同时运行5组实验
# 预计耗时: 30-40分钟 (串行需2-3小时)
#
# 注意: 余额不足时优先使用此脚本快速完成实验
#
# 使用方法:
#   cd /root/RL-TDAGCN
#   nohup bash run_parallel.sh > parallel_log.txt 2>&1 &
#   tail -f parallel_log.txt
# ============================================================

echo "============================================================"
echo "  RL-TDAGCN 并行实验 (5组同时运行)"
echo "  数据集: PEMS04 (307节点)"
echo "  设备: CUDA (RTX 4090D)"
echo "  预计耗时: 30-40分钟"
echo "============================================================"
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd /root/RL-TDAGCN
mkdir -p experiments/results

# 创建独立日志目录
mkdir -p logs/parallel

# ========== 并行启动5组实验 ==========
echo ">>> 启动5组并行实验..."

# 实验1: 基础模型对比
python3 -m experiments.run_experiments \
    --experiment 1 --device cuda --epochs 100 --batch_size 64 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir experiments/results \
    > logs/parallel/exp1.log 2>&1 &
PID1=$!
echo "  实验1 (基础模型对比) PID=$PID1"

# 实验2: RL有效性消融
python3 -m experiments.run_experiments \
    --experiment 2 --device cuda --epochs 100 --batch_size 64 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir experiments/results \
    > logs/parallel/exp2.log 2>&1 &
PID2=$!
echo "  实验2 (RL消融) PID=$PID2"

# 实验3: Topology有效性消融
python3 -m experiments.run_experiments \
    --experiment 3 --device cuda --epochs 100 --batch_size 64 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir experiments/results \
    > logs/parallel/exp3.log 2>&1 &
PID3=$!
echo "  实验3 (Topology消融) PID=$PID3"

# 实验4: 联合效果
python3 -m experiments.run_experiments \
    --experiment 4 --device cuda --epochs 100 --batch_size 64 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir experiments/results \
    > logs/parallel/exp4.log 2>&1 &
PID4=$!
echo "  实验4 (联合效果) PID=$PID4"

# 实验5: 多模态消融
python3 -m experiments.run_experiments \
    --experiment 5 --device cuda --epochs 100 --batch_size 64 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir experiments/results \
    > logs/parallel/exp5.log 2>&1 &
PID5=$!
echo "  实验5 (多模态消融) PID=$PID5"

echo ""
echo ">>> 5组实验已全部启动，等待完成..."
echo ">>> 监控进度: tail -f logs/parallel/exp*.log"
echo ""

# 等待所有实验完成
FAIL=0
for i in 1 2 3 4 5; do
    PID=$(eval echo "\$PID$i")
    if wait $PID; then
        echo "  [OK] 实验$i 完成 (PID=$PID)"
    else
        echo "  [FAIL] 实验$i 失败 (PID=$PID)"
        FAIL=$((FAIL+1))
        echo "  查看日志: cat logs/parallel/exp${i}.log | tail -30"
    fi
done

echo ""
END_TIME=$(date +%s)
START_TIME=$(date +%s -d "$(echo '开始时间: ' | head -1)")
echo "============================================================"
echo "  并行实验全部完成!"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  失败数: $FAIL/5"
echo "============================================================"
echo ""

# ========== 结果汇总 ==========
echo "  结果文件:"
ls -la experiments/results/*.json 2>/dev/null
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
    else
        echo "  实验${i}: [结果文件缺失]"
        echo ""
    fi
done
echo "  -----------------------------------------------------------------"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "  [警告] 有 $FAIL 组实验失败，请检查对应日志:"
    echo "    cat logs/parallel/exp*.log | grep -i error"
    echo ""
    echo "  常见原因:"
    echo "    1. 显存不足 -> 降低 batch_size 到 32 重试"
    echo "    2. 某个实验崩溃 -> 单独重跑该实验"
fi

echo ""
echo "  提示: 下载结果到本地:"
echo "    scp -P 16583 root@connect.cqa1.seetacloud.com:/root/RL-TDAGCN/experiments/results/*.json ./"
echo ""
