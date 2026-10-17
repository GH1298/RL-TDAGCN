#!/bin/bash
# ============================================================
# RL-TDAGCN 并行实验脚本 V2 (与串行实验同时运行，独立保存结果)
#
# 结果保存到: experiments/results_parallel/ (不与串行的 results/ 冲突)
# batch_size 降为 32 (与串行的64错开，降低显存压力)
#
# 使用方法:
#   cd /root/RL-TDAGCN
#   nohup bash run_parallel_v2.sh > parallel_v2_log.txt 2>&1 &
#   tail -f parallel_v2_log.txt
# ============================================================

echo "============================================================"
echo "  RL-TDAGCN 并行实验 V2 (独立结果目录)"
echo "  数据集: PEMS04 (307节点)"
echo "  batch_size: 32 (降低显存，与串行错开)"
echo "  结果保存: experiments/results_parallel/"
echo "  预计耗时: 30-50分钟"
echo "============================================================"
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd /root/RL-TDAGCN

# 独立的结果保存目录（不覆盖串行结果）
PARALLEL_DIR="experiments/results_parallel"
mkdir -p $PARALLEL_DIR
mkdir -p logs/parallel_v2

# ========== 并行启动5组实验 ==========
echo ">>> 启动5组并行实验 (保存到 $PARALLEL_DIR)..."

# 实验1: 基础模型对比
python3 -m experiments.run_experiments \
    --experiment 1 --device cuda --epochs 100 --batch_size 32 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $PARALLEL_DIR \
    > logs/parallel_v2/exp1.log 2>&1 &
PID1=$!
echo "  实验1 (基础模型对比) PID=$PID1"

# 实验2: RL有效性消融
python3 -m experiments.run_experiments \
    --experiment 2 --device cuda --epochs 100 --batch_size 32 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $PARALLEL_DIR \
    > logs/parallel_v2/exp2.log 2>&1 &
PID2=$!
echo "  实验2 (RL消融) PID=$PID2"

# 实验3: Topology有效性消融
python3 -m experiments.run_experiments \
    --experiment 3 --device cuda --epochs 100 --batch_size 32 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $PARALLEL_DIR \
    > logs/parallel_v2/exp3.log 2>&1 &
PID3=$!
echo "  实验3 (Topology消融) PID=$PID3"

# 实验4: 联合效果
python3 -m experiments.run_experiments \
    --experiment 4 --device cuda --epochs 100 --batch_size 32 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $PARALLEL_DIR \
    > logs/parallel_v2/exp4.log 2>&1 &
PID4=$!
echo "  实验4 (联合效果) PID=$PID4"

# 实验5: 多模态消融
python3 -m experiments.run_experiments \
    --experiment 5 --device cuda --epochs 100 --batch_size 32 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $PARALLEL_DIR \
    > logs/parallel_v2/exp5.log 2>&1 &
PID5=$!
echo "  实验5 (多模态消融) PID=$PID5"

echo ""
echo ">>> 5组并行实验已启动"
echo ">>> 结果保存到: $PARALLEL_DIR/"
echo ">>> 监控: tail -f logs/parallel_v2/exp*.log"
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
        echo "  查看日志: tail -30 logs/parallel_v2/exp${i}.log"
    fi
done

echo ""
echo "============================================================"
echo "  并行实验V2完成!"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  失败数: $FAIL/5"
echo "============================================================"
echo ""

# ========== 结果汇总 ==========
echo "  并行结果文件 ($PARALLEL_DIR):"
ls -la $PARALLEL_DIR/*.json 2>/dev/null
echo ""
echo "  并行结果汇总:"
echo "  -----------------------------------------------------------------"
for i in 1 2 3 4 5; do
    if [ -f "$PARALLEL_DIR/exp${i}_results.json" ]; then
        echo "  实验${i}:"
        python3 -c "
import json
with open('$PARALLEL_DIR/exp${i}_results.json') as f:
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

# 对比串行结果（如果存在）
SERIAL_DIR="experiments/results"
if ls $SERIAL_DIR/*.json 2>/dev/null | grep -q .; then
    echo ""
    echo "  串行结果 (experiments/results/):"
    ls -la $SERIAL_DIR/*.json 2>/dev/null
fi

echo ""
echo "  提示: 下载并行结果:"
echo "    scp -P 16583 root@connect.cqa1.seetacloud.com:/root/RL-TDAGCN/experiments/results_parallel/*.json ./"
echo ""
echo "  提示: 下载串行结果:"
echo "    scp -P 16583 root@connect.cqa1.seetacloud.com:/root/RL-TDAGCN/experiments/results/*.json ./"
echo ""
