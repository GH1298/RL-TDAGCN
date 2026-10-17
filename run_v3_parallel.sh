#!/bin/bash
# ============================================================
# RL-TDAGCN v3 并行实验脚本
# 
# v3改进内容:
#   1. RL负优化修复: batch 16->64, warmup 5->20, interval 1->5
#      ppo_epochs 10->5, rollout 10->30, action [0.3,0.7]->[0.1,0.9]
#      新增: 奖励裁剪[-10,0] + EMA移动平均基线
#   2. Topo微弱优化增强: loss_weight 0.3->0.05, beta 0.9->0.6, gcn 2->3层
#      新增: 度中心性嵌入 + 残差连接
#   3. Weather: 保持关闭(use_multimodal=False)
#
# 并行策略: 5组实验同时运行, batch_size=32降低显存
# 预计耗时: 45-60分钟 (RTX 4090D)
#
# 使用方法:
#   cd /root/RL-TDAGCN
#   nohup bash run_v3_parallel.sh > v3_parallel_log.txt 2>&1 &
#   tail -f v3_parallel_log.txt
# ============================================================

set -e  # 任何命令失败立即停止

echo "============================================================"
echo "  RL-TDAGCN v3 并行实验"
echo "  数据集: PEMS04 (307节点)"
echo "  batch_size: 32 (并行降低显存)"
echo "  结果保存: experiments/results_v3/"
echo "  预计耗时: 45-60分钟 (RTX 4090D)"
echo "============================================================"
echo ""
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

cd /root/RL-TDAGCN

# 独立结果目录 (不覆盖v2结果)
V3_DIR="experiments/results_v3"
mkdir -p $V3_DIR
mkdir -p logs/v3_parallel

# 备份v2结果到backup_v2 (如果存在且未备份)
if [ -d "experiments/results" ] && [ ! -d "experiments/results/backup_v2" ]; then
    mkdir -p experiments/results/backup_v2
    cp experiments/results/*.json experiments/results/backup_v2/ 2>/dev/null || true
    echo "v2结果已备份到 experiments/results/backup_v2/"
fi

echo ""
echo ">>> 启动5组v3并行实验..."
echo ""

# ========== 实验1: 基础模型对比 (6个模型) ==========
echo "  [启动] 实验1: 基础模型对比 (LSTM/STGCN/TGCN/DCRNN/GWN/RL-TDAGCN)"
python3 -m experiments.run_experiments \
    --experiment 1 --device cuda --epochs 100 --batch_size 32 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $V3_DIR \
    > logs/v3_parallel/exp1.log 2>&1 &
PID1=$!
echo "    PID=$PID1"

# ========== 实验2: RL有效性消融 ==========
echo "  [启动] 实验2: RL消融 (DAGGCN vs DAGGCN+RL)"
python3 -m experiments.run_experiments \
    --experiment 2 --device cuda --epochs 100 --batch_size 32 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $V3_DIR \
    > logs/v3_parallel/exp2.log 2>&1 &
PID2=$!
echo "    PID=$PID2"

# ========== 实验3: Topology有效性消融 ==========
echo "  [启动] 实验3: Topo消融 (DAGGCN vs DAGGCN+Topo)"
python3 -m experiments.run_experiments \
    --experiment 3 --device cuda --epochs 100 --batch_size 32 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $V3_DIR \
    > logs/v3_parallel/exp3.log 2>&1 &
PID3=$!
echo "    PID=$PID3"

# ========== 实验4: 联合效果 ==========
echo "  [启动] 实验4: 联合消融 (Base/+RL/+Topo/+RL+Topo)"
python3 -m experiments.run_experiments \
    --experiment 4 --device cuda --epochs 100 --batch_size 32 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $V3_DIR \
    > logs/v3_parallel/exp4.log 2>&1 &
PID4=$!
echo "    PID=$PID4"

# ========== 实验5: 多模态消融 ==========
echo "  [启动] 实验5: 多模态消融 (Traffic vs +Weather)"
python3 -m experiments.run_experiments \
    --experiment 5 --device cuda --epochs 100 --batch_size 32 \
    --hidden_dim 64 --dataset ./data/PEMS04 \
    --save_dir $V3_DIR \
    > logs/v3_parallel/exp5.log 2>&1 &
PID5=$!
echo "    PID=$PID5"

echo ""
echo ">>> 5组实验已全部启动 (并行运行中)"
echo ">>> 监控进度: tail -f logs/v3_parallel/exp*.log"
echo ">>> GPU使用: watch -n 5 nvidia-smi"
echo ""

# 等待所有实验完成
FAIL=0
for i in 1 2 3 4 5; do
    PID=$(eval echo "\$PID$i")
    if wait $PID; then
        echo "  [OK] 实验$i 完成 (PID=$PID)"
    else
        echo "  [FAIL] 实验$i 失败 (PID=$PID)"
        echo "    查看日志: tail -30 logs/v3_parallel/exp${i}.log"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "============================================================"
echo "  v3并行实验完成!"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  失败数: $FAIL/5"
echo "============================================================"
echo ""

# ========== 结果汇总 ==========
echo ""
echo "  v3结果文件 ($V3_DIR):"
ls -la $V3_DIR/*.json 2>/dev/null
echo ""

echo "  ============================================================"
echo "  v3 实验结果汇总"
echo "  ============================================================"
echo ""
for i in 1 2 3 4 5; do
    if [ -f "$V3_DIR/exp${i}_results.json" ]; then
        echo "  --- 实验${i} ---"
        python3 -c "
import json
with open('$V3_DIR/exp${i}_results.json') as f:
    r = json.load(f)
print(f'    {\"模型\":<18} {\"MAE\":<10} {\"RMSE\":<10} {\"SMAPE\":<10}')
for name, m in r.items():
    print(f'    {name:<18} {m[\"MAE\"]:<10.4f} {m[\"RMSE\"]:<10.4f} {m[\"SMAPE\"]*100:<9.2f}%')
"
        echo ""
    else
        echo "  --- 实验${i}: [结果文件缺失] ---"
        echo ""
    fi
done

# ========== v2 vs v3 对比 (如果v2结果存在) ==========
V2_DIR="experiments/results/backup_v2"
if [ -d "$V2_DIR" ] && ls $V2_DIR/*.json 2>/dev/null | grep -q .; then
    echo "  ============================================================"
    echo "  v2 vs v3 对比"
    echo "  ============================================================"
    echo ""
    for i in 1 2 3 4 5; do
        V2_FILE="$V2_DIR/exp${i}_results.json"
        V3_FILE="$V3_DIR/exp${i}_results.json"
        if [ -f "$V2_FILE" ] && [ -f "$V3_FILE" ]; then
            echo "  --- 实验${i} ---"
            python3 -c "
import json
with open('$V2_FILE') as f: v2 = json.load(f)
with open('$V3_FILE') as f: v3 = json.load(f)
print(f'    {\"模型\":<18} {\"v2 MAE\":<10} {\"v3 MAE\":<10} {\"变化\":<10}')
for name in v2:
    if name in v3:
        old = v2[name]['MAE']
        new = v3[name]['MAE']
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
echo "  下载结果到本地:"
echo "    scp -P <端口号> root@<IP>:/root/RL-TDAGCN/experiments/results_v3/*.json ./"
echo ""
echo "  下载日志:"
echo "    scp -P <端口号> root@<IP>:/root/RL-TDAGCN/logs/v3_parallel/*.log ./"
echo ""
