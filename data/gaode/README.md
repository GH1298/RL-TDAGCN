# RL-TDAGCN 环境配置与数据采集完整指南

本文档涵盖三部分内容:
1. **Conda 虚拟环境创建** - 从零搭建 RL-TDAGCN 项目环境
2. **高德交通 + 和风天气数据采集** - 国内城市交通数据采集流程
3. **高德数据与 PEMS 数据并存方案** - 两种数据集无缝切换训练

---

## 第一部分: Conda 虚拟环境创建

> 本机环境: Anaconda 已安装在 `D:\anaconda3`, Python 3.10.20, 虚拟环境名为 `traffic_pred`
> 本机 GPU: NVIDIA GeForce MX130 (2GB), 算力较弱, 使用 CPU 版 PyTorch

### 1.1 初始化 conda (仅需一次)

安装 Anaconda 后, 需要让 PowerShell 能识别 conda 命令:

```powershell
# 1. 解除 PowerShell 脚本执行限制
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. 初始化 conda
D:\anaconda3\Scripts\conda.exe init powershell

# 3. 关闭终端, 重新打开, 验证 conda 可用
conda --version
# 应输出: conda 25.11.1
```

### 1.2 配置清华镜像 (加速包下载)

```powershell
# 配置 conda 镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes

# 配置 pip 镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.3 创建并激活虚拟环境

```powershell
# 创建名为 traffic_pred 的环境, Python 3.10
conda create -n traffic_pred python=3.10 -y

# 激活环境
conda activate traffic_pred

# 验证
python --version
# 应输出: Python 3.10.20
```

环境位置: `C:\Users\21139\.conda\envs\traffic_pred\`

### 1.4 安装项目依赖

> 本机 GPU 为 MX130 (2GB), 算力太弱, 使用 CPU 版 PyTorch。
> 大规模训练在云服务器 (RTX 4090D) 上进行, 本地用于数据采集和小规模调试。

```powershell
conda activate traffic_pred
cd 'e:\1-traffic_congestion\RL-TDAGCN'

# 安装 CPU 版 PyTorch (体积小, 安装快, 不需要 CUDA)
pip install torch==2.1.2+cpu --index-url https://download.pytorch.org/whl/cpu

# 安装其余依赖
pip install numpy pandas scikit-learn matplotlib requests

# 验证安装
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
# 应输出: PyTorch: 2.1.2+cpu  CUDA: False
```

### 1.5 在 VSCode 中选择虚拟环境

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 `Python: Select Interpreter` 并回车
3. 选择 `traffic_pred (Python 3.10.20)` 对应的解释器
4. 路径应为: `C:\Users\21139\.conda\envs\traffic_pred\python.exe`

> **注意**: 每次在 VSCode 终端中运行项目命令前, 确保终端已激活环境:
> ```powershell
> conda activate traffic_pred
> ```

---

## 第二部分: 高德交通 + 和风天气数据采集

### 2.1 API Key 获取

#### 高德地图 API Key
1. 访问 [高德开放平台](https://lbs.amap.com/), 注册并完成实名认证
2. 控制台 -> 应用管理 -> 我的应用 -> 创建新应用
3. 应用类型选 **Web服务**
4. 获取 API Key (32位字母数字串)

#### 和风天气 API Key + 专属 API Host
1. 访问 [和风天气开发者平台](https://dev.qweather.com/), 注册账号
2. 控制台 -> 应用管理 -> 创建应用, 类型选 **Web API**
3. 获取 API Key
4. **重要**: 2026年起 devapi.qweather.com 已停用, 必须获取专属 API Host:
   - 进入 [控制台-设置](https://console.qweather.com/setting)
   - 复制你的 API Host (格式如 `pj3yfucjuk.re.qweatherapi.com`)

### 2.2 配置环境变量

在 VSCode 终端中 (每次新开终端都需要设置, 或用 setx 永久保存):

```powershell
# 高德地图
$env:GAODE_API_KEY = "d9994b6b8afd8a56a45511d800b7c361"

# 和风天气
$env:QWEATHER_API_KEY = "324c2a58f0a84eb683d0122885fad16a"
$env:QWEATHER_API_HOST = "pj3yfucjuk.re.qweatherapi.com"
```

如需永久保存 (写入系统环境变量, 重启 VSCode 生效):
```powershell
setx GAODE_API_KEY "d9994b6b8afd8a56a45511d800b7c361"
setx QWEATHER_API_KEY "324c2a58f0a84eb683d0122885fad16a"
setx QWEATHER_API_HOST "pj3yfucjuk.re.qweatherapi.com"
```

### 2.3 启动数据采集

**方法一: 一键启动脚本 (推荐)**

```powershell
conda activate traffic_pred
cd 'e:\1-traffic_congestion\RL-TDAGCN'
python -u start_collection.py
```

脚本会先做 API 连通性测试, 通过后自动开始 24 小时采集。

**方法二: 分终端采集 (长时间运行推荐)**

> **重要**: 交通采集终端只能开一个! 多开会导致API配额迅速耗尽。

打开两个 VSCode 终端, 分别运行:

```powershell
# === 终端1: 交通数据采集 (只能开一个!) ===
conda activate traffic_pred
cd 'e:\1-traffic_congestion\RL-TDAGCN'
$env:GAODE_API_KEY = "d9994b6b8afd8a56a45511d800b7c361"
python -u -m utils.gaode_fetcher
```

```powershell
# === 终端2: 天气数据采集 ===
conda activate traffic_pred
cd 'e:\1-traffic_congestion\RL-TDAGCN'
$env:QWEATHER_API_KEY = "324c2a58f0a84eb683d0122885fad16a"
$env:QWEATHER_API_HOST = "pj3yfucjuk.re.qweatherapi.com"
python -u -m utils.weather_fetcher
```

> **关键**: 必须加 `-u` 标志确保实时输出, 否则终端可能无显示。
> **采集间隔**: 5min, 与天气数据对齐。配额超限时脚本自动等待次日0点重置后继续。

### 2.4 采集参数说明

在 `utils/gaode_fetcher.py` 的 `__main__` 部分 (文件末尾) 可修改:

| 参数 | 当前值 | 说明 |
|------|--------|------|
| rectangle | `116.28,39.95;116.36,40.01` | 北京中关村区域, 对角线约9.5km |
| interval | 300 秒 | 采集间隔, 5min, 与天气数据对齐 |
| duration_hours | 336 | 采集时长, 14天 |

在 `utils/weather_fetcher.py` 的 `__main__` 部分可修改:
| 参数 | 当前值 | 说明 |
|------|--------|------|
| location | 101010100 | 北京城市ID |
| duration_hours | 24 | 采集时长 |

**常用城市ID**: 北京 101010100 / 上海 101020100 / 广州 101280101 / 深圳 101280601

### 2.5 采集配额

> **重要发现**: 高德交通态势API属于"高级服务接口", 免费配额远低于普通Web服务。
> 实测每日仅约100次调用即超限 (错误码10044: USER_DAILY_QUERY_OVER_LIMIT)。

| API | 实际免费配额 | 5min间隔日用量 | 采集模式 |
|-----|-------------|---------------|---------|
| 高德交通 | ~100次/天 | 288次/天(理论) | 每日采约80条后等待次日重置 |
| 和风天气 | 1000次/天 | 288次/天 | 连续采集, 无中断 |

**实际采集模式 (5min间隔)**:
- 每天可采集约80条 (约6.7小时数据)
- 配额超限后脚本自动等待到次日0点, 配额重置后继续
- 14天预计采集: 80 x 14 = 1120条 (有每日间隔, 但与天气数据时间戳对齐)

> **注意**: 不要同时开多个终端采集交通数据! 每个终端都会独立消耗API配额,
> 多终端并发会导致配额迅速耗尽。脚本已添加文件锁防止并发。

**配额说明**: 高德交通态势API免费配额仅约100次/天, 5min间隔每日理论需288次。
脚本内置智能等待: 采到约80条后自动暂停, 等待次日0点配额重置后继续采集。
因此每天实际采集约6.7小时数据, 14天累计约1120条, 与天气数据时间戳对齐。

### 2.6 采集数据保存位置

所有数据保存在 `data/gaode/` 目录:

| 文件 | 格式 | 说明 |
|------|------|------|
| `traffic_raw.json` | JSON | 高德API原始返回 |
| `weather_raw.json` | JSON | 和风API原始返回 |
| `traffic.npy` | numpy (T,N,3) | 交通特征 [flow近似, occupancy, speed] |
| `adj.csv` | CSV | 邻接边列表 (from, to, cost) |
| `weather_aligned.npy` | numpy (T,6) | 天气特征, 与交通时间对齐 |
| `node_names.json` | JSON | 道路名称到节点索引映射 |
| `meta.json` | JSON | 采集元信息 |

---

## 第三部分: 高德数据与 PEMS 数据并存方案

### 3.1 目录结构 (两套数据并存)

```
RL-TDAGCN/
├── data/
│   ├── PEMS04/                # PEMS04 数据集 (美国加州, 307节点)
│   │   ├── PEMS04_inner.npz   # 交通特征 (T, 307, 3)
│   │   └── PEMS04.csv         # 邻接边列表
│   │
│   ├── gaode/                 # 高德采集数据 (北京中关村, 约180节点)
│   │   ├── traffic_raw.json   # 原始采集记录
│   │   ├── weather_raw.json   # 天气原始记录
│   │   ├── traffic.npy        # 交通特征 (T, N, 3)
│   │   ├── adj.csv            # 邻接边列表
│   │   ├── weather_aligned.npy # 天气特征 (T, 6)
│   │   └── meta.json          # 元信息
│   │
│   ├── PEMS03/                # (如需扩展, 下载后放此)
│   ├── PEMS07/                #
│   └── PEMS08/                #
│
├── utils/
│   ├── dataloader.py          # PEMS 数据加载器
│   ├── gaode_loader.py        # 高德数据加载器
│   ├── gaode_fetcher.py       # 高德采集器
│   └── weather_fetcher.py     # 和风天气采集器
│
├── train.py                   # 训练脚本 (自动检测数据类型)
└── configs/
    └── default.py             # 配置文件
```

### 3.2 训练时切换数据集

`train.py` 已实现**自动检测**: 如果数据目录下有 `traffic.npy` 则用高德加载器, 否则用 PEMS 加载器。只需通过 `--dataset` 参数指定数据目录:

```powershell
conda activate traffic_pred
cd 'e:\1-traffic_congestion\RL-TDAGCN'

# 用 PEMS04 训练
python train.py --dataset data/PEMS04 --epochs 100 --device cpu

# 用高德数据训练 (纯交通模式)
python train.py --dataset data/gaode --epochs 100 --device cpu

# 用高德数据训练 (启用天气多模态)
python train.py --dataset data/gaode --epochs 100 --use_multimodal --device cpu
```

### 3.3 完整训练参数

```powershell
python train.py `
    --dataset data/gaode `
    --epochs 100 `
    --batch_size 64 `
    --lr 0.001 `
    --hidden_dim 64 `
    --use_rl `
    --use_topology `
    --use_multimodal `
    --device cpu `
    --exp_name gaode_multimodal
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | ../DAGGCN_ACMoE_RL/data/PEMS04 | 数据目录路径 |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 64 | 批大小 |
| `--lr` | 0.001 | 学习率 |
| `--hidden_dim` | 64 | 隐藏层维度 |
| `--use_rl` | 开启 | 启用RL图控制器 (加 `--no-use_rl` 关闭) |
| `--use_topology` | 开启 | 启用拓扑约束 (加 `--no-use_topology` 关闭) |
| `--use_multimodal` | 关闭 | 启用天气多模态融合 |
| `--device` | cpu | 训练设备 (cpu 或 cuda) |
| `--exp_name` | default | 实验名称 (结果保存子目录) |

### 3.4 两种数据集对比

| 维度 | PEMS04 | 高德采集 |
|------|--------|---------|
| 数据来源 | 美国加州传感器 | 北京中关村高德API |
| 节点数 | 307 | 约180 (随采集区域变化) |
| 特征数 | 3 (flow/occupancy/speed) | 3 (flow近似/状态映射/speed) |
| 时间间隔 | 5分钟 | 5分钟 (可配置) |
| 天气数据 | 无 | 有 (6维: 温度/湿度/风速/气压/降水/能见度) |
| 数据量 | 16,992 时间步 | 取决于采集时长 (14天约4032步) |
| 邻接矩阵 | 传感器距离 | 道路端点haversine距离 |

### 3.5 扩展其他 PEMS 数据集

如需添加 PEMS03/07/08, 下载后放入 `data/` 对应目录即可:

```powershell
# 目录结构
data/PEMS03/PEMS03_inner.npz
data/PEMS03/PEMS03.csv

# 训练时指定
python train.py --dataset data/PEMS03 --epochs 100
```

下载地址: https://github.com/VeritasYin/STGCN_PyTorch 或 https://github.com/liangzhehan/STSGCN

---

## 附录: 常见问题

### Q1: VSCode 终端运行 python 命令无输出
**A**: 加 `-u` 标志: `python -u -m utils.gaode_fetcher`, 或运行 `python ping.py` 验证。

### Q2: 高德 API 返回 INVALID_PARAMS
**A**: 矩形区域对角线超过 10km 限制。当前配置 `116.28,39.95;116.36,40.01` 对角线约 9.5km, 在限制内。

### Q2.5: 高德 API 返回 10044 USER_DAILY_QUERY_OVER_LIMIT
**A**: 高德交通态势API属于"高级服务接口", 免费配额仅约100次/天 (非2000次)。
5min间隔每日需288次, 远超配额。脚本内置智能等待: 连续3次配额错误后自动暂停,
等待次日0点重置后继续。如果手动重启遇到此错误:
1. 确认没有同时开多个终端采集 (脚本有文件锁, 但旧进程可能残留)
2. 删除 `data/gaode/.collection.lock` 文件后重试
3. 等待次日0点配额重置后重新启动
4. 如需更高配额, 联系高德商务咨询购买高级服务

### Q3: 和风天气 API 返回 403 Invalid Host
**A**: 2026年起 devapi.qweather.com 已停用。需设置 `QWEATHER_API_HOST` 环境变量为控制台中的专属 Host。

### Q4: conda activate 提示需要初始化
**A**: 运行 `D:\anaconda3\Scripts\conda.exe init powershell`, 然后重启 VSCode。如果提示脚本执行被禁止, 先运行 `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`。

### Q5: 训练时 GPU 不可用
**A**: 本机 MX130 显卡 (2GB) 算力太弱, 已安装 CPU 版 PyTorch。本地用 `--device cpu` 训练, 大规模实验在云服务器 (RTX 4090D) 上运行。

### Q6: pip install 报临时目录权限错误
**A**: 在 VSCode 终端 (非 TRAE 终端) 中运行 pip 命令。TRAE 终端有虚拟化限制, 不支持 pip install。
