"""
RL-TDAGCN 默认超参数配置
所有实验共享的基础配置

v5 改进日志 (2026-08-04):
  基于v3实验日志深度分析:
  - RL联合负优化根因: PPO更新导致图策略突变, 主模型特征失配
    证据: Exp4 +RL 在epoch25(最佳MAE=1.8874)后, PPO更新使epoch28 MAE跳至1.9857
  - RL修复: lr 3e-4->1e-4(减小策略变化), warmup 20->30(充分收敛),
            interval 5->10(降低更新频率), clip 0.2->0.1(限制策略突变),
            新增rl_freeze_main_model=True(PPO更新epoch冻结主模型梯度)
  - 训练参数: epochs 100->150(复杂模型需更多时间), patience 15->20(避免过早停止)
  - 高德采集: 新增多Key分时段轮转配置(4Key各6h=全天288次覆盖)

v3 改进日志 (2026-08-02):
  基于v2实验结果分析(10组实验)的改进:
  - RL负优化修复: batch 16->64, warmup 5->20, interval 1->5, ppo_epochs 10->5
  - RL负优化修复: rollout 10->30, 动作范围 [0.3,0.7]->[0.1,0.9], entropy 0.01->0.02
  - Topo微弱优化增强: loss_weight 0.3->0.05, beta 0.9->0.6, gcn_layers 2->3
  - Weather负优化: 默认关闭(use_multimodal=False), 等真实天气数据

v2 改进日志 (2026-07-31):
  - RL: 增加 PPO 更新频率 (warmup 10->5, interval 5->1, ppo_epochs 4->10)
  - RL: 新增相对奖励机制 (reward_type='relative')
  - RL: 新增课程学习支持 (curriculum_enabled=True)
  - RL: 连续动作空间收窄 [0.3, 0.7] 减少探索空间
  - Topo: 损失权重 0.1->0.3, 融合 beta 0.7->0.9
  - Topo: 新增可视化与路网类型支持
  - 新增高德数据采集配置
"""


class Config:
    # ========== 数据 ==========
    dataset = 'PEMS04'           # 数据集名称
    data_dir = 'data/PEMS04'     # 数据目录（相对于项目根目录）
    npz_name = 'PEMS04_inner.npz'  # 数据文件名
    csv_name = 'PEMS04.csv'      # 边距离 CSV 文件名
    num_nodes = 307              # PEMS04 传感器数量
    input_dim = 3                # 输入特征维度 (flow, occupancy, speed)
    output_dim = 1               # 输出维度 (预测速度)

    # ========== 序列 ==========
    seq_len = 12                 # 输入序列长度 (12 * 5min = 1小时)
    pred_len = 12                # 预测步数 (12 * 5min = 1小时)

    # ========== 模型 ==========
    hidden_dim = 64              # 隐藏层维度
    embed_dim = 10               # 动态图节点嵌入维度
    num_layers = 2               # 编码器层数
    num_heads = 4                # 注意力头数
    dropout = 0.3                # Dropout 率
    use_cheb = False             # 是否使用切比雪夫多项式近似（False 用扩散卷积）
    K = 3                        # 切比雪夫多项式阶数（use_cheb=True 时生效）

    # ========== 动态图 ==========
    graph_embed_dim = 10         # 动态图节点嵌入维度
    graph_heads = 4              # 图注意力头数
    graph_temp = 0.5             # softmax 温度参数

    # ========== 拓扑约束 (v3改进) ==========
    # v2问题: loss_weight=0.3太大干扰主任务, beta=0.9太高拓扑几乎无效
    # v3修复: 降低loss_weight到0.05减少干扰, 降低beta到0.6让拓扑占40%
    use_topology = True          # 是否启用拓扑约束
    topo_loss_weight = 0.05      # [v3改] 拓扑损失权重 0.3->0.05, 大幅降低对主任务的干扰
    topo_fusion_beta = 0.6       # [v3改] 初始融合 beta 0.9->0.6, 拓扑约束占40%增强效果
    topo_encoder_type = 'gcn'    # 拓扑编码器类型: 'gcn' 或 'laplacian'
    topo_gcn_layers = 3          # [v3改] 拓扑GCN层数 2->3, 更深层拓扑特征提取
    topo_visualize = True        # [P2新增] 是否输出拓扑约束可视化
    topo_visualize_interval = 10 # [P2新增] 每 N 个 epoch 输出一次热力图
    topo_use_road_type = False   # [P2新增] 是否使用道路等级拓扑 (高德数据时开启)

    # ========== 强化学习 (v5改进 - 修复联合负优化) ==========
    # v3问题: 独立消融RL有效(-0.69% MAE), 但联合设置RL导致+1.05% MAE
    # v5根因分析: PPO更新改变图策略后, 主模型已学习的特征与新图不匹配
    #   证据: Exp4 +RL epoch25 MAE=1.8874(最佳) → epoch28 MAE=1.9857(PPO更新后飙升)
    # v5修复策略:
    #   1. 降低RL学习率(3e-4→1e-4): 减小每次PPO更新的策略变化幅度
    #   2. 延长warmup(20→30): 确保主模型充分收敛后再启用RL
    #   3. 降低更新频率(5→10): 减少PPO更新次数, 给主模型更多适应时间
    #   4. 收紧PPO clip(0.2→0.1): 限制策略比率变化范围, 防止突变
    #   5. 新增梯度冻结: PPO更新epoch冻结主模型梯度, 避免反向传播干扰
    #   6. 增大mini-batch(64→128): 梯度更稳定, 更新次数减半(190→95), 减少策略漂移
    use_rl = True                # 是否启用 RL 图控制器
    rl_state_dim = 0             # 由代码自动计算，此处为占位
    rl_hidden_dim = 64           # RL 网络隐藏层维度
    rl_lr = 1e-4                 # [v5改] RL学习率 3e-4->1e-4, 减小策略变化幅度
    rl_gamma = 0.99              # 折扣因子
    rl_gae_lambda = 0.95         # GAE lambda
    rl_clip_epsilon = 0.1        # [v5改] PPO clip范围 0.2->0.1, 限制策略突变
    rl_entropy_coef = 0.02       # [v3改] 熵正则化 0.01->0.02, 鼓励更多探索避免早熟
    rl_value_coef = 0.5          # 价值损失系数
    rl_stability_weight = 0.05   # 动态图稳定性惩罚权重
    rl_update_freq_weight = 0.1  # 更新频率离散动作的权重
    rl_update_ratio_weight = 0.8 # 更新比例连续动作的权重
    rl_rollout_steps = 150       # [v5改] RL更新前最小经验数 30->150, 确保buffer多样性足够支撑128批量
    rl_ppo_epochs = 5            # [v3改] PPO每次更新的epoch数 10->5, 防止小buffer过拟合
    rl_mini_batch_size = 128     # [v5改] PPO mini-batch 64->128, 梯度更稳定+更新次数减半减少策略漂移
    rl_warmup_epochs = 30        # [v5改] RL预热阶段 20->30, 确保主模型充分收敛
    rl_alternate_interval = 10   # [v5改] 交替训练间隔 5->10, 降低PPO更新频率
    rl_freeze_main_model = True  # [v5新增] PPO更新epoch冻结主模型梯度, 避免图策略变化干扰特征学习

    # --- RL 奖励函数改进 ---
    rl_reward_type = 'relative'  # [P1新增] 奖励类型: 'absolute'(旧) 或 'relative'(新)
    rl_reward_baseline_alpha = 0.9  # [P1新增] 相对奖励的EMA基线平滑系数
    rl_reward_normalize = True   # [P1新增] 是否在PPO更新前对reward做z-score标准化
    rl_diversity_bonus = 0.02    # [P1新增] 图多样性奖励系数, 鼓励探索不同图结构
    rl_oscillation_penalty = 0.03  # [P1新增] 图震荡惩罚, 防止图频繁大幅变化

    # --- RL 动作空间 ---
    # v2: [0.3, 0.7] 太窄, 限制了有用的图更新
    # v3: [0.1, 0.9] 放宽范围, 允许更灵活的图融合比例
    rl_action_cont_min = 0.1     # [v3改] 连续动作下限 0.3->0.1, 允许更保守的更新
    rl_action_cont_max = 0.9     # [v3改] 连续动作上限 0.7->0.9, 允许更激进的更新

    # --- RL 课程学习 ---
    # v3: 关闭课程学习, 实验显示在307节点全图上直接训练效果更好
    rl_curriculum_enabled = False # [v3改] 关闭课程学习, 小图预训练策略迁移效果不佳
    rl_curriculum_pretrain_epochs = 20  # [P1新增] 小图预训练轮数
    rl_curriculum_sub_nodes = 50   # [P1新增] 预训练使用的子集节点数
    rl_curriculum_sub_timesteps = 1000  # [P1新增] 预训练使用的子集时间步数

    # ========== 多模态 ==========
    # v3: 默认关闭, v2实验显示Weather导致MAE上升1.6-3.9% (全零占位符数据)
    # 等高德真实天气数据采集完成后(7-14天)再启用
    use_multimodal = False       # [v3保持关闭] 等真实天气数据采集完成
    weather_dim = 6              # 天气特征维度 (温度/湿度/风速/气压/降水/能见度)
    weather_hidden = 32          # 天气编码器隐藏维度
    fusion_gate_hidden = 32      # 融合门控隐藏维度

    # ========== 高德数据采集配置 (v5: 多Key轮转) ==========
    gaode_api_key = 'd9994b6b8afd8a56a45511d800b7c361'           # 高德开放平台API Key (单Key模式)
    gaode_city = 'beijing'       # 目标城市
    # 注意: 矩形对角线不能超过10km, 否则高德API返回INVALID_PARAMS
    gaode_rectangle = '116.28,39.95;116.36,40.01'  # 北京中关村区域 (对角线约9.5km)
    gaode_collect_interval = 300  # 采集间隔 (秒), 300s = 5min, 与PEMS04频率一致
    gaode_collect_duration_days = 14  # 采集持续天数

    # [v5新增] 多Key分时段轮转配置
    # 单Key配额80次/天仅能采集6.7h, 4Key各6h可覆盖全天288次
    # 注册4个高德账户, 每个账户创建1个Key, 填入下方列表
    # 环境变量方式(推荐): GAODE_API_KEY_1 ~ GAODE_API_KEY_4
    gaode_api_keys = [
        '',  # Key1: 00:00-06:00 (凌晨) - 需填入或用环境变量
        '',  # Key2: 06:00-12:00 (上午)
        '',  # Key3: 12:00-18:00 (下午)
        '',  # Key4: 18:00-24:00 (晚间)
    ]

    # 和风天气 API 配置
    # 注意: 2026年起 devapi.qweather.com 已停用, 必须使用控制台中的专属 API Host
    weather_api_key = '324c2a58f0a84eb683d0122885fad16a'         # 和风天气API Key
    weather_api_host = 'pj3yfucjuk.re.qweatherapi.com'  # 专属API Host (从控制台获取)
    weather_api_url = 'https://pj3yfucjuk.re.qweatherapi.com/v7/weather/now'
    weather_location = '101010100'  # 和风天气城市ID (101010100=北京)

    # ========== 训练 (v5改进) ==========
    # v3问题: 100 epoch + patience 15 导致复杂变体(+RL+Topo)过早停止
    # v5修复: epochs 100->150(给RL充分收敛时间), patience 15->20(避免误判收敛)
    # 注意: run_experiments.py的--epochs默认值也需同步修改为150
    batch_size = 64              # [v5改] 批大小 48->64, 提升GPU利用率 (RTX 4090 24GB单实验可容纳)
    lr = 1e-3                    # 主模型学习率
    weight_decay = 1e-4
    epochs = 150                 # [v5改] 最大训练轮数 100->150, 给复杂变体更多收敛时间
    grad_clip = 5.0              # 梯度裁剪
    patience = 20                # [v5改] 早停耐心值 15->20, 避免RL波动导致的误判早停

    # ========== 设备 ==========
    device = 'cpu'
    seed = 42

    # ========== 日志 ==========
    log_interval = 10            # 每N个batch打印一次日志
    save_dir = 'checkpoints'     # 模型保存目录
    log_dir = 'logs'             # 日志目录
