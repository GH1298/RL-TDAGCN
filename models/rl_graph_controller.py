"""
PPO 强化学习图控制器 (RL Graph Controller)
这是 RL-TDAGCN 的核心创新模块。

功能:
  RL不直接预测交通，而是控制动态图的更新策略。
  在 Graph Generator 生成候选图 A_candidate 后，
  RL Agent 决定如何将 A_candidate 与 A_old 融合为 A_optimized。

动作空间（混合动作空间）:
  - a1 (离散): 更新频率 (0=不更新, 1=普通更新, 2=快速更新)
  - a2 (连续): 更新比例 alpha in [0, 1]
    A_new = (1 - alpha) * A_old + alpha * A_candidate

状态空间:
  s_t = [X_t_stats, e_t, delta_A]
  - X_t_stats: 当前交通状态的统计摘要 (均值, 标准差)
  - e_t: 上一时刻的预测误差
  - delta_A: 图变化量 ||A_t - A_{t-1}||

奖励:
  r_t = -(MAE_t + lambda * ||A_t - A_{t-1}||)
  目标: 预测准确 + 图结构稳定

算法: PPO (Proximal Policy Optimization)
  - 离散动作: Categorical分布
  - 连续动作: Normal分布
  - GAE (Generalized Advantage Estimation) 计算优势
  - Clip 机制防止策略更新过大
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np


class RolloutBuffer:
    """
    经验回放缓冲区
    存储RL Agent与环境交互的轨迹数据，用于PPO批量更新

    存储内容:
      - states: 状态
      - actions: 动作 (离散+连续)
      - log_probs: 动作对数概率
      - rewards: 奖励
      - values: 状态价值估计
      - dones: 是否结束标志
    """
    def __init__(self):
        self.clear()

    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions_discrete = []
        self.actions_continuous = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.pred_errors = []  # 额外记录预测误差用于分析

    def push(self, state, action_disc, action_cont, log_prob, reward, value, done, pred_error=0.0):
        """存储一步经验"""
        self.states.append(state)
        self.actions_discrete.append(action_disc)
        self.actions_continuous.append(action_cont)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.pred_errors.append(pred_error)

    def __len__(self):
        return len(self.states)

    def get_tensors(self, device='cpu'):
        """将所有数据转为tensor"""
        return {
            'states': torch.stack(self.states).to(device),
            'actions_disc': torch.stack(self.actions_discrete).to(device),
            'actions_cont': torch.stack(self.actions_continuous).to(device),
            'log_probs': torch.stack(self.log_probs).to(device),
            'rewards': torch.FloatTensor(self.rewards).to(device),
            'values': torch.stack(self.values).to(device),
            'dones': torch.FloatTensor(self.dones).to(device),
        }


class PPOGraphController(nn.Module):
    """
    PPO 强化学习图控制器

    使用 Actor-Critic 架构，混合动作空间:
    - Actor(离散): 输出更新频率的 Categorical 分布
    - Actor(连续): 输出更新比例的 Normal 分布
    - Critic: 输出状态价值 V(s)

    Args:
        state_dim: 状态维度（自动计算）
        hidden_dim: 隐藏层维度
        num_freq_actions: 离散动作数（更新频率: 3 = 不更新/普通/快速）
        action_bound: 连续动作范围 [0, action_bound]
        clip_epsilon: PPO clip 范围
        entropy_coef: 熵正则化系数
        value_coef: 价值损失系数
        gae_lambda: GAE lambda 参数
        gamma: 折扣因子
        action_cont_min: [P1新增] 连续动作下限，默认0.0（向后兼容旧版[0,1]范围）
        action_cont_max: [P1新增] 连续动作上限，默认1.0（向后兼容旧版[0,1]范围）
        normalize_reward: [P1新增] 是否在PPO更新前对reward做z-score标准化，默认False
    """
    def __init__(self, state_dim, hidden_dim=64, num_freq_actions=3,
                 clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5,
                 gae_lambda=0.95, gamma=0.99,
                 action_cont_min=0.0, action_cont_max=1.0,
                 normalize_reward=False):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_freq_actions = num_freq_actions
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        # === [P1新增] 连续动作空间收窄 ===
        # 将连续动作 alpha 的采样范围从 [0, 1] 收窄到 [action_cont_min, action_cont_max]
        # 目的: 减少无效探索空间，加速PPO在大规模图(307节点)上的收敛
        # 默认值 0.0/1.0 保证不传参时行为与旧版完全一致
        self.action_cont_min = action_cont_min
        self.action_cont_max = action_cont_max

        # === [P1新增] 奖励标准化开关 ===
        # 开启后在 ppo_update 中对 rewards 做 z-score 标准化，消除不同 rollout 间奖励尺度差异
        self.normalize_reward = normalize_reward

        # === [v3新增] 奖励裁剪 ===
        # 将奖励限制在 [-10, 0] 范围内，防止极端值（如MAE突然飙升）导致PPO策略崩溃
        # v2问题: 某些batch的MAE异常大，导致reward极端负值，PPO梯度爆炸
        self.reward_clip_min = -10.0
        self.reward_clip_max = 0.0

        # === [v3新增] 移动平均奖励基线 ===
        # 维护一个EMA形式的奖励基线，用于计算相对奖励
        # 相对奖励 = actual_reward - baseline，消除全局奖励趋势的影响
        # v2问题: 绝对奖励波动大，PPO难以区分"好的动作"和"好的趋势"
        self.reward_baseline = 0.0
        self.baseline_alpha = 0.95  # EMA平滑系数，越大越保守

        # === 共享特征编码器 ===
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # === Actor: 离散分支（更新频率）===
        self.actor_disc = nn.Linear(hidden_dim, num_freq_actions)

        # === Actor: 连续分支（更新比例 alpha）===
        self.actor_cont_mean = nn.Linear(hidden_dim, 1)
        self.actor_cont_log_std = nn.Parameter(torch.zeros(1))  # 可学习的log标准差

        # === Critic ===
        self.critic = nn.Linear(hidden_dim, 1)

    def apply_config(self, config):
        """
        [P1新增] 从配置对象读取RL控制器相关参数

        设计原因:
          rlt_dagcn.py 在构造 PPOGraphController 时未传入新增参数
          (action_cont_min/max, normalize_reward)，通过此方法可在
          不修改 rlt_dagcn.py 的前提下，从 config 激活这些新特性。

        用法:
          model.rl_controller.apply_config(config)

        Args:
            config: 配置对象 (configs/default.py 中的 Config 实例)
        """
        self.action_cont_min = getattr(config, 'rl_action_cont_min', 0.0)
        self.action_cont_max = getattr(config, 'rl_action_cont_max', 1.0)
        self.normalize_reward = getattr(config, 'rl_reward_normalize', False)

    def _encode_state(self, state):
        """共享特征编码"""
        return self.shared(state)

    def _get_cont_std(self):
        """获取连续动作的标准差（限制范围保证稳定）"""
        return torch.clamp(torch.exp(self.actor_cont_log_std), min=0.01, max=1.0)

    def select_action(self, state, deterministic=False):
        """
        选择动作（训练时采样，推理时取均值）

        Args:
            state: [B, state_dim] 状态向量
            deterministic: 是否使用确定性策略

        Returns:
            action_disc: [B] 离散动作（更新频率）
            action_cont: [B, 1] 连续动作（更新比例 alpha in [0,1]）
            log_prob: [B] 总对数概率
            value: [B, 1] 状态价值
        """
        features = self._encode_state(state)

        # --- 离散动作: 更新频率 ---
        disc_logits = self.actor_disc(features)  # [B, num_freq_actions]
        disc_dist = Categorical(logits=disc_logits)

        if deterministic:
            action_disc = disc_logits.argmax(dim=-1)  # [B]
        else:
            action_disc = disc_dist.sample()  # [B]

        disc_log_prob = disc_dist.log_prob(action_disc)  # [B]
        disc_entropy = disc_dist.entropy()  # [B]

        # --- 连续动作: 更新比例 alpha ---
        # [P1改] 动作范围从 [0,1] 收窄到 [action_cont_min, action_cont_max]
        # 原理: sigmoid 输出 [0,1] -> 线性映射到 [min, max] 目标区间
        # 效果: 减少无效探索空间(如alpha接近0或1的极端值)，加速PPO在大规模图上收敛
        cont_mean = self.action_cont_min + (self.action_cont_max - self.action_cont_min) \
            * torch.sigmoid(self.actor_cont_mean(features))  # [B, 1], 映射到目标区间
        cont_std = self._get_cont_std()

        if deterministic:
            action_cont = cont_mean
            cont_log_prob = torch.zeros(state.size(0), device=state.device)
            cont_entropy = torch.zeros(state.size(0), device=state.device)
        else:
            cont_dist = Normal(cont_mean, cont_std)
            action_cont = cont_dist.sample()  # [B, 1]
            # [P1改] clamp 范围同步收窄，确保采样值不超出目标区间
            action_cont = torch.clamp(action_cont, self.action_cont_min, self.action_cont_max)
            cont_log_prob = cont_dist.log_prob(action_cont).squeeze(-1)  # [B]
            cont_entropy = cont_dist.entropy().squeeze(-1)  # [B]

        # --- Critic ---
        value = self.critic(features)  # [B, 1]

        # 总对数概率 = 离散 + 连续
        total_log_prob = disc_log_prob + cont_log_prob
        total_entropy = disc_entropy + cont_entropy

        return action_disc, action_cont, total_log_prob, value, total_entropy

    def evaluate(self, state, action_disc, action_cont):
        """
        评估已有动作的对数概率和价值（PPO更新时使用）

        Args:
            state: [B, state_dim]
            action_disc: [B] 离散动作
            action_cont: [B, 1] 连续动作

        Returns:
            log_prob: [B] 对数概率
            value: [B, 1] 状态价值
            entropy: [B] 策略熵
        """
        features = self._encode_state(state)

        # 离散动作评估
        disc_logits = self.actor_disc(features)
        disc_dist = Categorical(logits=disc_logits)
        disc_log_prob = disc_dist.log_prob(action_disc)
        disc_entropy = disc_dist.entropy()

        # 连续动作评估
        # [P1改] 与 select_action 保持一致的动作范围映射
        # 必须使用相同的 [min, max] 映射，否则 PPO 的 ratio 计算会产生偏差
        cont_mean = self.action_cont_min + (self.action_cont_max - self.action_cont_min) \
            * torch.sigmoid(self.actor_cont_mean(features))
        cont_std = self._get_cont_std()
        cont_dist = Normal(cont_mean, cont_std)
        cont_log_prob = cont_dist.log_prob(action_cont).squeeze(-1)
        cont_entropy = cont_dist.entropy().squeeze(-1)

        # 合并
        total_log_prob = disc_log_prob + cont_log_prob
        total_entropy = disc_entropy + cont_entropy
        value = self.critic(features)

        return total_log_prob, value, total_entropy

    def compute_gae(self, rewards, values, dones, next_value=None):
        """
        计算 GAE (Generalized Advantage Estimation)

        Args:
            rewards: [T] 奖励序列
            values: [T] 价值序列
            dones: [T] 结束标志
            next_value: [1] 下一状态的初始价值

        Returns:
            advantages: [T] 优势估计
            returns: [T] 回报估计
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=rewards.device)
        returns = torch.zeros(T, device=rewards.device)

        if next_value is not None:
            last_gae = next_value.item()
        else:
            last_gae = 0.0

        # 逆向计算GAE
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value if next_value is not None else torch.tensor(0.0)
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = last_gae + values[t]

        return advantages, returns

    def ppo_update(self, buffer, optimizer, mini_batch_size=32, ppo_epochs=4):
        """
        PPO 更新

        v3改进:
          1. 奖励裁剪: 防止极端reward导致梯度爆炸
          2. 移动平均基线: 减去EMA基线计算相对奖励，减少全局趋势干扰
          3. 优势标准化已有(v2)，但配合更大的mini_batch(64)效果更好

        Args:
            buffer: RolloutBuffer 经验缓冲区
            optimizer: PPO优化器
            mini_batch_size: mini-batch大小
            ppo_epochs: PPO更新轮数

        Returns:
            dict: 包含各项损失的信息
        """
        data = buffer.get_tensors(device=next(self.parameters()).device)
        states = data['states']
        actions_disc = data['actions_disc']
        actions_cont = data['actions_cont']
        old_log_probs = data['log_probs']
        rewards = data['rewards']
        old_values = data['values']
        dones = data['dones']

        # === [v3新增] 奖励裁剪 ===
        # 将奖励限制在合理范围内，防止极端值破坏PPO训练
        rewards = torch.clamp(rewards, self.reward_clip_min, self.reward_clip_max)

        # === [v3新增] 移动平均基线减法 ===
        # 更新EMA基线并计算相对奖励
        # 相对奖励 = reward - baseline，使PPO关注"比平均水平好/差"的动作
        current_mean_reward = rewards.mean().item()
        self.reward_baseline = (self.baseline_alpha * self.reward_baseline
                                + (1 - self.baseline_alpha) * current_mean_reward)
        rewards = rewards - self.reward_baseline

        # 计算GAE
        with torch.no_grad():
            # [P1新增] 奖励标准化 (z-score)
            # 在计算GAE之前对 rewards 做标准化，消除不同 rollout 间奖励尺度差异
            # 这对大规模图(307节点)尤为重要: 不同batch的预测误差波动范围差异大，
            # 不做标准化会导致GAE优势估计被个别极端奖励主导，PPO更新方向不稳定
            if self.normalize_reward:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            _, _, _, last_value, _ = self.select_action(states[-1:], deterministic=True)
            advantages, returns = self.compute_gae(rewards, old_values.squeeze(-1), dones, last_value.squeeze(-1))

            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        dataset_size = states.size(0)

        for _ in range(ppo_epochs):
            # 随机排列
            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, mini_batch_size):
                end = min(start + mini_batch_size, dataset_size)
                mb_indices = indices[start:end]

                mb_states = states[mb_indices]
                mb_actions_disc = actions_disc[mb_indices]
                mb_actions_cont = actions_cont[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # 评估当前策略
                new_log_probs, new_values, entropy = self.evaluate(
                    mb_states, mb_actions_disc, mb_actions_cont
                )

                # PPO Clip 策略损失
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = F.mse_loss(new_values.squeeze(-1), mb_returns)

                # 熵正则化（鼓励探索）
                entropy_bonus = entropy.mean()

                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
                num_updates += 1

        return {
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'value_loss': total_value_loss / max(num_updates, 1),
            'entropy': total_entropy / max(num_updates, 1),
        }