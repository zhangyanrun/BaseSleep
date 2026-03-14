import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy

# ==========================================
# 1. 核心网络定义 (严格对应论文 Table III)
# ==========================================
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        # 论文使用 800 -> 600 的隐藏层
        # 注意：为了适配单步决策(Batch=1)，这里使用 LayerNorm 替代 BatchNorm 避免崩溃
        self.fc1 = nn.Linear(state_dim, 800)
        self.ln1 = nn.LayerNorm(800)
        self.fc2 = nn.Linear(800, 600)
        self.ln2 = nn.LayerNorm(600)
        self.fc3 = nn.Linear(600, action_dim)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        # 输出 0~1 的连续概率
        action_prob = torch.sigmoid(self.fc3(x))
        return action_prob

class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 800)
        self.ln1 = nn.LayerNorm(800)
        self.fc2 = nn.Linear(800, 600)
        self.ln2 = nn.LayerNorm(600)
        self.fc3 = nn.Linear(600, 1)

    def forward(self, state, action):
        # Critic 将 State 和 Action 拼接作为输入
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x) # Linear 输出
        return q_value

# ==========================================
# 2. DeepBSC 智能体 (包含 Explorer Network 机制)
# ==========================================
class DeepBSCAgent:
    def __init__(self, num_bs, lr_actor=1e-4, lr_critic=1e-3, gamma=0.9, tau=0.005, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 论文定义状态为：预测流量 + 上一时刻的动作
        self.num_bs = num_bs
        self.state_dim = num_bs * 2  # N 个基站的流量 + N 个基站的上一动作
        self.action_dim = num_bs
        self.gamma = gamma
        self.tau = tau
        
        # 探索者网络超参数
        self.explore_alpha = 0.1 
        self.explore_sigma = 0.05
        
        # 初始化 Actor 和 Critic
        self.actor = ActorNet(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = CriticNet(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.explorer_actor = ActorNet(self.state_dim, self.action_dim).to(self.device)

    def _get_explorer_actor(self):
        """生成探索者网络 (Explorer Network)"""
        # 【修改】：不再使用 copy.deepcopy，而是使用极速的权重加载
        self.explorer_actor.load_state_dict(self.actor.state_dict())
        
        with torch.no_grad():
            for param in self.explorer_actor.parameters():
                # 注入扰动噪声：W' = W + alpha * rand(-1,1) * W
                noise = self.explore_alpha * (torch.rand_like(param) * 2 - 1) * param
                param.add_(noise)
                
        return self.explorer_actor
        

    def select_actions(self, node_features, prev_actions):
        """
        与环境交互：利用 Explorer Network 机制选择最优动作
        """
        # 从特征中提取预测流量 (Load_t+1)，也就是你的 node_features[:, 1]
        predicted_load = node_features[:, 1]
        
        # 拼装 State
        state_np = np.concatenate([predicted_load, prev_actions])
        state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            # 1. 主 Actor 给出动作 a_a
            action_a_prob = self.actor(state_tensor)
            
            # 2. Explorer Network 给出动作 a_e
            explorer_actor = self._get_explorer_actor()
            action_e_prob = explorer_actor(state_tensor)
            
            # 3. 让 Critic 评价谁更好 (Q 值越大越好，论文里是 Cost 越小越好，这里对应 Reward 取大)
            q_a = self.critic(state_tensor, action_a_prob)
            q_e = self.critic(state_tensor, action_e_prob)
            
            if q_e.item() > q_a.item():
                # 如果探索者更好，软更新主网络
                action_prob = action_e_prob
                for param, target_param in zip(explorer_actor.parameters(), self.actor.parameters()):
                    target_param.data.copy_(target_param.data + self.explore_sigma * (param.data - target_param.data))
            else:
                action_prob = action_a_prob
                
        self.actor.train()
        self.critic.train()
        
        # 将概率映射为 0/1 离散动作
        action_np = action_prob.cpu().numpy()[0]
        final_action = np.where(action_np > 0.5, 1, 0)
        return final_action, state_np, action_np # 返回连续动作给 Buffer 用于 DDPG 训练

    def train_step(self, replay_buffer, batch_size=64):
        """DDPG 的标准训练步骤"""
        if len(replay_buffer) < batch_size:
            return
            
        # 从 Buffer 采样 (这里假设你实现了一个简单的 List Buffer)
        batch = replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(batch['states']).to(self.device)
        action_batch = torch.FloatTensor(batch['actions']).to(self.device)
        reward_batch = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch['next_states']).to(self.device)
        
        # ==========================================
        # 1. 更新 Critic
        # ==========================================
        with torch.no_grad():
            next_action = self.actor_target(next_state_batch)
            target_q = reward_batch + self.gamma * self.critic_target(next_state_batch, next_action)
            
        current_q = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ==========================================
        # 2. 更新 Actor
        # ==========================================
        # 策略梯度：使得 Critic 评估的 Q 值最大化
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ==========================================
        # 3. 软更新 Target 网络
        # ==========================================
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = {
            'states': state,
            'actions': action,
            'rewards': reward,
            'next_states': next_state
        }
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return {
            'states': np.array([transition['states'] for transition in batch]),
            'actions': np.array([transition['actions'] for transition in batch]),
            'rewards': np.array([transition['rewards'] for transition in batch]),
            'next_states': np.array([transition['next_states'] for transition in batch])
        }

    def __len__(self):
        return len(self.buffer)