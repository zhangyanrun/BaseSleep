import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

# ==========================================
# 1. 论文核心: 两个独立的 Q 网络 (采用论文设定的 2层 MLP)
# ==========================================
class SleepNet(nn.Module):
    def __init__(self, state_dim, num_bs):
        super(SleepNet, self).__init__()
        self.num_bs = num_bs
        # 论文设定隐含层为 200, 200
        self.fc1 = nn.Linear(state_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        # 输出每个基站 2 个动作的 Q 值 (Sleep or Active)
        self.fc3 = nn.Linear(200, num_bs * 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out.view(-1, self.num_bs, 2) # [Batch, N, 2]

class PowerNet(nn.Module):
    def __init__(self, state_dim, num_bs, power_levels=5):
        super(PowerNet, self).__init__()
        self.num_bs = num_bs
        self.power_levels = power_levels
        # PowerNet 的状态输入是：环境 State + SleepNet 的动作结果
        self.fc1 = nn.Linear(state_dim + num_bs, 200)
        self.fc2 = nn.Linear(200, 200)
        # 输出每个基站 K 个功率档位的 Q 值
        self.fc3 = nn.Linear(200, num_bs * power_levels)

    def forward(self, state, sleep_actions):
        # 将 State 和 休眠动作 拼接
        x = torch.cat([state, sleep_actions.float()], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out.view(-1, self.num_bs, self.power_levels) # [Batch, N, K]

# ==========================================
# 2. Two-DQNS 级联智能体
# ==========================================
class TwoDQNSAgent:
    def __init__(self, num_bs, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, power_levels=5, memory_size=10000, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.num_bs = num_bs
        self.state_dim = num_bs * 2 # 预测流量 + 上一时刻动作
        self.power_levels = power_levels
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # 实例化 休眠控制 DQN-1
        self.sleep_net = SleepNet(self.state_dim, num_bs).to(self.device)
        self.sleep_target = SleepNet(self.state_dim, num_bs).to(self.device)
        self.sleep_target.load_state_dict(self.sleep_net.state_dict())
        self.sleep_optimizer = optim.Adam(self.sleep_net.parameters(), lr=lr)
        
        # 实例化 功率控制 DQN-2
        self.power_net = PowerNet(self.state_dim, num_bs, power_levels).to(self.device)
        self.power_target = PowerNet(self.state_dim, num_bs, power_levels).to(self.device)
        self.power_target.load_state_dict(self.power_net.state_dict())
        self.power_optimizer = optim.Adam(self.power_net.parameters(), lr=lr)

        self.memory = deque(maxlen=memory_size)

    def select_actions(self, node_features, prev_actions):
        """级联动作选择"""
        predicted_load = node_features[:, 1]
        state_np = np.concatenate([predicted_load, prev_actions])
        state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
        
        sleep_actions = np.zeros(self.num_bs, dtype=int)
        power_actions = np.zeros(self.num_bs, dtype=int)
        
        # --- 1. DQN-1 决定休眠动作 ---
        if np.random.rand() <= self.epsilon:
            sleep_actions = np.random.randint(0, 2, self.num_bs)
        else:
            with torch.no_grad():
                q_sleep = self.sleep_net(state_tensor)
                sleep_actions = q_sleep.argmax(dim=2).squeeze(0).cpu().numpy()
                
        # --- 2. DQN-2 决定功率档位 (基于 DQN-1 的决定) ---
        sleep_tensor = torch.FloatTensor(sleep_actions).unsqueeze(0).to(self.device)
        if np.random.rand() <= self.epsilon:
            power_actions = np.random.randint(0, self.power_levels, self.num_bs)
        else:
            with torch.no_grad():
                q_power = self.power_net(state_tensor, sleep_tensor)
                power_actions = q_power.argmax(dim=2).squeeze(0).cpu().numpy()
                
        return sleep_actions, power_actions, state_np
        
    def train_step(self, batch_size=64):
        """双网络联合训练"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        state_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        sleep_action_batch = torch.LongTensor(np.array([x[1] for x in batch])).to(self.device)
        power_action_batch = torch.LongTensor(np.array([x[2] for x in batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([x[4] for x in batch])).to(self.device)
        
        # ------------------------------------
        # 更新 DQN-1 (休眠控制)
        # ------------------------------------
        q_sleep_eval = self.sleep_net(state_batch).gather(2, sleep_action_batch.unsqueeze(2)).squeeze(2)
        with torch.no_grad():
            q_sleep_next = self.sleep_target(next_state_batch).max(dim=2)[0]
            q_sleep_target = reward_batch.unsqueeze(1) + self.gamma * q_sleep_next
            
        loss_sleep = F.mse_loss(q_sleep_eval, q_sleep_target)
        self.sleep_optimizer.zero_grad()
        loss_sleep.backward()
        self.sleep_optimizer.step()
        
        # ------------------------------------
        # 更新 DQN-2 (功率控制)
        # ------------------------------------
        # 下一时刻的 sleep 动作需要通过 sleep_target 网络得到
        with torch.no_grad():
            next_sleep_actions = self.sleep_target(next_state_batch).argmax(dim=2)
            
        q_power_eval = self.power_net(state_batch, sleep_action_batch).gather(2, power_action_batch.unsqueeze(2)).squeeze(2)
        with torch.no_grad():
            q_power_next = self.power_target(next_state_batch, next_sleep_actions).max(dim=2)[0]
            q_power_target = reward_batch.unsqueeze(1) + self.gamma * q_power_next
            
        loss_power = F.mse_loss(q_power_eval, q_power_target)
        self.power_optimizer.zero_grad()
        loss_power.backward()
        self.power_optimizer.step()
        
        # 衰减 Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.sleep_target.load_state_dict(self.sleep_net.state_dict())
        self.power_target.load_state_dict(self.power_net.state_dict())