import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .model import GCN_QNetwork # 确保您在 model.py 里定义了 GCN_QNetwork

class PS_DQNAgent:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, gcn_output_dim, lr, 
                 gamma, epsilon_start, epsilon_min, epsilon_decay, memory_size, 
                 batch_size, device):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        self.device = torch.device(device)
        
        # --- 修改点: 传递更多参数给 GCN_QNetwork ---
        self.policy_net = GCN_QNetwork(input_dim, hidden_dim1, hidden_dim2, gcn_output_dim).to(self.device)
        self.target_net = GCN_QNetwork(input_dim, hidden_dim1, hidden_dim2, gcn_output_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size) 
        self.batch_size = batch_size

    def select_actions(self, node_features, adj):
        """
        接收特征和邻接矩阵
        """
        num_nodes = node_features.shape[0]
        
        if random.random() < self.epsilon:
            return np.random.randint(0, 2, size=num_nodes)
        
        with torch.no_grad():
            # 增加 Batch 维度: [1, N, F]
            x = torch.FloatTensor(node_features).unsqueeze(0).to(self.device)
            a = torch.FloatTensor(adj).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(x, a) # [1, N, 2]
            actions = q_values.argmax(dim=2) # [1, N]
            return actions.cpu().numpy()[0]

    def store_transition(self, feat, adj, action, reward, next_feat, next_adj, done):
        """
        存储整个 Mesh 的快照
        """
        self.memory.append((feat, adj, action, reward, next_feat, next_adj, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样 1 个 Mesh 的经验
        batch = random.sample(self.memory, self.batch_size)
        feat, adj, action, reward, next_feat, next_adj, done = batch[0]
        
        # 转为 Tensor (增加 Batch 维度)
        feat = torch.FloatTensor(feat).unsqueeze(0).to(self.device)      # [1, N, F]
        adj = torch.FloatTensor(adj).unsqueeze(0).to(self.device)        # [1, N, N]
        action = torch.LongTensor(action).unsqueeze(0).to(self.device)   # [1, N]
        reward = torch.FloatTensor([reward]).to(self.device)             # [1] (标量)
        next_feat = torch.FloatTensor(next_feat).unsqueeze(0).to(self.device)
        next_adj = torch.FloatTensor(next_adj).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        # 计算 Q_eval: [1, N, 2] -> gather -> [1, N]
        q_eval = self.policy_net(feat, adj).gather(2, action.unsqueeze(2)).squeeze(2)
        
        # 计算 Q_target
        with torch.no_grad():
            q_next = self.target_net(next_feat, next_adj).max(dim=2)[0] # [1, N]
            # 广播 Reward: [1] + [1, N] -> [1, N]
            q_target = reward + (1 - done) * self.gamma * q_next
        
        # 计算 Loss (对 N 个基站求平均)+
        loss = F.mse_loss(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)