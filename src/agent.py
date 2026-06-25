import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from .model import GCN_QNetwork 
from .model import MLP_QNetwork

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
        
        self.policy_net = GCN_QNetwork(input_dim, hidden_dim1, hidden_dim2, gcn_output_dim).to(self.device)
        self.target_net = GCN_QNetwork(input_dim, hidden_dim1, hidden_dim2, gcn_output_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = deque(maxlen=memory_size) 

    def select_actions(self, node_features, adj):
        """
        根据接收的特征和邻接矩阵，做出action
        前期主要是随机动作，后期是根据当前的策略网络输出动作
        输出: 每个基站的动作 (0 或 1)，形状为 [N]，其中 N 是基站数量
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
        存储整个 Mesh 某一时刻的快照
        """
        self.memory.append((feat, adj, action, reward, next_feat, next_adj, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样一批经验
        batch = random.sample(self.memory, self.batch_size)
        
        feat_list = []
        action_list = []
        reward_list = []
        next_feat_list = []
        done_list = []

        # 先计算所有图的节点总数，用于初始化大矩阵
        total_nodes = sum(transition[0].shape[0] for transition in batch)

        # 1. 核心加速技巧: 构建分块对角邻接矩阵 (Block Diagonal Matrix)
        batch_adj = torch.zeros((total_nodes, total_nodes), dtype=torch.float32)
        batch_next_adj = torch.zeros((total_nodes, total_nodes), dtype=torch.float32)

        current_idx = 0
        for feat, adj, action, reward, next_feat, next_adj, done in batch:
            n = feat.shape[0] # 当前 Mesh 的基站数
            
            # 填充邻接矩阵的对角块
            batch_adj[current_idx:current_idx+n, current_idx:current_idx+n] = torch.FloatTensor(adj)
            batch_next_adj[current_idx:current_idx+n, current_idx:current_idx+n] = torch.FloatTensor(next_adj)
            current_idx += n
            
            # 收集其他特征
            feat_list.append(torch.FloatTensor(feat))
            next_feat_list.append(torch.FloatTensor(next_feat))
            action_list.append(torch.LongTensor(action))
            
            # 由于整个网格只有一个共享的 Reward 和 Done，需要将其复制(Broadcast)到每个节点
            reward_list.append(torch.full((n,), reward, dtype=torch.float32))
            done_list.append(torch.full((n,), done, dtype=torch.float32))

        # 2. 拼接节点维度 (dim=0) -> [Sum(N), ...]
        # unsqueeze(0) 是为了满足模型输入格式 [Batch=1, N_all, Features]
        batch_feat = torch.cat(feat_list, dim=0).unsqueeze(0).to(self.device)
        batch_next_feat = torch.cat(next_feat_list, dim=0).unsqueeze(0).to(self.device)
        batch_action = torch.cat(action_list, dim=0).unsqueeze(0).to(self.device)
        batch_reward = torch.cat(reward_list, dim=0).unsqueeze(0).to(self.device)
        batch_done = torch.cat(done_list, dim=0).unsqueeze(0).to(self.device)
        
        batch_adj = batch_adj.unsqueeze(0).to(self.device)
        batch_next_adj = batch_next_adj.unsqueeze(0).to(self.device)

        # 3. 完全向量化的前向与反向传播
        # 计算 Q_eval: [1, Sum(N), 2] -> gather -> [1, Sum(N)]
        q_eval = self.policy_net(batch_feat, batch_adj).gather(2, batch_action.unsqueeze(2)).squeeze(2)
        
        with torch.no_grad():
            # 计算 Q_next: [1, Sum(N), 2] -> max -> [1, Sum(N)]
            q_next = self.target_net(batch_next_feat, batch_next_adj).max(dim=2)[0]
            # 计算目标 Q 值
            q_target = batch_reward + (1 - batch_done) * self.gamma * q_next
        
        # 计算 Loss (均方误差 MSE)
        loss = F.mse_loss(q_eval, q_target)
        
        # 梯度下降更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class PS_MLPAgent(PS_DQNAgent):
    """消融实验 1：继承原智能体，仅将核心网络替换为 MLP"""
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, lr, 
                 gamma, epsilon_start, epsilon_min, epsilon_decay, memory_size, 
                 batch_size, device):
        # 调用父类初始化（随便传个gcn_output_dim=16，反正不用它）
        super().__init__(input_dim, hidden_dim1, hidden_dim2, 16, lr, 
                         gamma, epsilon_start, epsilon_min, epsilon_decay, memory_size, 
                         batch_size, device)
        
        # 【核心】：用 MLP 覆盖掉父类的 GCN
        self.policy_net = MLP_QNetwork(input_dim, hidden_dim1, hidden_dim2).to(self.device)
        self.target_net = MLP_QNetwork(input_dim, hidden_dim1, hidden_dim2).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)