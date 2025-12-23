import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN_QNetwork(nn.Module):
    def __init__(self, input_dim=16, hidden_dim1=64, hidden_dim2=128, gcn_output_dim=16):
        """
        input_dim: 原始特征维度 (例如 5: 1 Load + 4 OneHot Type)
        hidden_dim: GCN 中间层维度
        gcn_output_dim: GCN 最终输出的向量长度 (你要求的 16)
        """
        super(GCN_QNetwork, self).__init__()
        
        # --- Part 1: GCN 层 (用于提取 Network Context) ---
        # 输入: [Batch, N, input_dim] -> 输出: [Batch, N, 16]
        self.gcn0 = nn.Linear(input_dim, hidden_dim1)
        self.gcn1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.gcn2 = nn.Linear(hidden_dim2, gcn_output_dim) # 输出你指定的 16 维向量
        
        # --- Part 2: Q-Value 输出层 (MLP) ---
        # 输入维度 = GCN提取的全局特征(16) + 本节点原始特征(input_dim)
        self.concat_dim = gcn_output_dim + input_dim
        
        self.fc1 = nn.Linear(self.concat_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 2)

    def forward(self, x, adj):
        """
        x:   [Batch, N, Input_Dim]  (包含: 本节点流量 + 本节点类型)
        adj: [Batch, N, N]          (邻接矩阵)
        """
        # 0. 保留原始输入用于后续拼接 (Skip Connection)
        x_input = x 
        
        # --- 预处理: 邻接矩阵归一化 ---
        degree = torch.sum(adj, dim=2, keepdim=True).clamp(min=1.0)
        adj_norm = adj / degree
        
        # --- Part 1: GCN ---
        # Layer 0: 先通过 gcn0 变换维度，再聚合
        x_gcn = self.gcn0(x)             # Linear
        x_gcn = torch.matmul(adj_norm, x_gcn) # Graph Conv
        x_gcn = F.relu(x_gcn)            # Activation
        
        # Layer 1
        x_gcn = self.gcn1(x_gcn)
        x_gcn = torch.matmul(adj_norm, x_gcn)
        x_gcn = F.relu(x_gcn)
        
        # Layer 2 (输出 16维)# [Batch, N, 16]
        x_gcn = self.gcn2(x_gcn)
        x_gcn = torch.matmul(adj_norm, x_gcn)
        x_gcn = F.relu(x_gcn) 
        
        # --- Part 2: 特征拼接 ---
        # 拼接 [GCN全局特征, 本节点原始特征]
        x_combined = torch.cat([x_input, x_gcn], dim=-1)
        
        # --- Part 3: MLP ---
        # Layer 1
        x_out = F.relu(self.fc1(x_combined))
        # Layer 2
        x_out = F.relu(self.fc2(x_out))
        # Layer 3 (输出 Q 值，通常最后一层不加 ReLU，除非你需要 Q 值非负)
        q_values = self.fc3(x_out)
        
        return q_values