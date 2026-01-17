# ==========================================
# 1. 基站物理参数 (BS Physical Params)
# ==========================================
BS_CONFIG = {
    '4G_macro': {
        'capacity': 300.0,
        'radius_eff': 500.0, 
        'radius_max': 1500.0,
        'p_zero': 400.0,      # 800W * 0.6
        'slope': 200.0,       # 800W - 480W
        'p_sleep': 80.0
    },
    '4G_micro': {
        'capacity': 100.0,
        'radius_eff': 200.0,
        'radius_max': 300.0,
        'p_zero': 100.0,       # 150W * 0.6
        'slope': 60.0,        # 150W - 90W
        'p_sleep': 20.0
    },
    '5G_macro': {
        'capacity': 2000.0,
        'radius_eff': 300.0,
        'radius_max': 1000.0,
        'p_zero': 1000.0,      # 1200W * 0.6
        'slope': 500.0,       # 1200W - 720W
        'p_sleep': 150.0
    },
    '5G_micro': {
        'capacity': 800.0,
        'radius_eff': 100.0,
        'radius_max': 300.0,
        'p_zero': 180.0,      # 300W * 0.6
        'slope': 120.0,       # 300W - 180W
        'p_sleep': 40.0
    }
}

# 流量接收优先级 (分数越高越优先接盘)
PRIORITY_MAP = {
    '4G_macro': 4,
    '5G_macro': 3,
    '4G_micro': 2,
    '5G_micro': 1
}

# 基站类型索引 (用于生成 One-Hot 编码)
TYPE_TO_INDEX = {
    '4G_macro': 0,
    '5G_macro': 1,
    '4G_micro': 2,
    '5G_micro': 3
}

# ==========================================
# 2. 强化学习超参数 (RL Hyperparameters)
# ==========================================
RL_PARAMS = {
    # 状态维度: 1(Load) + 4(OneHot Type) = 5
    'input_dim': 5,
    
    # 隐藏层维度: 128 (GCN 和 MLP 共享)
    'hidden_dim1': 64,
    'hidden_dim2': 256,

    # GCN 最终输出的特征维度 (你要求的 16)
    'gcn_output_dim': 16,
    
    # 学习率 (Learning Rate)
    'lr': 1e-1,
    
    # 折扣因子 (Discount Factor)
    'gamma': 0.95,
    
    # 探索率 (Epsilon Greedy)
    'epsilon_start': 1.0,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.9998, # 减慢衰减速度
    
    # 经验回放池大小
    'memory_size': 10000,
    
    # 训练批次大小 (Batch Size)
    # 注意: PS-IDQN 每次训练一个 Mesh，这里指从 Buffer 中采样的 Mesh 数量
    'batch_size': 1 
}

# ==========================================
# 3. 奖励与惩罚系数 (Reward Coefficients)
# ==========================================
REWARD_PARAMS = {
    # 【修改】不再直接用能耗系数，而是用“节省能耗”的缩放系数
    'w_energy_saving': 1, 
    
    # 【新增】非线性 QoS 惩罚参数
    # 公式: alpha * (exp(beta * load) - 1)
    'qos_alpha': 0.3,   # 基础系数，控制整体惩罚幅度
    'qos_beta': 2,    # 指数系数，控制"陡峭"程度。beta=6时，Load=1.0 -> exp(6)≈403 (惩罚巨大)
    
    # 掉线依然是不可接受的，保留最严厉的线性惩罚
    'w_drop': 2.5,

    'global_scale': 0.00001
}

# ==========================================
# 4. 训练流程参数 (Training Flow)
# ==========================================
TRAIN_PARAMS = {
    'num_epochs': 200,      # 总共把所有 Mesh 轮询多少遍
    'log_interval': 20,    # 每训练多少个 Mesh 打印一次日志
    'target_update': 10,   # 每多少个 Mesh 更新一次目标网络
    # 'device': "cuda" if torch.cuda.is_available() else "cpu",
    'device' : 'cuda:1',
    'train_data_path' : "data/train_dataset.pkl",
    'test_data_path' : "data/test_dataset.pkl",
    'save_path' : "train_experiments"

}