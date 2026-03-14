# ==========================================
# 1. 基站物理参数 (BS Physical Params)
# ==========================================
BS_CONFIG = {
    '4G_macro': {
        'capacity': 300.0,
        'radius_eff': 500.0, 
        'radius_max': 1500.0,
        'p_zero': 400.0,      
        'slope': 200.0,     
        'p_sleep': 80.0
    },
    '4G_micro': {
        'capacity': 100.0,
        'radius_eff': 200.0,
        'radius_max': 300.0,
        'p_zero': 100.0,       
        'slope': 60.0,       
        'p_sleep': 20.0
    },
    '5G_macro': {
        'capacity': 2000.0,
        'radius_eff': 300.0,
        'radius_max': 1000.0,
        'p_zero': 1000.0,      
        'slope': 500.0,      
        'p_sleep': 150.0
    },
    '5G_micro': {
        'capacity': 800.0,
        'radius_eff': 100.0,
        'radius_max': 300.0,
        'p_zero': 180.0,      
        'slope': 120.0,      
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
    # 状态维度: [Load_t(1), Load_t+1(1), OneHot_Type(4)]
    'input_dim': 6,
    
    # 隐藏层维度: 128 (GCN 和 MLP 共享)
    'hidden_dim1': 64,
    'hidden_dim2': 256,

    # GCN 最终输出的特征维度 (你要求的 16)
    'gcn_output_dim': 16,
    
    # 学习率 (Learning Rate)
    'lr': 1e-3,
    
    # 折扣因子 (Discount Factor)
    'gamma': 0.95,
    
    # 探索率 (Epsilon Greedy)
    'epsilon_start': 1.0,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.9998, # 减慢衰减速度
    
    # 经验回放池大小
    'memory_size': 10000,
    
    # 注意: PS-IDQN 每次训练一个 Mesh，这里指从 Buffer 中采样的数量
    'batch_size': 64
    }

# ==========================================
# 3. 奖励与惩罚系数 (Reward Coefficients)
# ==========================================
REWARD_PARAMS = {
    # 用“节省能耗”的缩放系数
    'w_energy_saving': 15, 
    
    # 非线性 QoS 惩罚参数 公式: alpha * (exp(beta * load) - 1)
    'qos_alpha': 5,     # 基础系数，控制整体惩罚幅度
    'qos_beta': 6,    # 指数系数，控制"陡峭"程度。beta=6时，Load=1.0 -> exp(6)≈403 (惩罚巨大)
    'qos_threshold': 0.8,   # 安全负载阈值 (rho_th)，例如 0.8 表示 80%

    # 掉线依然是不可接受的，保留最严厉的线性惩罚
    'w_drop': 150,

    # 假设一次切换惩罚相当于扣掉省 100W 电的奖励，则设为 500 (100 * 5)
    'w_switch': 200,

    'global_scale': 0.000001
}

# ==========================================
# 4. 训练流程参数 (Training Flow)
# ==========================================
TRAIN_PARAMS = {
    'num_epochs': 150,      # 总共把所有 Mesh 轮询多少遍
    'log_interval': 20,     # 每训练多少个 Mesh 打印一次日志
    'target_update': 30,    # 每多少个 Mesh 更新一次目标网络
    'device' : 'cuda:1',
    'train_data_path': 'data/dataset_3d_test01.pkl', 
    'test_data_path': 'data/dataset_3d_test01.pkl',
    'save_path' : "train_experiments"

}

# ==========================================
# 4. DeepBSC (DDPG + BT + EN) 专属超参数
# 严格参考原论文 Table III 和 Table IV 设置
# ==========================================
DEEPBSC_PARAMS = {
    # --- 网络结构参数 (Table III) ---
    'hidden_dim1': 800,   # Actor/Critic 第一层隐藏层维度
    'hidden_dim2': 600,   # Actor/Critic 第二层隐藏层维度
    
    # --- 学习率 (Table IV) ---
    'lr_actor': 1e-4,     # Actor 学习率 (\alpha_\pi)
    'lr_critic': 1e-3,    # Critic 学习率 (\alpha_Q)
    
    # --- DDPG 训练参数 ---
    'gamma': 0.9,         # 折扣因子 (\gamma)
    'tau': 1e-4,          # 目标网络软更新系数 (\tau) 
    'memory_size': 10000, # 经验回放池大小 (N_{exp})
    'batch_size': 64,     # 训练批次大小 (N_e)
    
    # --- Explorer Network 参数 ---
    'explore_alpha': 0.1, # 探索者网络扰动幅度 (\alpha)
    'explore_sigma': 0.05 # 探索者网络权重软更新系数 (\sigma) 
}