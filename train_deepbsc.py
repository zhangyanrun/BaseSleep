import os
import pickle
import time
import numpy as np
import torch
from tqdm import tqdm

from src import utils
from src.env import NetworkEnv
from src.baselines import AllOnAgent
from src.config import BS_CONFIG, TRAIN_PARAMS, RL_PARAMS, REWARD_PARAMS, DEEPBSC_PARAMS

# 导入刚才写好的 DeepBSC 智能体和回放池
from src.deepbsc_agent import DeepBSCAgent, ReplayBuffer

def main():
    # ==========================================
    # 1. 准备日志和目录
    # ==========================================
    ROOT_DIR = TRAIN_PARAMS['save_path']
    timestamp = time.strftime("%Y%m%d_%H%M%S") + "_DeepBSC"
    run_dir = os.path.join(ROOT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    models_dir = os.path.join(run_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"DeepBSC 对比实验目录已创建: {run_dir}")

    # ==========================================
    # 2. 加载数据
    # ==========================================
    print("正在加载数据集...")
    with open(TRAIN_PARAMS['train_data_path'], 'rb') as f:
        dataset = pickle.load(f)
    mesh_ids = list(dataset.keys())
    
    max_bs_count = 0
    for data in dataset.values():
        n = data['traffic_tensor'].shape[1]
        if n > max_bs_count:
            max_bs_count = n
            
    num_bs = max_bs_count # 统一使用最大维度初始化网络
    print(f"检测到全局最大基站数为: {num_bs}，网络将以此维度构建。")

    # ==========================================
    # 3. 初始化 DeepBSC 智能体与基准策略 (BT机制)
    # ==========================================
    # 这里我们不用你的 DQN 里的复杂配置，直接给 DeepBSC 初始化
    agent = DeepBSCAgent(num_bs=num_bs, lr_actor=1e-4, lr_critic=1e-3, gamma=RL_PARAMS['gamma'],device=TRAIN_PARAMS['device'])
    replay_buffer = ReplayBuffer(capacity=RL_PARAMS['memory_size'])
    
    # 实例化全开策略作为 Benchmark Transformation 的对照组
    baseline_agent = AllOnAgent()

    # ==========================================
    # 4. 初始化环境 (移到循环外部，传入完整 dataset)
    # ==========================================
    # 主环境给 DeepBSC 用
    env_ai = NetworkEnv(dataset, BS_CONFIG, **REWARD_PARAMS, is_training=True)
    # 影子环境给 Baseline 用
    env_base = NetworkEnv(dataset, BS_CONFIG, **REWARD_PARAMS, is_training=True)

   #  epochs = range(TRAIN_PARAMS['num_epochs'])

    # 把所有的配置字典整合起来
    all_configs = {
        "RL_PARAMS": RL_PARAMS,
        "REWARD_PARAMS": REWARD_PARAMS,
        "TRAIN_PARAMS": TRAIN_PARAMS
    }
    utils.save_config_to_json(run_dir, all_configs)
    
    for epoch in range(TRAIN_PARAMS['num_epochs']):
        print(f"\n=== DeepBSC Epoch {epoch+1}/{TRAIN_PARAMS['num_epochs']} ===")
        epoch_reward_gaps = []
        
        for mesh_id in tqdm(mesh_ids, desc="Training Meshes"):
            # 【修复点】：在这里传入 mesh_id 进行 reset
            node_features, adj = env_ai.reset(mesh_id)
            env_base.reset(mesh_id) # 保持环境初始状态同步
            
            # 初始上一时刻动作 (全开)
            prev_actions = np.ones(env_ai.real_n)
            
            done = False
            while not done:
               # ---------------------------------------------------
                # 【新增】：补齐逻辑 (Padding)
                # ---------------------------------------------------
                real_n = env_ai.real_n
                pad_len = num_bs - real_n
                
                # 对特征和上一步动作补零，使其长度强制等于 num_bs
                padded_features = np.pad(node_features, ((0, pad_len), (0, 0)), 'constant')
                padded_prev_actions = np.pad(prev_actions, (0, pad_len), 'constant', constant_values=1)
                # ---------------------------------------------------
                # a) DeepBSC (AI) 在主环境中执行
                # ---------------------------------------------------
                # 【修改点】：这里必须传入 padded_features 和 padded_prev_actions
                action_ai_padded, state_np, action_prob_padded = agent.select_actions(padded_features, padded_prev_actions)
                
                # 【修改点】：从全尺寸动作中截取真实的动作 (抛弃多余的虚拟基站动作)，去和环境交互
                action_ai = action_ai_padded[:real_n]
                next_features, next_adj, reward_ai, done, info = env_ai.step(action_ai)
                
                # ---------------------------------------------------
                # b) Baseline 在影子环境中执行 (Benchmark Transformation)
                # ---------------------------------------------------
                action_base = baseline_agent.select_actions(node_features, adj)
                _, _, reward_base, _, _ = env_base.step(action_base)
                
                # ---------------------------------------------------
                # c) 计算 Gap 并存入经验池进行训练
                # ---------------------------------------------------
                # 计算 AI 比全开策略多出来的收益 (Gap)
                reward_gap = reward_ai - reward_base 
                epoch_reward_gaps.append(reward_gap)
                
                # 构建 next_state (预测流量 + 本次动作)
                # 对下一个状态也进行 Padding，保证 Replay Buffer 里的形状永远一致
                padded_next_features = np.pad(next_features, ((0, pad_len), (0, 0)), 'constant')
                next_state_np = np.concatenate([padded_next_features[:, 1], action_ai_padded])
               #  next_state_np = np.concatenate([next_features[:, 1], action_ai])
                
                replay_buffer.push(state_np, action_prob_padded, reward_gap, next_state_np)
                
                # 执行一次 DDPG 梯度下降
                agent.train_step(replay_buffer, batch_size=RL_PARAMS.get('batch_size', 64))
                
                # 状态更替
                node_features = next_features
                adj = next_adj
                prev_actions = action_ai

        print(f"Epoch {epoch+1} 平均 Reward Gap (AI比全开多赚的): {np.mean(epoch_reward_gaps):.4f}")
        
        # 每隔几个 epoch 保存一次模型
        if (epoch + 1) % TRAIN_PARAMS.get('save_interval', 1) == 0:
            torch.save(agent.actor.state_dict(), os.path.join(models_dir, f"deepbsc_actor_epoch_{epoch+1}.pth"))
            
    # 最终保存
    torch.save(agent.actor.state_dict(), os.path.join(models_dir, "deepbsc_actor_final.pth"))
    print("DeepBSC 训练完成！")

if __name__ == "__main__":
    log_path = os.path.join(run_dir, 'train.log')
    sys.stdout = utils.Logger(log_path, sys.stdout)
    main()