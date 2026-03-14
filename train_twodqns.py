import os
import pickle
import time
import numpy as np
import torch
from tqdm import tqdm

from src import utils
from src.env import NetworkEnv
from src.config import BS_CONFIG, TRAIN_PARAMS, RL_PARAMS, REWARD_PARAMS

from src.two_dqns_agent import TwoDQNSAgent

def main():
    ROOT_DIR = TRAIN_PARAMS['save_path']
    timestamp = time.strftime("%Y%m%d_%H%M%S") + "_TwoDQNS"
    run_dir = os.path.join(ROOT_DIR, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    models_dir = os.path.join(run_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Two-DQNS 对比实验目录已创建: {run_dir}")

    # --- 1. 数据加载与最大维度探测 ---
    print("正在加载数据集...")
    with open(TRAIN_PARAMS['train_data_path'], 'rb') as f:
        dataset = pickle.load(f)
    mesh_ids = list(dataset.keys())
    
    max_bs_count = max([data['traffic_tensor'].shape[1] for data in dataset.values()])
    num_bs = max_bs_count
    print(f"检测到全局最大基站数为: {num_bs}，Two-DQNS 将以此维度构建。")

    # --- 2. 初始化智能体与环境 ---
    agent = TwoDQNSAgent(num_bs=num_bs, lr=1e-3, gamma=RL_PARAMS['gamma'],device=TRAIN_PARAMS['device'])
    env_ai = NetworkEnv(dataset, BS_CONFIG, **REWARD_PARAMS, is_training=True)

    # 把所有的配置字典整合起来
    all_configs = {
        "RL_PARAMS": RL_PARAMS,
        "REWARD_PARAMS": REWARD_PARAMS,
        "TRAIN_PARAMS": TRAIN_PARAMS
    }
    utils.save_config_to_json(run_dir, all_configs)
    for epoch in range(TRAIN_PARAMS['num_epochs']):
        print(f"\n=== Two-DQNS Epoch {epoch+1}/{TRAIN_PARAMS['num_epochs']} ===")
        epoch_rewards = []
        
        for mesh_id in tqdm(mesh_ids, desc="Training Meshes"):
            node_features, adj = env_ai.reset(mesh_id)
            prev_actions = np.ones(env_ai.real_n)
            
            done = False
            while not done:
                # Padding 逻辑
                real_n = env_ai.real_n
                pad_len = num_bs - real_n
                padded_features = np.pad(node_features, ((0, pad_len), (0, 0)), 'constant')
                padded_prev_actions = np.pad(prev_actions, (0, pad_len), 'constant', constant_values=1)
                
                # DQN 级联动作选择
                sleep_actions_padded, power_actions_padded, state_np = agent.select_actions(padded_features, padded_prev_actions)
                
                # 截断真实动作传给你的环境
                sleep_actions = sleep_actions_padded[:real_n]
                
                # 【论文精髓映射】：因为当前 env 无法接收 power_actions，我们只传 sleep_actions
                # DQN-2 依然会正常跑完前向传播和反向传播，只是它此时像一个伴生观察者
                next_features, next_adj, reward, done, info = env_ai.step(sleep_actions)
                epoch_rewards.append(reward)
                
                padded_next_features = np.pad(next_features, ((0, pad_len), (0, 0)), 'constant')
                next_state_np = np.concatenate([padded_next_features[:, 1], sleep_actions_padded])
                
                # 存入完整的经验组合：(S, A_sleep, A_power, R, S')
                agent.memory.append((state_np, sleep_actions_padded, power_actions_padded, reward, next_state_np))
                
                agent.train_step(batch_size=RL_PARAMS.get('batch_size', 64))
                
                node_features = next_features
                prev_actions = sleep_actions
                
        # Epoch 结束更新 Target 网络
        agent.update_target_network()
        print(f"Epoch {epoch+1} 平均 Reward: {np.mean(epoch_rewards):.4f}, 探索率 Epsilon: {agent.epsilon:.4f}")
        
        if (epoch + 1) % TRAIN_PARAMS.get('save_interval', 1) == 0:
            torch.save(agent.sleep_net.state_dict(), os.path.join(models_dir, f"twodqns_sleep_epoch_{epoch+1}.pth"))
            torch.save(agent.power_net.state_dict(), os.path.join(models_dir, f"twodqns_power_epoch_{epoch+1}.pth"))
   # 循环结束后，强制保存最终模型
    torch.save(agent.sleep_net.state_dict(), os.path.join(models_dir, "twodqns_sleep_final.pth"))
    torch.save(agent.power_net.state_dict(), os.path.join(models_dir, "twodqns_power_final.pth")) 
    print("Two-DQNS 训练完成！")

if __name__ == "__main__":
    log_path = os.path.join(run_dir, 'train.log')
    sys.stdout = utils.Logger(log_path, sys.stdout)
    main()