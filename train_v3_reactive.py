import os, pickle, time, sys
import numpy as np
import torch
from src import utils
from src.env import NetworkEnv
from src.agent import PS_DQNAgent
from src.config import BS_CONFIG, TRAIN_PARAMS, RL_PARAMS, REWARD_PARAMS

def main(run_dir):
    models_dir = os.path.join(run_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    with open(TRAIN_PARAMS['train_data_path'], 'rb') as f:
        dataset = pickle.load(f)
    mesh_ids = list(dataset.keys())

    agent = PS_DQNAgent(
        input_dim=RL_PARAMS['input_dim'], hidden_dim1=RL_PARAMS['hidden_dim1'],
        hidden_dim2=RL_PARAMS['hidden_dim2'], gcn_output_dim=RL_PARAMS['gcn_output_dim'],
        lr=RL_PARAMS['lr'], gamma=RL_PARAMS['gamma'],
        epsilon_start=RL_PARAMS['epsilon_start'], epsilon_min=RL_PARAMS['epsilon_min'],
        epsilon_decay=RL_PARAMS['epsilon_decay'], memory_size=RL_PARAMS['memory_size'],
        batch_size=RL_PARAMS['batch_size'], device=TRAIN_PARAMS['device']
    )
    env = NetworkEnv(dataset, BS_CONFIG, **REWARD_PARAMS, is_training=True)

    for epoch in range(TRAIN_PARAMS['num_epochs']):
        epoch_reward = []
        for mesh_id in mesh_ids:
            features, adj = env.reset(mesh_id)
            done = False
            while not done:
                # 【消融核心 3】：剥夺预测能力
                # 将 state 中的“预测流量列[:, 1]” 替换为 “当前流量列[:, 0]”
                blind_features = features.copy()
                blind_features[:, 1] = blind_features[:, 0] 

                actions = agent.select_actions(blind_features, adj)
                next_features, next_adj, reward, done, info = env.step(actions)
                
                blind_next_features = next_features.copy()
                blind_next_features[:, 1] = blind_next_features[:, 0]

                # 存入经验池的也是没有预测能力的状态
                agent.store_transition(blind_features, adj, actions, reward, blind_next_features, next_adj, done)
                agent.learn()
                
                features = next_features
                adj = next_adj
                epoch_reward.append(reward)
                
            agent.decay_epsilon()
            if (epoch + 1) % TRAIN_PARAMS['target_update'] == 0:
                agent.update_target()
                
        print(f"Epoch {epoch+1} | 平均 Reward: {np.mean(epoch_reward):.4f}")

    torch.save(agent.policy_net.state_dict(), os.path.join(models_dir, 'final_model.pth'))

if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d_%H%M%S") + "_V3_Reactive"
    run_dir = os.path.join(TRAIN_PARAMS['save_path'], timestamp)
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = utils.Logger(os.path.join(run_dir, 'train.log'), sys.stdout)
    main(run_dir)