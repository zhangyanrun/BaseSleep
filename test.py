import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import pickle
import json
import torch
import sys
import numpy as np
from tqdm import tqdm 

from src.agent import PS_DQNAgent
from src.env import NetworkEnv
# 【修改】引入新的 MetricTracker
from src import utils

# 注意：plot_sample_mesh 需要你自己保留在 test.py 或者也移到 utils.py
# 这里假设你把 plot_sample_mesh 留在了 test.py 或者 utils.py 中

def evaluate(run_dir):
    # 1. 加载配置 (保持不变)
    json_path = os.path.join(run_dir, 'hyperparameters.json')
    if not os.path.exists(json_path):
        print(f"错误：找不到配置文件 {json_path}")
        return

    with open(json_path, 'r') as f:
        all_configs = json.load(f)
    
    RL_PARAMS = all_configs['RL_PARAMS']
    BS_CONFIG = all_configs['BS_CONFIG']
    REWARD_PARAMS = all_configs['REWARD_PARAMS']
    TRAIN_PARAMS = all_configs['TRAIN_PARAMS']
    
    test_data_path = "data/test_dataset.pkl" 
    
    print(f"正在加载测试数据集: {test_data_path} ...")
    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)
    mesh_ids = list(test_dataset.keys())

    # 2. 初始化 (保持不变)
    env = NetworkEnv(test_dataset, BS_CONFIG, 
                     w_energy_saving=REWARD_PARAMS['w_energy_saving'],
                     qos_alpha=REWARD_PARAMS['qos_alpha'],
                     qos_beta=REWARD_PARAMS['qos_beta'],
                     w_drop=REWARD_PARAMS['w_drop'],
                     global_scale=REWARD_PARAMS['global_scale']
                     )

    agent = PS_DQNAgent(
        input_dim=RL_PARAMS['input_dim'],
        hidden_dim1=RL_PARAMS['hidden_dim1'],
        hidden_dim2=RL_PARAMS['hidden_dim2'],
        gcn_output_dim=RL_PARAMS['gcn_output_dim'],
        lr=RL_PARAMS['lr'],
        gamma=RL_PARAMS['gamma'],
        epsilon_start=0.0, 
        epsilon_min=0.0,
        epsilon_decay=0.0,
        memory_size=RL_PARAMS['memory_size'],
        batch_size=RL_PARAMS['batch_size'],
        device=TRAIN_PARAMS['device']
    )
    
    model_path = os.path.join(run_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(run_dir, 'final_model.pth')
    
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy_net.eval()
    
    # ==========================================
    # 3. 测试循环
    # ==========================================
    global_tracker = utils.MetricTracker() # 用于计算全局总分
    
    mesh_metrics_list = [] # 【新增】用于存储每个 Mesh 的独立表现
    
    # 采样用于时间序列画图的 Mesh (比如选第 0 个)
    sample_mesh_id = mesh_ids[0]
    sample_history = {'traffic': [], 'power_base': [], 'power_ai': [], 'active_rate': []}
    
    print("开始评估...")
    for mesh_id in tqdm(mesh_ids):
        features, adj = env.reset(mesh_id)
        done = False
        
        # --- 单个 Mesh 的临时统计器 ---
        local_base_energy = 0.0
        local_ai_energy = 0.0
        local_demand = 0.0
        local_dropped = 0.0
        local_steps = 0
        local_load_sum = 0.0 # 用于计算平均负载
        
        while not done:
            actions = agent.select_actions(features, adj)
            next_features, next_adj, reward, done, info = env.step(actions)
            
            # 1. 更新全局统计
            global_tracker.update(info, actions)
            
            # 2. 更新局部统计 (用于分布图)
            local_base_energy += info['baseline_total_w']
            local_ai_energy += info['actual_total_w']
            local_demand += info['total_demand_mbps']
            local_dropped += info['dropped_mbps']
            
            # 记录负载 (归一化值)
            # features: [N, 5], 第0列是 load ratio
            # 我们取当前时刻所有基站的平均负载率
            current_avg_load = np.mean(features[:, 0])
            local_load_sum += current_avg_load
            
            local_steps += 1
            
            # 3. 记录采样 Mesh 的时间序列 (用于原来的 plot_sample_mesh)
            if mesh_id == sample_mesh_id:
                sample_history['traffic'].append(np.sum(features[:, 0]))
                sample_history['power_base'].append(info['baseline_total_w'])
                sample_history['power_ai'].append(info['actual_total_w'])
                sample_history['active_rate'].append(np.mean(actions))
            
            features = next_features
            adj = next_adj
            
        # --- 单个 Mesh 循环结束，计算该 Mesh 的指标 ---
        if local_base_energy > 0:
            m_esr = (local_base_energy - local_ai_energy) / local_base_energy * 100
        else:
            m_esr = 0.0
            
        if local_demand > 0:
            m_drop = (local_dropped / local_demand) * 100
        else:
            m_drop = 0.0
            
        m_avg_load = local_load_sum / max(local_steps, 1)
        
        # 存入列表
        mesh_metrics_list.append({
            'mesh_id': mesh_id,
            'esr': m_esr,
            'drop_rate': m_drop,
            'avg_load': m_avg_load
        })

    # ==========================================
    # 4. 输出结果
    # ==========================================
    metrics = global_tracker.report()
    
    print("\n" + "="*50)
    print("全局测试报告 (Global Report)")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:<30}: {value:.4f}")
    print("="*50)

    # 准备保存目录
    save_dir = os.path.join(run_dir, 'test_results')
    
    # 1. 画单个 Mesh 的时间序列图 (原来的)
    # plot_sample_mesh(sample_history, save_dir, sample_mesh_id)
    
    # 2. 【新增】画全集分布图
    utils.plot_test_distribution(mesh_metrics_list, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    args = parser.parse_args()

    log_path = os.path.join(args.run_dir, 'test.log')
    sys.stdout = utils.Logger(log_path, sys.stdout)
    
    print(f"日志系统已启动，记录文件: {log_path}")
    print("-" * 50)

    evaluate(args.run_dir)
