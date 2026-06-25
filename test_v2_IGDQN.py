import os
import argparse
import pickle
import json
import torch
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm 

from src.agent import PS_DQNAgent
from src.env import NetworkEnv
from src import utils
from src.config import BS_CONFIG

def evaluate_igdqn(run_dir):
    json_path = os.path.join(run_dir, 'hyperparameters.json')
    with open(json_path, 'r') as f:
        all_configs = json.load(f)
    
    RL_PARAMS = all_configs['RL_PARAMS']
    REWARD_PARAMS = all_configs['REWARD_PARAMS']
    TRAIN_PARAMS = all_configs['TRAIN_PARAMS']
    test_data_path = TRAIN_PARAMS.get('test_data_path', "data/dataset_3d_test01.pkl")
    
    print(f"正在加载测试数据集: {test_data_path} ...")
    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)
    mesh_ids = list(test_dataset.keys())

    env = NetworkEnv(test_dataset, BS_CONFIG, 
                     w_energy_saving=REWARD_PARAMS.get('w_energy_saving', 1.0),
                     qos_alpha=REWARD_PARAMS.get('qos_alpha', 1.0),
                     qos_beta=REWARD_PARAMS.get('qos_beta', 1.0),
                     qos_threshold=REWARD_PARAMS.get('qos_threshold', 0.9),
                     w_drop=REWARD_PARAMS.get('w_drop', 1.0),
                     w_switch=REWARD_PARAMS.get('w_switch', 1.0),
                     global_scale=REWARD_PARAMS.get('global_scale', 1.0),
                     is_training=False)

    global_tracker = utils.MetricTracker()
    mesh_metrics_list = []
    step_metrics_list = []
    
    print("开始评估 I-GDQN (独立多智能体)...")
    
    # 提前实例化一个空壳智能体，避免在循环中重复创建计算图
    agent = PS_DQNAgent(
        input_dim=RL_PARAMS['input_dim'], hidden_dim1=RL_PARAMS['hidden_dim1'],
        hidden_dim2=RL_PARAMS['hidden_dim2'], gcn_output_dim=RL_PARAMS['gcn_output_dim'],
        lr=RL_PARAMS['lr'], gamma=RL_PARAMS['gamma'],
        epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=0.0,
        memory_size=10, batch_size=10, device=TRAIN_PARAMS['device']
    )

    for mesh_id in tqdm(mesh_ids):
        # 【核心差异】：针对当前 Mesh，动态加载属于它自己的专属权重
        model_path = os.path.join(run_dir, 'models', f'igdqn_mesh_{mesh_id}.pth')
        if not os.path.exists(model_path):
            print(f"警告：找不到网格 {mesh_id} 的专属模型，跳过该网格。")
            continue
            
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.policy_net.eval()
        
        features, adj = env.reset(mesh_id)
        done = False
        
        local_tracker = utils.MetricTracker()
        local_steps = 0
        local_load_sum = 0.0 
        
        while not done:
            actions = agent.select_actions(features, adj)
            next_features, next_adj, reward, done, info = env.step(actions)
            
            global_tracker.update(info, actions)
            local_tracker.update(info, actions)
            
            current_avg_load = np.mean(features[:, 0])
            local_load_sum += current_avg_load
            
            # 记录详细的 step 数据
            base_w = info.get('baseline_total_w', 0)
            ai_w = info.get('actual_total_w', 0)
            demand = info.get('total_demand_mbps', 0)
            drop = info.get('dropped_mbps', 0)
            step_esr = ((base_w - ai_w) / base_w * 100) if base_w > 0 else 0.0
            step_drop_rate = ((drop / demand) * 100) if demand > 0 else 0.0
            num_bs = len(actions)
            num_sleep = num_bs - np.sum(actions)

            step_metrics_list.append({
                'Mesh_ID': mesh_id, 'TimeStep': local_steps,
                'Avg_Load_Ratio': current_avg_load,
                'Baseline_Power_W': base_w, 'AI_Power_W': ai_w,
                'Saved_Power_W': base_w - ai_w,
                'Demand_Mbps': demand, 'Dropped_Mbps': drop,
                'Instant_ESR(%)': step_esr, 'Instant_DropRate(%)': step_drop_rate,
                'Active_BS_Count': np.sum(actions), 'Sleep_BS_Count': num_sleep,
                'Sleep_Ratio(%)': (num_sleep / num_bs * 100) if num_bs > 0 else 0
            })

            local_steps += 1
            features = next_features
            adj = next_adj

        global_tracker.new_episode()
        local_metrics = local_tracker.report()
        
        mesh_record = {
            'mesh_id': mesh_id,
            'esr': local_metrics['ESR (%)'],
            'drop_rate': local_metrics['Drop Rate (%)'],
            'avg_load': local_load_sum / max(local_steps, 1),
        }
        mesh_record.update(local_metrics)
        mesh_metrics_list.append(mesh_record)

    # 结果输出与保存
    metrics = global_tracker.report()
    print("\n" + "="*50)
    print("I-GDQN 全局测试报告 (Global Report)")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:<30}: {value:.4f}")
    print("="*50)

    save_dir = os.path.join(run_dir, 'test_results_v2_igdqn')
    os.makedirs(save_dir, exist_ok=True)
    
    pd.DataFrame(step_metrics_list).to_csv(os.path.join(save_dir, 'step_details_metrics.csv'), index=False)
    pd.DataFrame(mesh_metrics_list).to_csv(os.path.join(save_dir, 'mesh_average_metrics.csv'), index=False)
    utils.plot_test_distribution(mesh_metrics_list, save_dir)
    print(f"✅ I-GDQN 测试结果与分布图已保存至: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help="I-GDQN训练结果文件夹")
    args = parser.parse_args()

    log_path = os.path.join(args.run_dir, 'test.log')
    sys.stdout = utils.Logger(log_path, sys.stdout)
    evaluate_igdqn(args.run_dir)