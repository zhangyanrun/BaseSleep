import os
import argparse
import pickle
import json
import torch
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

from src import utils
from src.env import NetworkEnv
from src.config import BS_CONFIG

# 只需要导入休眠控制网络进行推理
from src.two_dqns_agent import SleepNet

def evaluate_twodqns(run_dir, model_name):
    # ==========================================
    # 1. 加载配置
    # ==========================================
    json_path = os.path.join(run_dir, 'hyperparameters.json')
    if not os.path.exists(json_path):
        print(f"错误：找不到配置文件 {json_path}")
        return

    with open(json_path, 'r') as f:
        all_configs = json.load(f)
    
    REWARD_PARAMS = all_configs['REWARD_PARAMS']
    
    test_data_path = "data/dataset_3d_test01.pkl" 
    
    print(f"正在加载测试数据集: {test_data_path} ...")
    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)
    mesh_ids = list(test_dataset.keys())

    # ==========================================
    # 2. 确定维度并初始化网络
    # ==========================================
    max_bs_count = 0
    for data in test_dataset.values():
        n = data['traffic_tensor'].shape[1]
        if n > max_bs_count:
            max_bs_count = n
            
    num_bs = max_bs_count
    print(f"检测到测试集全局最大基站数为: {num_bs}，网络将以此维度构建。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sleep_net = SleepNet(state_dim=num_bs * 2, num_bs=num_bs).to(device)
    
    model_path = os.path.join(run_dir, 'models', model_name)
    if not os.path.exists(model_path):
        print(f"错误：找不到模型权重文件 {model_path}")
        return
        
    sleep_net.load_state_dict(torch.load(model_path, map_location=device))
    sleep_net.eval()
    print(f"成功加载 Two-DQNS (SleepNet) 模型权重: {model_path}")

    # ==========================================
    # 3. 初始化评估工具
    # ==========================================
    global_tracker = utils.MetricTracker()
    mesh_metrics_list = []

    # ==========================================
    # 4. 开始测试循环
    # ==========================================
    env = NetworkEnv(test_dataset, BS_CONFIG, **REWARD_PARAMS, is_training=False)

    for mesh_id in tqdm(mesh_ids, desc="Testing Two-DQNS"):
        node_features, adj = env.reset(mesh_id)
        
        prev_actions = np.ones(env.real_n)
        global_tracker.prev_actions = np.ones(env.real_n)

        # 【新增】：为每个 Mesh 单独统计指标
        local_tracker = utils.MetricTracker()
        local_steps = 0
        local_load_sum = 0.0
        
        done = False
        while not done:
            real_n = env.real_n
            pad_len = num_bs - real_n
            
            padded_features = np.pad(node_features, ((0, pad_len), (0, 0)), 'constant')
            padded_prev_actions = np.pad(prev_actions, (0, pad_len), 'constant', constant_values=1)
            
            predicted_load = padded_features[:, 1]
            state_np = np.concatenate([predicted_load, padded_prev_actions])
            state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = sleep_net(state_tensor) # [1, num_bs, 2]
                sleep_actions_padded = q_values.argmax(dim=2).squeeze(0).cpu().numpy()
            
            sleep_actions = sleep_actions_padded[:real_n]
            
            next_features, next_adj, reward, done, info = env.step(sleep_actions)
            
            # 【新增】：记录全局和局部评估指标
            global_tracker.update(info, sleep_actions)
            local_tracker.update(info, sleep_actions)
            
            # 记录平均负载
            current_avg_load = np.mean(node_features[:real_n, 0])
            local_load_sum += current_avg_load
            local_steps += 1
            
            prev_actions = sleep_actions
            node_features = next_features
            
        # 【新增】：计算该 Mesh 最终表现并存入列表
        local_metrics = local_tracker.report()
        mesh_metrics_list.append({
            'mesh_id': mesh_id,
            'esr': local_metrics['ESR (%)'],
            'drop_rate': local_metrics['Drop Rate (%)'],
            'avg_load': local_load_sum / max(local_steps, 1)
        })
        global_tracker.new_episode() # 重置 prev_actions

    # ==========================================
    # 5. 输出测试结果与画图
    # ==========================================
    metrics = global_tracker.report()
    print("\n" + "="*50)
    print("Two-DQNS 全局测试报告 (Global Report)")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key:<30}: {value:.4f}")
    print("="*50)

    save_dir = os.path.join(run_dir, 'test_results', 'twodqns')
    os.makedirs(save_dir, exist_ok=True)
    
    # 【新增】：保存为 CSV 文件
    csv_path = os.path.join(save_dir, 'twodqns_metrics.csv')
    df = pd.DataFrame(mesh_metrics_list)
    df.to_csv(csv_path, index=False)
    print(f"📊 各 Mesh 测试指标已保存至 CSV: {csv_path}")
    
    # 【新增】：绘制分布图
    utils.plot_test_distribution(mesh_metrics_list, save_dir)
    print(f"📈 Two-DQNS 测试结果图表已保存至: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help="训练输出目录 (包含 hyperparameters.json 和 models/)")
    parser.add_argument('--model', type=str, required=True, help="要测试的模型文件名 (例如: twodqns_sleep_epoch_150.pth)")
    args = parser.parse_args()
    
    # 确保 log 打印成功接管
    log_path = os.path.join(args.run_dir, 'test_twodqns.log')
    sys.stdout = utils.Logger(log_path, sys.stdout)
    
    print(f"日志系统已启动，记录文件: {log_path}")
    print("-" * 50)

    evaluate_twodqns(args.run_dir, args.model)