import os
import argparse
import pickle
import json
import torch
import sys
import numpy as np
from tqdm import tqdm 
import pandas as pd

from src.agent import PS_DQNAgent, PS_MLPAgent
from src.env import NetworkEnv
from src.baselines import AllOnAgent, RandomAgent, ReactiveAgent, MiLSFAgent, DDSSAgent, DFSCSAgent, REDEEMAgent, FullSleepAgent
from src import utils
from src.config import BS_CONFIG


def evaluate(run_dir, mode='drl'):
    # 1. 加载配置 (保持不变)
    json_path = os.path.join(run_dir, 'hyperparameters.json')
    if not os.path.exists(json_path):
        print(f"错误：找不到配置文件 {json_path}")
        return

    with open(json_path, 'r') as f:
        all_configs = json.load(f)
    
    RL_PARAMS = all_configs['RL_PARAMS']
    # BS_CONFIG = all_configs['BS_CONFIG']
    REWARD_PARAMS = all_configs['REWARD_PARAMS']
    TRAIN_PARAMS = all_configs['TRAIN_PARAMS']
    
    test_data_path = TRAIN_PARAMS.get('test_data_path', None)
    
    print(f"正在加载测试数据集: {test_data_path} ...")
    with open(test_data_path, 'rb') as f:
        test_dataset = pickle.load(f)
    mesh_ids = list(test_dataset.keys())

    # 2. 初始化 (保持不变)
    env = NetworkEnv(test_dataset, BS_CONFIG, 
                     w_energy_saving=REWARD_PARAMS['w_energy_saving'],
                     qos_alpha=REWARD_PARAMS['qos_alpha'],
                     qos_beta=REWARD_PARAMS['qos_beta'],
                     qos_threshold=REWARD_PARAMS['qos_threshold'],
                     w_drop=REWARD_PARAMS['w_drop'],
                     w_switch=REWARD_PARAMS['w_switch'],
                     global_scale=REWARD_PARAMS['global_scale'],
                     is_training=False
                     )

    agent = None
    
    if mode == 'drl':
        print(f"正在加载 DRL 模型进行测试...")
        # 初始化 DRL Agent
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
        # 加载权重
        model_path = os.path.join(run_dir, 'best_model.pth') # 需要修改
        if not os.path.exists(model_path):
            print(f"警告：找不到指定的模型文件 {model_path}，尝试加载 final_model.pth ...")
            model_path = os.path.join(run_dir, 'final_model.pth')
        
        agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.policy_net.eval()
    
    # 【新增】变体 1：无 GCN 的 MLP 模型
    elif mode == 'v1_mlp':
        print(f"正在加载变体 1: PS-MLP (无 GCN) 模型进行测试...")
        agent = PS_MLPAgent(
            input_dim=RL_PARAMS['input_dim'],
            hidden_dim1=RL_PARAMS['hidden_dim1'],
            hidden_dim2=RL_PARAMS['hidden_dim2'],
            lr=RL_PARAMS['lr'],
            gamma=RL_PARAMS['gamma'],
            epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=0.0,
            memory_size=RL_PARAMS['memory_size'],
            batch_size=RL_PARAMS['batch_size'],
            device=TRAIN_PARAMS['device']
        )

        model_path = os.path.join(run_dir, 'final_model.pth')
        if os.path.exists(model_path):
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
            agent.policy_net.eval()
            print(f"成功加载变体 1 权重: {model_path}")
        else:
            print(f"严重错误：找不到权重文件 {model_path}")
        
    # 【新增】变体 3：无预测的被动模型 (网络结构和主模型一样，只是输入数据被挖空)
    elif mode == 'v3_reactive':
        print(f"正在加载变体 3: Reactive PS-GDQN (无预测) 模型进行测试...")
        agent = PS_DQNAgent(
            input_dim=RL_PARAMS['input_dim'],
            hidden_dim1=RL_PARAMS['hidden_dim1'],
            hidden_dim2=RL_PARAMS['hidden_dim2'],
            gcn_output_dim=RL_PARAMS['gcn_output_dim'],
            lr=RL_PARAMS['lr'],
            gamma=RL_PARAMS['gamma'],
            epsilon_start=0.0, epsilon_min=0.0, epsilon_decay=0.0,
            memory_size=RL_PARAMS['memory_size'],
            batch_size=RL_PARAMS['batch_size'],
            device=TRAIN_PARAMS['device']
        )

        model_path = os.path.join(run_dir, 'final_model.pth')
        if os.path.exists(model_path):
            agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
            agent.policy_net.eval()
            print(f"成功加载变体 3 权重: {model_path}")
        else:
            print(f"严重错误：找不到权重文件 {model_path}")

    elif mode == 'all_on':
        print(f"正在运行 All-On (全开) 基准策略...")
        agent = AllOnAgent()
        
    elif mode == 'random':
        print(f"正在运行 Random (随机) 基准策略...")
        agent = RandomAgent(seed=42) # 固定种子保证可复现

    elif mode == 'reactive':
        print(f"正在运行无预测的被动模式 (Reactive) 基准策略...")
        agent = ReactiveAgent(bs_config=BS_CONFIG)
    
    elif mode == 'milsf':
        print(f"正在运行 MiLSF (论文复现) 基准策略...")
        agent = MiLSFAgent(bs_config=BS_CONFIG) # 传入配置表

    elif mode == 'ddss':
        print(f"正在运行 DDSS (数据驱动区域统筹) 基准策略...")
        agent = DDSSAgent(bs_config=BS_CONFIG)
    
    elif mode == 'dfscs':
        print(f"正在运行 DFSCS (深度优先协同休眠) 基准策略...")
        agent = DFSCSAgent(bs_config=BS_CONFIG)
    
    elif mode == 'redeem':
        print(f"正在运行 REDEEM (数据驱动能效画像与卸载) 基准策略...")
        agent = REDEEMAgent(bs_config=BS_CONFIG)

    elif mode == 'full_sleep':
        print(f"正在运行 Full-Sleep (全休眠) 基准策略...")
        agent = FullSleepAgent()
    
    else:
        raise ValueError(f"未知的模式: {mode}")
    
    model_path = os.path.join(run_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(run_dir, 'final_model.pth')
    
    
    # ==========================================
    # 3. 测试循环
    # ==========================================
    global_tracker = utils.MetricTracker() # 用于计算全局总分
    
    # 【新增】两个列表，分别存储宏观和微观数据
    mesh_metrics_list = []  # 存储每个 Mesh 的综合平均数据
    step_metrics_list = []  # 存储每个 Mesh 在每个 TimeStep 的详细瞬时数据
    
    print("开始评估...")
    for mesh_id in tqdm(mesh_ids):
        features, adj = env.reset(mesh_id)
        done = False
        
        # 【统一接口】局部 Tracker，用于计算当前这个 Mesh 独有的分数
        local_tracker = utils.MetricTracker()
        # --- 单个 Mesh 的临时统计器 ---
        local_steps = 0
        local_load_sum = 0.0 # 用于计算平均负载
        
        while not done:
            if mode == 'v3_reactive':
                blind_features = features.copy()
                blind_features[:, 1] = blind_features[:, 0]
                actions = agent.select_actions(blind_features, adj)
            else:
                actions = agent.select_actions(features, adj)
            next_features, next_adj, reward, done, info = env.step(actions)
            
            # 1. 更新全局统计
            global_tracker.update(info, actions)
            
            # 2. 更新局部统计 (用于分布图)
            local_tracker.update(info, actions)
            current_avg_load = np.mean(features[:, 0])
            local_load_sum += current_avg_load
            
            base_w = info.get('baseline_total_w', 0)
            ai_w = info.get('actual_total_w', 0)
            demand = info.get('total_demand_mbps', 0)
            drop = info.get('dropped_mbps', 0)
            
            step_esr = ((base_w - ai_w) / base_w * 100) if base_w > 0 else 0.0
            step_drop_rate = ((drop / demand) * 100) if demand > 0 else 0.0
            
            num_bs = len(actions)
            num_active = np.sum(actions)
            num_sleep = num_bs - num_active

            step_metrics_list.append({
                'Mesh_ID': mesh_id,
                'TimeStep': local_steps,
                'Avg_Load_Ratio': current_avg_load,
                'Baseline_Power_W': base_w,
                'AI_Power_W': ai_w,
                'Saved_Power_W': base_w - ai_w,
                'Demand_Mbps': demand,
                'Dropped_Mbps': drop,
                'Instant_ESR(%)': step_esr,
                'Instant_DropRate(%)': step_drop_rate,
                'Active_BS_Count': num_active,
                'Sleep_BS_Count': num_sleep,
                'Sleep_Ratio(%)': (num_sleep / num_bs * 100) if num_bs > 0 else 0
            })

            local_steps += 1
            features = next_features
            adj = next_adj

        global_tracker.new_episode()
        
        # ==========================================
        # 【修改】记录该 Mesh 的总计/平均指标
        # 将 Tracker 计算出的精准宏观结果解包，结合 load 放进字典
        # ==========================================
        local_metrics = local_tracker.report()
        m_avg_load = local_load_sum / max(local_steps, 1)
        
        # 为了兼容 utils 中旧的散点图画图函数，必须保留这三个小写 key
        mesh_record = {
            'mesh_id': mesh_id,
            'esr': local_metrics['ESR (%)'],
            'drop_rate': local_metrics['Drop Rate (%)'],
            'avg_load': m_avg_load,
        }
        # 将 Tracker 里的其他高级信息也合并进去，为了存 CSV 时内容更丰满
        mesh_record.update(local_metrics)
        
        mesh_metrics_list.append(mesh_record)

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

    # 为了不同算法对比不冲突，把结果存入带模式后缀的专属文件夹
    save_dir = os.path.join(run_dir, f'test_results_{mode}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # ==========================================
    # 【新增核心部分】利用 Pandas 将数据一键导出为 CSV
    # ==========================================
    # 1. 导出高颗粒度的 Step 数据 (每行代表某 Mesh 的某 1 秒)
    df_step = pd.DataFrame(step_metrics_list)
    step_csv_path = os.path.join(save_dir, 'step_details_metrics.csv')
    df_step.to_csv(step_csv_path, index=False)
    
    # 2. 导出网格级的汇总数据 (每行代表某 1 个 Mesh 的全程平均表现)
    df_mesh = pd.DataFrame(mesh_metrics_list)
    mesh_csv_path = os.path.join(save_dir, 'mesh_average_metrics.csv')
    df_mesh.to_csv(mesh_csv_path, index=False)
    
    print(f"\n✅ 高颗粒度单步数据已保存至:\n  -> {step_csv_path} (行数: {len(df_step)})")
    print(f"✅ 网格级平均数据已保存至:\n  -> {mesh_csv_path} (行数: {len(df_mesh)})")
    
    # 2. 【新增】画全集分布图
    utils.plot_test_distribution(mesh_metrics_list, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, default='drl', choices=['drl', 'all_on', 'random', 'reactive','milsf', 'ddss', 'dfscs', 'redeem', 'full_sleep', 'v1_mlp', 'v3_reactive'], help="选择测试模式: drl, all_on, random")
    args = parser.parse_args()

    log_path = os.path.join(args.run_dir, 'test.log')
    sys.stdout = utils.Logger(log_path, sys.stdout)
    
    print(f"日志系统已启动，记录文件: {log_path}")
    print("-" * 50)

    evaluate(args.run_dir, args.mode)
