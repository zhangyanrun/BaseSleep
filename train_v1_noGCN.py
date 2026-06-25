import os
import pickle
import sys
import time
import numpy as np
import torch
import pandas as pd

from src import utils
from src.env import NetworkEnv
from src.agent import PS_MLPAgent
from src.config import BS_CONFIG, TRAIN_PARAMS, RL_PARAMS, REWARD_PARAMS

def main(run_dir):
    # 0. 在 run_dir 下创建一个 models 文件夹，用来存放所有 epoch 的模型
    models_dir = os.path.join(run_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 1. 加载数据
    print("正在加载数据集...")
    with open(TRAIN_PARAMS['train_data_path'], 'rb') as f:
        dataset = pickle.load(f)
    mesh_ids = list(dataset.keys())

    # 2. 初始化模型
    print("正在初始化模型...")
    agent = PS_MLPAgent(input_dim=RL_PARAMS['input_dim'], 
                        hidden_dim1=RL_PARAMS['hidden_dim1'],
                        hidden_dim2=RL_PARAMS['hidden_dim2'], 
                        lr=RL_PARAMS['lr'], 
                        gamma=RL_PARAMS['gamma'],
                        epsilon_start=RL_PARAMS['epsilon_start'], 
                        epsilon_min=RL_PARAMS['epsilon_min'],
                        epsilon_decay=RL_PARAMS['epsilon_decay'], 
                        memory_size=RL_PARAMS['memory_size'],
                        batch_size=RL_PARAMS['batch_size'], 
                        device=TRAIN_PARAMS['device'])
    
    env = NetworkEnv(dataset, BS_CONFIG, 
                     REWARD_PARAMS['w_energy_saving'], 
                     REWARD_PARAMS['qos_alpha'], 
                     REWARD_PARAMS['qos_beta'],
                     REWARD_PARAMS['qos_threshold'],
                     REWARD_PARAMS['w_drop'], 
                     REWARD_PARAMS['w_switch'],
                     REWARD_PARAMS['global_scale'], 
                     is_training=True)
    
    print(f"开始训练... 共 {len(mesh_ids)} 个 Mesh, 训练 {TRAIN_PARAMS['num_epochs']} 轮 (Epochs)")
    
    # 3. 初始化宏观记录容器 (用于画曲线和存 CSV)
    history = {
        'rewards': [], 'esr': [], 'drop_rate': [], 'sleep_ratio': [],
        'power_efficiency': [], 'switch_freq': [], 'dropped_mbps': [],
        'save_power_kw': [], 'epsilon': []
    }

    # 追踪最佳模型
    best_reward = -float('inf') 
    best_epoch = -1

    # ==========================================
    # 开始训练循环
    # ==========================================
    for epoch in range(TRAIN_PARAMS['num_epochs']):
        np.random.shuffle(mesh_ids)

        epoch_reward = 0 
        epoch_tracker = utils.MetricTracker()
        
        for i, mesh_id in enumerate(mesh_ids):
            features, adj = env.reset(mesh_id)
            done = False
            mesh_reward = 0
            
            while not done:
                # 1. 决策: 根据当前的状态获得动作
                actions = agent.select_actions(features, adj)
                
                # 2. 执行: 执行动作，计算奖励
                next_features, next_adj, reward, done, info = env.step(actions)
                
                # 3. 存储: 将经验存入回放池
                agent.store_transition(features, adj, actions, reward, next_features, next_adj, done)
                
                # 4. 学习: 计算Q值，更新网络参数
                agent.learn()
                
                # 更新状态与局部统计
                features = next_features
                adj = next_adj
                mesh_reward += reward

                epoch_tracker.update(info, actions)
                
            # --- 单个 Mesh 结束 ---
            epoch_reward += mesh_reward
            agent.decay_epsilon()
            epoch_tracker.new_episode()

            # 更新目标网络
            if (i + 1) % TRAIN_PARAMS['target_update'] == 0:
                agent.update_target()
                
            # 打印进度
            if i % TRAIN_PARAMS['log_interval'] == 0: 
                print(f"Epoch {epoch+1}/{TRAIN_PARAMS['num_epochs']} | Progress {i}/{len(mesh_ids)} | "
                      f"Mesh {mesh_id:<3} | Rw: {mesh_reward:.1f} | Sp: {info['save_power_w']:.1f} | "
                      f"Drop: {info['dropped_mbps']:.1f} | Eps: {agent.epsilon:.3f}")

        # --- Epoch 结束，结算全局指标 ---
        avg_reward = epoch_reward / len(mesh_ids)
        metrics = epoch_tracker.report()
        
        history['rewards'].append(avg_reward)
        history['esr'].append(metrics['ESR (%)'])
        history['drop_rate'].append(metrics['Drop Rate (%)'])
        history['sleep_ratio'].append(metrics['Sleep Ratio (%)'])
        history['power_efficiency'].append(metrics['Power Efficiency (Mbps/kW)'])
        history['switch_freq'].append(metrics['Switching Frequency (%)'])
        history['dropped_mbps'].append(metrics['Total Dropped (Mb)'])
        history['save_power_kw'].append(metrics['Total Saved Energy (kW)'])
        history['epsilon'].append(agent.epsilon)

        print(f"Epoch {epoch+1} 完成! 平均奖励: {avg_reward:.2f}\n")

        # ==========================================
        # 将当前 Epoch 的全局均值追加写入 CSV (防崩溃)
        # ==========================================
        epoch_csv_path = os.path.join(run_dir, 'train_epoch_learning_curve.csv')
        current_epoch_data = {k: [v[-1]] for k, v in history.items()}
        current_epoch_data['Epoch'] = [epoch + 1] 
        
        df_epoch = pd.DataFrame(current_epoch_data)
        df_epoch.to_csv(epoch_csv_path, mode='a', header=not os.path.exists(epoch_csv_path), index=False)

        # ==========================================
        # 判断并保存最佳模型
        # ==========================================
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_epoch = epoch + 1
            
            best_model_path = os.path.join(run_dir, 'best_model.pth')
            torch.save(agent.policy_net.state_dict(), best_model_path)
            print(f"🌟 发现新高! 最佳模型已更新 (Epoch {best_epoch}, Reward {best_reward:.2f})")
        
        print("") # 空行分隔
            
    # ==========================================
    # 训练全部结束，保存最终模型与图表
    # ==========================================
    final_model_path = os.path.join(run_dir, 'final_model.pth')
    torch.save(agent.policy_net.state_dict(), final_model_path)
    print(f"最终模型已保存至: {final_model_path}")

    print("正在生成可视化结果...")
    utils.plot_learning_curve(history, os.path.join(run_dir, "train_results"))

    print("="*50)
    print(f"训练全部结束!")
    print(f"最佳表现 Epoch: {best_epoch}")
    print(f"最高平均 Reward: {best_reward:.4f}")
    print("="*50)

if __name__ == "__main__":
    total_start_time = time.time()
    
    # 1. 创建本次实验的专属文件夹
    ROOT_DIR = TRAIN_PARAMS['save_path']
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    timestamp = time.strftime("%Y%m%d_%H%M%S"+ "_V1_noDQN")
    run_dir = os.path.join(ROOT_DIR, timestamp)
    os.makedirs(run_dir)

    print(f"本次实验目录已创建: {run_dir}")

    # 2. 配置日志记录 
    log_path = os.path.join(run_dir, 'train.log')
    sys.stdout = utils.Logger(log_path, sys.stdout)
    
    print(f"日志系统已启动，记录文件: {log_path}")
    print("-" * 50)

    # 3. 保存超参数为 JSON
    all_configs = {
        "RL_PARAMS": RL_PARAMS,
        "REWARD_PARAMS": REWARD_PARAMS,
        "TRAIN_PARAMS": TRAIN_PARAMS
    }
    utils.save_config_to_json(run_dir, all_configs)
    
    # 4. 运行主训练函数
    main(run_dir)

    # 5. 训练结束，输出耗时
    total_training_time = time.time() - total_start_time
    avg_epoch_time = total_training_time / TRAIN_PARAMS['num_epochs']
    print(f"训练总耗时: {utils.format_duration(total_training_time)}")
    print(f"平均每轮耗时: {utils.format_duration(avg_epoch_time)}")