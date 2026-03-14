import os
import pickle
import sys
import time
import numpy as np
import torch


from src import utils
from src.env import NetworkEnv
from src.agent import PS_DQNAgent
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
    agent = PS_DQNAgent(input_dim=RL_PARAMS['input_dim'], 
                        hidden_dim1=RL_PARAMS['hidden_dim1'],
                        hidden_dim2=RL_PARAMS['hidden_dim2'], 
                        gcn_output_dim=RL_PARAMS['gcn_output_dim'],
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
    
    # 3. 初始化记录容器
    # total_step = 0 
    history = {
        'rewards': [],
        'esr': [],
        'drop_rate': [],
        'sleep_ratio': [],
        'power_efficiency': [],
        'switch_freq': [],
        'dropped_mbps': [],
        'save_power_kw': [],
        'epsilon': []
    }

    # 追踪最佳模型
    best_reward = -float('inf') # 初始化为负无穷
    best_epoch = -1
    best_model_path = None

    # 开始训练循环
    for epoch in range(TRAIN_PARAMS['num_epochs']):
        np.random.shuffle(mesh_ids)

        # Reward 是机器眼里的“虚拟分数”，它不对应具体的物理量。强化学习的目的是最大化每个 Episode (Mesh) 的累计回报。
        # 因此，我们需要先算出一个 Mesh 的总 Reward，最后再求所有 Mesh 的平均 Reward，以此来评估模型的“得分能力”。
        epoch_reward = 0 
        # epoch_dropped = 0
        # epoch_save_power = 0
        epoch_tracker = utils.MetricTracker()
        epoch_mesh_metrics_list = []
        
        for i, mesh_id in enumerate(mesh_ids):
            # reset 返回两个值，该时刻的流量数据和邻接矩阵
            features, adj = env.reset(mesh_id)
            
            # 循环标志位
            done = False
            
            # 临时统计单个 Mesh 的掉线和能耗
            mesh_reward = 0
            # mesh_dropped = 0
            # mesh_save_power = 0
            # steps = 0
            # 局部 Tracker，记录单个 Mesh 的表现
            local_tracker = utils.MetricTracker()
            local_steps = 0
            local_load_sum = 0.0
            
            while not done:
                # 1. 决策:根据当前的状态获得动作
                actions = agent.select_actions(features, adj)
                
                # 2. 执行：执行动作，计算奖励
                next_features, next_adj, reward, done, info = env.step(actions)
                
                # 3. 存储：
                agent.store_transition(features, adj, actions, reward, next_features, next_adj, done)
                
                # 4. 学习：计算Q值，更新参数
                agent.learn()
                
                # 更新状态
                features = next_features
                adj = next_adj
                mesh_reward += reward

                epoch_tracker.update(info, actions)
                local_tracker.update(info, actions)
                
                # ==========================================
                # 【修改】只提取计算该 Mesh 平均负载所需的变量，删除了所有时间步的繁琐计算
                # ==========================================
                current_avg_load = np.mean(features[:, 0])
                local_load_sum += current_avg_load
                
                local_steps += 1

            
            #Reward 追求的是“总和最大化”，而 Power 和 Dropped 追求的是“平均性能的可比性”
            epoch_reward += mesh_reward
            # epoch_dropped += (mesh_dropped / steps) 
            # epoch_save_power += (mesh_save_power / steps)
            # total_step += 1

            agent.decay_epsilon()

            epoch_tracker.new_episode() #
            local_tracker.new_episode()

            # ==========================================
            # 宏观数据：记录该 Mesh 本 Epoch 的平均/总计表现
            # ==========================================
            local_metrics = local_tracker.report()
            mesh_record = {
                'Epoch': epoch + 1,
                'Mesh_ID': mesh_id,
                'avg_load': local_load_sum / max(local_steps, 1)
            }
            mesh_record.update(local_metrics)
            epoch_mesh_metrics_list.append(mesh_record)

            if (i + 1) % TRAIN_PARAMS['target_update'] == 0:
                agent.update_target()
            # 打印进度
            if i % 20 == 0: 
                print(f"Epoch {epoch+1}/{TRAIN_PARAMS['num_epochs']} | Progress {i}/{len(mesh_ids)} | "
                      f"Mesh {mesh_id:<3} | Rw: {mesh_reward:.1f} | Sp: {info['save_power_w']:.1f} | "
                      f"Drop: {info['dropped_mbps']:.1f} | Eps: {agent.epsilon:.3f}")

        # --- Epoch 结束，结算指标 ---
        
        # 1. 计算平均 Reward
        avg_reward = epoch_reward / len(mesh_ids)
        
        # 2. 从 Tracker 获取工程物理指标
        # 【重要注释：为什么物理指标不用平均值？】
        # 假设 Mesh A 需求 1000M，掉线 100M (掉线率 10%)
        # 假设 Mesh B 需求 10M，掉线 5M (掉线率 50%)
        # 如果我们简单取平均，掉线率是 (10%+50%)/2 = 30%，这就严重放大了边缘情况的影响。
        # epoch_tracker 的底层是：先用积累值算出 (100+5) / (1000+10) = 10.3%，这才是真正严谨的“宏观掉线率 (Macro Drop Rate)”。
        metrics = epoch_tracker.report()
        # avg_dropped = metrics['dropped_mbps'] / len(mesh_ids)
        # avg_save_power = metrics['save_power_w'] / len(mesh_ids)
        
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
        # 【修改核心】仅将当前 Epoch 的网格数据实时追加写入 CSV 
        # ==========================================
        mesh_csv_path = os.path.join(run_dir, 'train_mesh_average_metrics.csv')
        df_mesh = pd.DataFrame(epoch_mesh_metrics_list)
        df_mesh.to_csv(mesh_csv_path, mode='a', header=not os.path.exists(mesh_csv_path), index=False)
        # ==========================================

        # 保存当前 Epoch 的模型
        current_model_path = os.path.join(models_dir, f'model_epoch_{epoch+1}_rw_{avg_reward:.2f}.pth')
        torch.save(agent.policy_net.state_dict(), current_model_path)

        # 【新增 2】 判断并保存最佳模型 (Best Model)
        # ==========================================
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_epoch = epoch + 1
            
            # 保存最佳模型到根目录，方便直接取用
            best_model_path = os.path.join(run_dir, 'best_model.pth')
            torch.save(agent.policy_net.state_dict(), best_model_path)

            print(f"发现新高! 最佳模型已更新 (Epoch {best_epoch}, Reward {best_reward:.2f})")
        
        print("") # 空行分隔
            
    # 保存最终模型
    final_model_path = os.path.join(run_dir, 'final_model.pth')
    torch.save(agent.policy_net.state_dict(), final_model_path)
    print(f"最终模型已保存至: {final_model_path}")

    # ==========================================
    # 【修改点 2】 调用绘图函数时传入 timestamp
    # ==========================================
    print("正在生成可视化结果...")
    utils.plot_learning_curve(history, run_dir + "/train_results")

    # 【新增 3】 输出最终总结
    print("="*50)
    print(f"训练全部结束!")
    print(f"最佳表现 Epoch: {best_epoch}")
    print(f"最高平均 Reward: {best_reward:.4f}")
    print(f"最佳模型 epoch: {best_epoch}")
    print("="*50)

if __name__ == "__main__":
    total_start_time = time.time()
    # ==========================================
    # 1. 创建本次实验的专属文件夹 (Train Folder)
    # ==========================================
    # 根目录名字
    ROOT_DIR = TRAIN_PARAMS['save_path']
    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    # 生成时间戳文件夹: train_experiments/20251122_203015
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ROOT_DIR, timestamp)
    os.makedirs(run_dir)

    print(f"本次实验目录已创建: {run_dir}")

    # ==========================
    # 2. 配置日志记录 
    # ==========================
    log_path = os.path.join(run_dir, 'train.log')
    sys.stdout = utils.Logger(log_path, sys.stdout)
    
    print(f"日志系统已启动，记录文件: {log_path}")
    print("-" * 50)

    # ==========================================
    # 3. 保存超参数为 JSON
    # ==========================================
    # 把所有的配置字典整合起来
    all_configs = {
        "RL_PARAMS": RL_PARAMS,
        "REWARD_PARAMS": REWARD_PARAMS,
        "TRAIN_PARAMS": TRAIN_PARAMS
    }
    utils.save_config_to_json(run_dir, all_configs)
    
    # ==========================================
    # 4. 运行主训练函数
    # ==========================================
    main(run_dir)

    # ==========================================
    # 5. 训练结束，输出总耗时和平均每轮耗时
    # ==========================================
    total_training_time = time.time() - total_start_time
    avg_epoch_time = total_training_time/TRAIN_PARAMS['num_epochs']
    print(f"训练总耗时: {utils.format_duration(total_training_time)}")
    print(f"平均每轮耗时: {utils.format_duration(avg_epoch_time)}")