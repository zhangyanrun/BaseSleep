import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    # ==========================================
    # 1. 准备模型保存子目录
    # ==========================================
    # 在 run_dir 下创建一个 models 文件夹，用来存放所有 epoch 的模型
    models_dir = os.path.join(run_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 1. 加载数据
    print("正在加载数据集...")
    with open(TRAIN_PARAMS['train_data_path'], 'rb') as f:
        dataset = pickle.load(f)
    mesh_ids = list(dataset.keys())

    # 2. 初始化
    print("正在初始化模型...")
    agent = PS_DQNAgent(input_dim=RL_PARAMS['input_dim'], hidden_dim1=RL_PARAMS['hidden_dim1'],
                        hidden_dim2=RL_PARAMS['hidden_dim2'], gcn_output_dim=RL_PARAMS['gcn_output_dim'],
                        lr=RL_PARAMS['lr'], gamma=RL_PARAMS['gamma'],
                        epsilon_start=RL_PARAMS['epsilon_start'], epsilon_min=RL_PARAMS['epsilon_min'],
                        epsilon_decay=RL_PARAMS['epsilon_decay'], memory_size=RL_PARAMS['memory_size'],
                        batch_size=RL_PARAMS['batch_size'], device=TRAIN_PARAMS['device'])
    env = NetworkEnv(dataset, BS_CONFIG, REWARD_PARAMS['w_energy_saving'], 
                     REWARD_PARAMS['qos_alpha'], REWARD_PARAMS['qos_beta'],
                     REWARD_PARAMS['w_drop'], REWARD_PARAMS['global_scale'])
    
    print(f"开始训练... 共 {len(mesh_ids)} 个 Mesh, 训练 {TRAIN_PARAMS['num_epochs']} 轮 (Epochs)")
    
    total_step = 0 
    # ==========================================
    # 初始化记录容器
    # ==========================================
    history = {
        'rewards': [],
        'dropped_mbps': [],
        'power_kw': [],
        'epsilon': []
    }

    # 追踪最佳模型
    best_reward = -float('inf') # 初始化为负无穷
    best_epoch = -1

    for epoch in range(TRAIN_PARAMS['num_epochs']):
        np.random.shuffle(mesh_ids)
        epoch_reward = 0 
        epoch_dropped = 0
        epoch_power = 0
        
        for i, mesh_id in enumerate(mesh_ids):
            # reset 返回两个值，该时刻的流量数据和邻接矩阵
            features, adj = env.reset(mesh_id)
            
            total_reward = 0
            done = False

            # 临时统计单个 Mesh 的掉线和能耗
            mesh_dropped = 0
            mesh_power = 0
            steps = 0
            
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
                total_reward += reward

                # 累加 Info 数据
                mesh_dropped += info['dropped_mbps']
                mesh_power += info['save_power_w']
                steps += 1
            
            epoch_reward += total_reward
            epoch_dropped += (mesh_dropped / steps) 
            epoch_power += (mesh_power / steps)
            total_step += 1

            agent.decay_epsilon()

            # 打印进度
            if i % 20 == 0: 
                agent.update_target()
                print(f"Epoch {epoch+1}/{TRAIN_PARAMS['num_epochs']} | Progress {i}/{len(mesh_ids)} | "
                      f"Mesh {mesh_id:<3} | Rw: {total_reward:.1f} | Sp: {info['save_power_w']:.1f} | "
                      f"Drop: {info['dropped_mbps']:.1f} | Eps: {agent.epsilon:.3f}")

        avg_reward = epoch_reward / len(mesh_ids)
        avg_dropped = epoch_dropped / len(mesh_ids)
        avg_power = epoch_power / len(mesh_ids)
        
        history['rewards'].append(avg_reward)
        history['dropped_mbps'].append(avg_dropped)
        history['power_kw'].append(avg_power)
        history['epsilon'].append(agent.epsilon)

        print(f"✨ Epoch {epoch+1} 完成! 平均奖励: {avg_reward:.2f}\n")

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

    final_best_model_path = "未找到最佳模型"
    
    # 原始最佳模型路径 (在循环中被反复覆盖的那个文件)
    original_best_path = os.path.join(run_dir, 'best_model.pth')
    
    if os.path.exists(original_best_path) and best_epoch != -1:
        # 构造新名字: best_model_epoch_159_rw_-30.53.pth
        new_best_name = f'best_model_epoch_{best_epoch}_rw_{best_reward:.2f}.pth'
        new_best_path = os.path.join(run_dir, new_best_name)
        
        try:
            os.rename(original_best_path, new_best_path)
            final_best_model_path = new_best_path
            print(f"✨ 最佳模型已重命名为: {new_best_name}")
        except OSError as e:
            print(f"⚠️ 重命名失败: {e}")
            final_best_model_path = original_best_path # 回退到旧名字
    else:
        # 如果训练没跑起来，或者逻辑有问题导致没生成 best_model
        final_best_model_path = original_best_path

        # 保存最终模型
    final_model_path = os.path.join(run_dir, 'final_model.pth')
    torch.save(agent.policy_net.state_dict(), final_model_path)
    print(f"最终模型已保存至: {final_model_path}")

    # 【新增 3】 输出最终总结
    print("="*50)
    print(f"训练全部结束!")
    print(f"最佳表现 Epoch: {best_epoch}")
    print(f"最高平均 Reward: {best_reward:.4f}")
    print(f"最佳模型位置: {final_best_model_path}")
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
        "TRAIN_PARAMS": TRAIN_PARAMS,
        "RL_PARAMS": RL_PARAMS,
        "REWARD_PARAMS": REWARD_PARAMS,
        "BS_CONFIG": BS_CONFIG
    }
    utils.save_config_to_json(run_dir, all_configs)
    
    
    main(run_dir)
    
    total_training_time = time.time() - total_start_time
    avg_epoch_time = total_training_time/TRAIN_PARAMS['num_epochs']
    print(f"⏱️  训练总耗时: {utils.format_duration(total_training_time)}")
    print(f"⏱️  平均每轮耗时: {utils.format_duration(avg_epoch_time)}")