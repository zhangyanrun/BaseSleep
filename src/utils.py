import json
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import time

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8') # 追加模式，指定编码防止乱码

    def write(self, message):
        # 同时写入终端和文件
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 立即刷新缓冲区，确保实时写入

    def flush(self):
        # 必须实现 flush 方法，适配 Python 的流操作接口
        self.terminal.flush()
        self.log.flush()

class MetricTracker:
    """
    评估指标计算器。
    负责在测试循环中累加数据，并计算最终的 ESR, DropRate, SleepRatio, PowerEfficiency。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_baseline_energy_w = 0.0
        self.total_ai_energy_w = 0.0
        self.total_traffic_demand_mbps = 0.0
        self.total_dropped_mbps = 0.0
        
        self.total_steps = 0
        self.total_bs_count = 0      # 累计遍历过的基站总数 (用于分母)
        self.total_sleep_count = 0   # 累计休眠的基站总数 (用于分子)

    def update(self, info, actions):
        """
        在每个 Step 调用一次，累加数据
        info: env.step 返回的 info 字典
        actions: agent 输出的动作数组 (0=休眠, 1=激活)
        """
        # 1. 累加能耗
        self.total_baseline_energy_w += info['baseline_total_w']
        self.total_ai_energy_w += info['actual_total_w']
        
        # 2. 累加流量
        self.total_traffic_demand_mbps += info['total_demand_mbps']
        self.total_dropped_mbps += info['dropped_mbps']
        
        # 3. 累加休眠统计
        # actions: 1=Active, 0=Sleep
        num_bs = len(actions)
        num_active = np.sum(actions)
        num_sleep = num_bs - num_active
        
        self.total_bs_count += num_bs
        self.total_sleep_count += num_sleep
        
        self.total_steps += 1

    def report(self):
        """
        计算并返回最终的四大指标
        """
        # 1. 节能率 (ESR)
        # (基准 - AI) / 基准
        saved_energy = self.total_baseline_energy_w - self.total_ai_energy_w
        if self.total_baseline_energy_w > 0:
            esr = (saved_energy / self.total_baseline_energy_w) * 100
        else:
            esr = 0.0

        # 2. 掉线率 (Drop Rate)
        # 掉线 / 总需求
        if self.total_traffic_demand_mbps > 0:
            drop_rate = (self.total_dropped_mbps / self.total_traffic_demand_mbps) * 100
        else:
            drop_rate = 0.0

        # 3. 平均休眠率 (Sleeping Ratio)
        # 累计休眠人次 / 累计总人次
        if self.total_bs_count > 0:
            sleep_ratio = (self.total_sleep_count / self.total_bs_count) * 100
        else:
            sleep_ratio = 0.0

        # 4. 能效比 (Power Efficiency)
        # 承载流量 (Mbps) / 消耗电能 (kW)
        # 承载流量 = 总需求 - 掉线
        carried_traffic = self.total_traffic_demand_mbps - self.total_dropped_mbps
        # 瓦转千瓦
        total_ai_kw = self.total_ai_energy_w / 1000.0
        
        if total_ai_kw > 0:
            power_efficiency = carried_traffic / total_ai_kw
        else:
            power_efficiency = 0.0

        return {
            "ESR (%)": esr,
            "Drop Rate (%)": drop_rate,
            "Sleep Ratio (%)": sleep_ratio,
            "Power Efficiency (Mbps/kW)": power_efficiency,
            # 附带一些绝对值数据方便查阅
            "Total Saved Energy (kW)": saved_energy / 1000.0,
            "Total Dropped (Mb)": self.total_dropped_mbps
        }



def save_config_to_json(save_dir, config_dict):# 将配置字典保存为 JSON 文件

    json_path = os.path.join(save_dir, 'hyperparameters.json')
    
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)
        print(f"超参数已保存至: {json_path}")
    except TypeError as e:
        print(f"保存 JSON 失败 (可能有不可序列化的对象): {e}")

def plot_learning_curve(history, save_dir):
    """
    绘制训练曲线并保存到指定目录
    history: 数据字典
    save_dir: 图片保存的文件夹路径 (即 timestamp 文件夹)
    """
    # 这里的 save_dir 已经是 create 好的 timestamp 文件夹了
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['rewards']) + 1)
    
    # 辅助函数: 滑动平均
    def moving_average(data, window_size=5):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    plt.style.use('default')

    # --- 图 1: Reward ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['rewards'], alpha=0.3, color='gray', label='Raw')
    smooth_rw = moving_average(history['rewards'])
    plt.plot(epochs[:len(smooth_rw)], smooth_rw, color='blue', linewidth=2, label='Smoothed')
    plt.title('Average Reward')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'curve_reward.png'), dpi=300)
    plt.close()

    # --- 图 2: Drop ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['dropped_mbps'], color='red', alpha=0.8)
    plt.title('Average Dropped Traffic')
    plt.xlabel('Epoch')
    plt.ylabel('Mbps')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'curve_dropped.png'), dpi=300)
    plt.close()

    # --- 图 3: Power ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['power_kw'], color='green', alpha=0.8)
    plt.title('Average Saved Power Consumption')
    plt.xlabel('Epoch')
    plt.ylabel('kW')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'curve_power.png'), dpi=300)
    plt.close()

    # --- 图 4: Epsilon ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['epsilon'], color='orange', linestyle='--')
    plt.title('Epsilon Decay')
    plt.xlabel('Epoch')
    plt.ylabel('Epsilon')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'curve_epsilon.png'), dpi=300)
    plt.close()

    print(f"可视化图片已保存至: {save_dir}")

def plot_sample_mesh(history, save_dir, mesh_id):
    """
    绘制单个 Mesh 在测试过程中的指标变化
    history: 包含 'traffic', 'power_base', 'power_ai', 'active_rate' 的字典
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    time_steps = range(len(history['traffic']))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # --- 左轴: 功率对比 ---
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Power Consumption (W)', color='black')
    
    # 画基准能耗 (虚线)
    l1 = ax1.plot(time_steps, history['power_base'], 'k--', label='Baseline (All-On)', alpha=0.5)
    # 画 AI 能耗 (实线)
    l2 = ax1.plot(time_steps, history['power_ai'], 'g-', label='AI Agent', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='black')
    
    # 填充节省区域
    ax1.fill_between(time_steps, history['power_ai'], history['power_base'], color='green', alpha=0.1, label='Saved Energy')

    # --- 右轴: 流量负载 ---
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Traffic Load Index (Sum)', color='blue')
    l3 = ax2.plot(time_steps, history['traffic'], 'b:', label='Traffic Demand', alpha=0.4)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # 合并图例
    lines = l1 + l2 + l3
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='upper left')
    
    plt.title(f'Test Evaluation: Mesh {mesh_id}\nEnergy Saving Visualization')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'test_vis_mesh_{mesh_id}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 可视化图表已保存至: {save_path}")

def plot_test_distribution(mesh_metrics, save_dir):
    """
    绘制整个测试集的性能分布图
    mesh_metrics: 列表，每个元素是一个字典，包含单个 Mesh 的指标
                  [{'mesh_id':.., 'esr':.., 'drop_rate':.., 'avg_load':..}, ...]
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 提取数据列
    esrs = [m['esr'] for m in mesh_metrics]
    drops = [m['drop_rate'] for m in mesh_metrics]
    loads = [m['avg_load'] for m in mesh_metrics]
    # 避免除零或无效值
    esrs = np.array(esrs)
    drops = np.array(drops)
    loads = np.array(loads)

    plt.style.use('default')
    
    # ==========================================
    # 图 1: 节能率分布直方图 (ESR Histogram)
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.hist(esrs, bins=20, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(esrs), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(esrs):.2f}%')
    plt.axvline(np.median(esrs), color='blue', linestyle='dashed', linewidth=2, label=f'Median: {np.median(esrs):.2f}%')
    
    plt.title('Distribution of Energy Saving Ratio (ESR) across all Meshes')
    plt.xlabel('Energy Saving Ratio (%)')
    plt.ylabel('Number of Meshes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'dist_esr_hist.png'), dpi=300)
    plt.close()

    # ==========================================
    # 图 2: 掉线率分布箱线图 (Drop Rate Boxplot)
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.boxplot(drops, vert=True, patch_artist=True, boxprops=dict(facecolor="red", color="black", alpha=0.6))
    plt.title('Distribution of Drop Rate')
    plt.ylabel('Drop Rate (%)')
    plt.grid(True, alpha=0.3)
    
    # 在图上标注有多少个 Mesh 是 0 掉线
    zero_drop_count = np.sum(drops == 0)
    plt.text(0.95, 0.95, f'{zero_drop_count}/{len(drops)} Meshes have 0% Drop', 
             transform=plt.gca().transAxes, ha='right', va='top', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(save_dir, 'dist_drop_boxplot.png'), dpi=300)
    plt.close()

    # ==========================================
    # 图 3: 负载 vs 节能率 散点图 (Load vs ESR)
    # ==========================================
    # 这张图非常有意义，它能回答“是不是只有负载低的时候才能节能？”
    plt.figure(figsize=(10, 6))
    plt.scatter(loads, esrs, c='blue', alpha=0.6, edgecolors='w', s=60)
    
    # 拟合一条趋势线
    if len(loads) > 1:
        m, b = np.polyfit(loads, esrs, 1)
        plt.plot(loads, m*loads + b, color='red', linestyle='--', alpha=0.8, label=f'Trend')
    
    plt.title('Correlation: Traffic Load vs Energy Saving')
    plt.xlabel('Average Traffic Load Ratio (0-1)')
    plt.ylabel('Energy Saving Ratio (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'scatter_load_vs_esr.png'), dpi=300)
    plt.close()

    print(f"📊 全局分布可视化图表已保存至: {save_dir}")

def pad_mesh_data(dataset):
    """
    遍历数据集，找出最大基站数 N_max，并对所有数据进行 Padding。
    返回: 填充后的数据字典, N_max
    """
    # 1. 找出全局最大基站数
    max_bs_count = 0
    for mesh_id in dataset:
        num_bs = len(dataset[mesh_id]['bs_ids'])
        if num_bs > max_bs_count:
            max_bs_count = num_bs
    
    print(f"⚡ 全局最大基站数 (N_max) 检测为: {max_bs_count}")
    
    processed_data = {}
    
    for mesh_id, data in dataset.items():
        # 原始数据
        # traffic shape: [Time, N, 1]
        raw_traffic = data['traffic_tensor']
        # adj shape: [N, N]
        raw_adj = data['adj_matrix']
        # static_type shape: [N] (我们需要保存类型索引以便恢复参数)
        # 假设 data['static_info']['Type'] 是字符串，我们在 Env 里处理映射
        
        T, N, F = raw_traffic.shape
        
        # --- Padding ---
        # 1. Traffic: [T, N_max, F]
        padded_traffic = np.zeros((T, max_bs_count, F))
        padded_traffic[:, :N, :] = raw_traffic
        
        # 2. Adj: [N_max, N_max]
        padded_adj = np.zeros((max_bs_count, max_bs_count))
        padded_adj[:N, :N] = raw_adj
        
        # 3. Mask: [N_max] (1=Real, 0=Fake)
        mask = np.zeros(max_bs_count)
        mask[:N] = 1.0
        
        # 存回字典
        processed_data[mesh_id] = {
            'traffic': padded_traffic,
            'adj': padded_adj,
            'mask': mask,
            'real_n': N,
            'static_info': data['static_info'], # 原始 DF
            'bs_ids': data['bs_ids']
        }
    
    print("数据填充成功")
        
    return processed_data, max_bs_count


# 【新增辅助函数】用于把秒数转成 "Xh Ym Zs" 格式
def format_duration(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{int(h)}h {int(m)}m {int(s)}s"
    return f"{int(m)}m {int(s)}s"