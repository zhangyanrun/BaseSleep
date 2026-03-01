import json
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import time

class Logger(object):
    '''
    日志记录器和配置保存工具
    '''
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
    计算 ESR, DropRate, SleepRatio, PowerEfficiency, SwitchFreq
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

        self.total_switch_count = 0  
        self.prev_actions = None      # 用于计算切换频率

    def new_episode(self):
        """
        在新的一局（新的 Mesh）开始时调用，
        仅清空上一步的动作记录，但不清空累计的能耗和流量等全局指标。
        """
        self.prev_actions = None

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

        # 计算状态切换次数
        if self.prev_actions is not None:
            switch_count = np.sum(np.abs(actions - self.prev_actions))
            self.total_switch_count += switch_count
            
        self.prev_actions = np.copy(actions)
        
        self.total_steps += 1

    def report(self):
        """
        返回所有计算完毕的工程指标
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
        total_ai_kw = self.total_ai_energy_w / 1000.0
        if total_ai_kw > 0:
            power_efficiency = carried_traffic / total_ai_kw
        else:
            power_efficiency = 0.0

        #转换频率
        if self.total_steps > 1:
            bs_per_step = self.total_bs_count / self.total_steps 
            switch_freq = (self.total_switch_count / ((self.total_steps - 1) * bs_per_step)) * 100
        else:
            switch_freq = 0.0

        return {
            "ESR (%)": esr,
            "Drop Rate (%)": drop_rate,
            "Sleep Ratio (%)": sleep_ratio,
            "Power Efficiency (Mbps/kW)": power_efficiency,
            "Switching Frequency (%)": switch_freq,
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
    绘制训练阶段的学习曲线并保存到指定目录。
    支持自适应绘制，只要 history 字典里有该指标就会自动画出来。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['rewards']) + 1)
    
    # 辅助函数: 滑动平均，用于让曲线更平滑，更容易看出收敛趋势
    def moving_average(data, window_size=5):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    plt.style.use('default')

    # ==========================================
    # 图 1: 综合奖励曲线 (Reward)
    # 含义: 智能体的“考试总分”。反映了模型是否在学习。
    # 趋势: 应该稳步上升，最终在一个区间内震荡收敛。
    # ==========================================
    if 'rewards' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['rewards'], alpha=0.3, color='gray', label='Raw')
        smooth_rw = moving_average(history['rewards'])
        plt.plot(epochs[:len(smooth_rw)], smooth_rw, color='blue', linewidth=2, label='Smoothed')
        plt.title('Training: Average Reward per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'curve_1_reward.png'), dpi=300)
        plt.close()

    # ==========================================
    # 图 2: 掉线率曲线 (Drop Rate)
    # 含义: 服务质量(QoS)的核心指标。表示因为错误关机导致无法服务的流量比例。
    # 趋势: 应该在训练初期较高，随着模型变聪明，迅速下降并逼近 0%。
    # ==========================================
    if 'drop_rate' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['drop_rate'], color='red', alpha=0.8)
        plt.title('Training: QoS - Drop Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Drop Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'curve_2_dropped.png'), dpi=300)
        plt.close()

    # ==========================================
    # 图 3: 节能率曲线 (Energy Saving Ratio, ESR)
    # 含义: 经济效益的核心指标。表示相较于所有基站全开，AI 帮你省了百分之几的电。
    # 趋势: 通常会先随着掉线率一起下降（模型怕掉线不敢关机），然后在确保不掉线的前提下，慢慢上升并稳定。
    # ==========================================
    if 'esr' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['esr'], color='green', alpha=0.8)
        plt.title('Training: Economy - Energy Saving Ratio (ESR)')
        plt.xlabel('Epoch')
        plt.ylabel('ESR (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'curve_3_esr.png'), dpi=300)
        plt.close()

    # ==========================================
    # 图 4: 平均休眠率 (Sleep Ratio)
    # 含义: 宏观物理指标。表示在整个训练周期内，平均有多少比例的基站处于关闭状态。
    # 趋势: 反映了策略的激进程度，通常与 ESR 走势高度相关。
    # ==========================================
    if 'sleep_ratio' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['sleep_ratio'], color='purple', alpha=0.8)
        plt.title('Training: Physical - Sleep Ratio')
        plt.xlabel('Epoch')
        plt.ylabel('Sleep Ratio (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'curve_4_sleep_ratio.png'), dpi=300)
        plt.close()
        
    # ==========================================
    # 图 5: 开关频率 (Switching Frequency)
    # 含义: 硬件损耗指标。表示基站状态(0/1)发生切换的频繁程度。
    # 趋势: 越低越好。如果该值居高不下，说明模型在“反复横跳”，缺乏稳定性。
    # ==========================================
    if 'switch_freq' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['switch_freq'], color='brown', alpha=0.8)
        plt.title('Training: Hardware - Switching Frequency')
        plt.xlabel('Epoch')
        plt.ylabel('Switch Frequency (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'curve_5_switch_freq.png'), dpi=300)
        plt.close()

    # ==========================================
    # 图 6: Epsilon 衰减曲线
    # 含义: 算法自身的超参数监控。展示了模型从“随机探索(1.0)”到“利用经验(0.05)”的过程。
    # ==========================================
    if 'epsilon' in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history['epsilon'], color='orange', linestyle='--')
        plt.title('Algorithm: Epsilon Decay')
        plt.xlabel('Epoch')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'curve_6_epsilon.png'), dpi=300)
        plt.close()

    print(f"训练曲线可视化图片已保存至: {save_dir}")

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
    esrs = np.array([m.get('esr', 0) for m in mesh_metrics])
    drops = np.array([m.get('drop_rate', 0) for m in mesh_metrics])
    loads = np.array([m.get('avg_load', 0) for m in mesh_metrics])

    plt.style.use('default')
    
    # ==========================================
    # 图 1: 节能率分布直方图 (ESR Histogram)
    # 含义: 评估模型泛化能力。看看模型在不同的城市网格(Mesh)中，是不是都能稳定省电。
    # 期望表现: 柱子整体越靠右（省电越多）越好，且方差不要太大。
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
    
    plt.savefig(os.path.join(save_dir, 'dist_1_esr_hist.png'), dpi=300)
    plt.close()

    # ==========================================
    # 图 2: 掉线率分布箱线图 (Drop Rate Boxplot)
    # 含义: 评估极端崩溃风险。箱线图可以一眼看出有没有产生灾难性掉线的异常区域。
    # 期望表现: 整个箱体最好都被压扁在 0% 刻度线上。
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.boxplot(drops, vert=True, patch_artist=True, boxprops=dict(facecolor="red", color="black", alpha=0.6))
    plt.title('Distribution of Drop Rate (Lower is Better)')
    plt.ylabel('Drop Rate (%)')
    plt.grid(True, alpha=0.3)
    
    # 在图上标注有多少个 Mesh 是完美的 0 掉线
    zero_drop_count = np.sum(drops == 0)
    plt.text(0.95, 0.95, f'{zero_drop_count}/{len(drops)} Meshes have 0% Drop', 
             transform=plt.gca().transAxes, ha='right', va='top', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(save_dir, 'dist_2_drop_boxplot.png'), dpi=300)
    plt.close()

    # ==========================================
    # 图 3: 负载 vs 节能率 散点图 (Load vs ESR)
    # 含义: 策略的“智能程度”。证明节能不是碰运气，而是因为网络闲置。
    # 期望表现: 明显的负相关趋势线（向右下方倾斜）。即负载越低，网络越空闲，模型关停的基站越多，省电率越高。
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.scatter(loads, esrs, c='blue', alpha=0.6, edgecolors='w', s=60)
    
    # 拟合一条趋势线 (仅在有足够多数据点且不报错的情况下拟合)
    if len(loads) > 1 and np.var(loads) > 0:
        m, b = np.polyfit(loads, esrs, 1)
        plt.plot(loads, m*loads + b, color='red', linestyle='--', alpha=0.8, label=f'Trend Line')
    
    plt.title('Correlation: Traffic Load vs Energy Saving')
    plt.xlabel('Average Traffic Load Ratio (0-1)')
    plt.ylabel('Energy Saving Ratio (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(save_dir, 'dist_3_scatter_load_vs_esr.png'), dpi=300)
    plt.close()

    print(f"全局分布可视化图表已保存至: {save_dir}")

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