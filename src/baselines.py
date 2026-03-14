import numpy as np
import random

class AllOnAgent:
    """
    全开模式策略：无论状态如何，所有基站始终保持开启 (Action = 1)。
    用于评估系统的性能上限 (QoS 最好) 和能耗下限 (最费电)。
    """
    def __init__(self, num_actions=2):
        self.num_actions = num_actions

    def select_actions(self, node_features, adj):
        """
        node_features: [N, F]
        adj: [N, N]
        返回: [N] 大小的全 1 数组
        """
        num_nodes = node_features.shape[0]
        # 全 1 代表所有基站 Active
        return np.ones(num_nodes, dtype=int)

class RandomAgent:
    """
    随机策略：每个基站以 50% 概率随机开启或休眠。
    用于评估算法的下限，证明 DRL 确实学到了东西 (如果 DRL 比随机还差，那就是失败)。
    """
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def select_actions(self, node_features, adj):
        """
        返回: [N] 大小的随机 0/1 数组
        """
        num_nodes = node_features.shape[0]
        # 生成 0 或 1 的随机整数
        return np.random.randint(0, 2, size=num_nodes)

class ReactiveAgent:
    """
    对比算法 A：无预测纯被动模式 (Reactive)
    逻辑：只看到当下的流量 (Load_t)，并以此决定下一时刻的开关机。
    后果：永远慢 5 分钟。遇到流量突发时必定导致 QoS 断崖式下跌。
    """
    def __init__(self, bs_config):
        self.bs_config = bs_config
        self.idx_to_cap = {0: bs_config['4G_macro']['capacity'], 1: bs_config['5G_macro']['capacity'],
                           2: bs_config['4G_micro']['capacity'], 3: bs_config['5G_micro']['capacity']}
        self.idx_to_pzero = {0: bs_config['4G_macro']['p_zero'], 1: bs_config['5G_macro']['p_zero'],
                             2: bs_config['4G_micro']['p_zero'], 3: bs_config['5G_micro']['p_zero']}
        self.idx_to_slope = {0: bs_config['4G_macro']['slope'], 1: bs_config['5G_macro']['slope'],
                             2: bs_config['4G_micro']['slope'], 3: bs_config['5G_micro']['slope']}

    def select_actions(self, node_features, adj):
        num_nodes = node_features.shape[0]
        
        # 【核心差异】：只取第 0 列，也就是时刻 t 的真实负载 Load_t
        current_load_ratios = node_features[:, 0] 
        types_onehot = node_features[:, 2:6]

        caps = np.zeros(num_nodes)
        efficiencies = np.zeros(num_nodes) 
        
        for i in range(num_nodes):
            t_idx = np.argmax(types_onehot[i])
            caps[i] = self.idx_to_cap[t_idx]
            max_power = self.idx_to_pzero[t_idx] + self.idx_to_slope[t_idx]
            efficiencies[i] = caps[i] / max_power if max_power > 0 else 0

        # 以 t 时刻的负载去预估需要的总容量
        total_demand_mbps = np.sum(current_load_ratios * caps)
        
        # 按照能效从高到低排序，依次开启，直到容量满足需求
        sorted_indices = np.argsort(efficiencies)[::-1]
        
        actions = np.zeros(num_nodes, dtype=int)
        accumulated_cap = 0.0
        
        for idx in sorted_indices:
            actions[idx] = 1
            accumulated_cap += caps[idx]
            if accumulated_cap >= total_demand_mbps:
                break
                
        return actions

class MiLSFAgent:
    """
    复现论文: Minimum Load Sleep First (MiLSF)
    适配版: 基于宏观流量与拓扑邻接矩阵
    """
    def __init__(self, bs_config):
        self.bs_config = bs_config
        
        # 建立 One-Hot 索引到物理参数的映射 (严格对应 config.py 中的 TYPE_TO_INDEX)
        # 0: 4G_macro, 1: 5G_macro, 2: 4G_micro, 3: 5G_micro
        self.idx_to_cap = {
            0: bs_config['4G_macro']['capacity'],
            1: bs_config['5G_macro']['capacity'],
            2: bs_config['4G_micro']['capacity'],
            3: bs_config['5G_micro']['capacity']
        }
        # 区分宏基站 (MaBS) 和 微基站 (MiBS)
        self.idx_to_is_macro = {0: True, 1: True, 2: False, 3: False}

    def select_actions(self, node_features, adj):
        """
        node_features: [N, 6] -> [Load_t, Load_t+1, Type_0, Type_1, Type_2, Type_3]
        adj: [N, N]
        """
        num_nodes = node_features.shape[0]
        # 初始状态：所有基站全开 (MaBS 永远不关)
        actions = np.ones(num_nodes, dtype=int) 

        # 1. 解析当前状态特征
        load_ratios = node_features[:, 0]  # 当前负载率
        types_onehot = node_features[:, 2:6] # 基站类型 One-Hot

        caps = np.zeros(num_nodes)
        is_macro = np.zeros(num_nodes, dtype=bool)

        for i in range(num_nodes):
            t_idx = np.argmax(types_onehot[i]) # 反推真实类型索引
            caps[i] = self.idx_to_cap[t_idx]
            is_macro[i] = self.idx_to_is_macro[t_idx]

        # 计算当前的真实需求流量 (Mbps)
        current_load_mbps = load_ratios * caps

        # 2. 筛选出 MiBS 并按负载从小到大排序 (核心逻辑 1)
        mibs_indices = np.where(~is_macro)[0]
        mibs_sorted = mibs_indices[np.argsort(current_load_mbps[mibs_indices])]

        # 3. 模拟流量卸载，尝试让 MiBS 依次休眠
        temp_load_mbps = np.copy(current_load_mbps)

        for i in mibs_sorted:
            demand = temp_load_mbps[i]
            
            # 如果本身就是空载，直接休眠
            if demand <= 0.001: 
                actions[i] = 0
                continue

            # 寻找可以接盘的“激活”且“非自身”的邻居
            neighbors = np.where(adj[i] == 1)[0]
            active_neighbors = [n for n in neighbors if actions[n] == 1 and n != i]

            if not active_neighbors:
                continue # 没有开机的邻居，不能休眠，硬扛

            # 将邻居分为 MaBS 和 MiBS 以满足论文的优先级策略
            mabs_neighbors = [n for n in active_neighbors if is_macro[n]]
            mibs_neighbors = [n for n in active_neighbors if not is_macro[n]]

            # 论文策略: 优先给 MaBS。代码实现: 将 MaBS 排在前面，按剩余容量降序 (保证吞吐)
            mabs_neighbors.sort(key=lambda n: caps[n] - temp_load_mbps[n], reverse=True)
            
            # 论文策略: 如果没有 MaBS，给负载最高的 MiBS。代码实现: MiBS 按当前负载降序排列
            mibs_neighbors.sort(key=lambda n: temp_load_mbps[n], reverse=True)

            # 合并接盘顺位
            sorted_candidates = mabs_neighbors + mibs_neighbors

            # 评估接盘侠们的总剩余容量是否能吃下这波 demand
            total_available = sum([max(0, caps[n] - temp_load_mbps[n]) for n in sorted_candidates])

            if total_available >= demand:
                # 可以休眠！模拟流量转移，更新 temp_load_mbps
                actions[i] = 0
                rem_demand = demand
                temp_load_mbps[i] = 0

                for n in sorted_candidates:
                    if rem_demand <= 0:
                        break
                    room = max(0, caps[n] - temp_load_mbps[n])
                    if room > 0:
                        flow = min(rem_demand, room)
                        temp_load_mbps[n] += flow
                        rem_demand -= flow
            else:
                # 算过了，接不住，该 MiBS 必须乖乖保持开机
                pass

        return actions

class DDSSAgent:
    """
    复现论文: A Data-driven Base Station Sleeping Strategy Based on Traffic Prediction
    简称: DDSS (Data-Driven Sleeping Strategy)
    核心逻辑: 基于区域总流量预测，计算需要的基站总容量，开启最少数量的基站以满足总需求。
    """
    def __init__(self, bs_config):
        self.bs_config = bs_config
        # 建立 One-Hot 索引到物理容量的映射
        self.idx_to_cap = {
            0: bs_config['4G_macro']['capacity'],
            1: bs_config['5G_macro']['capacity'],
            2: bs_config['4G_micro']['capacity'],
            3: bs_config['5G_micro']['capacity']
        }

    def select_actions(self, node_features, adj):
        """
        注意：DDSS 是区域级统筹策略，完全忽略 adj 矩阵，只看全局需求和全局容量。
        node_features: [N, 6] -> [Load_t, Load_t+1, Type_0, Type_1, Type_2, Type_3]
        """
        num_nodes = node_features.shape[0]
        actions = np.zeros(num_nodes, dtype=int) 

        # 1. 获取所有基站的预测负载 (node_features 第1列是 Load_t+1) 和类型
        predicted_load_ratios = node_features[:, 1] 
        types_onehot = node_features[:, 2:6]

        caps = np.zeros(num_nodes)
        for i in range(num_nodes):
            t_idx = np.argmax(types_onehot[i])
            caps[i] = self.idx_to_cap[t_idx]

        # 2. 计算整个 Mesh (区域) 的未来总流量需求
        predicted_demand_mbps = predicted_load_ratios * caps
        total_mesh_demand = np.sum(predicted_demand_mbps)

        # 3. 决定开启哪些基站 
        # 论文逻辑: 开启最优数量的基站，使得容量刚好覆盖总需求。
        # 在你的环境中落地时，为了尽量减少掉线，我们优先开启“预测流量本身就大”的基站，直到总容量达标。
        sorted_indices = np.argsort(predicted_demand_mbps)[::-1]
        
        current_capacity = 0.0
        # 加一个 5% 的安全冗余系数，模仿论文中保证 QoS 的容量建模边界
        target_capacity = total_mesh_demand * 1.05 

        for idx in sorted_indices:
            actions[idx] = 1
            current_capacity += caps[idx]
            if current_capacity >= target_capacity:
                break # 容量达标，剩下的基站全部休眠 (actions 保持为 0)
                
        # 兜底机制：如果全网没流量，也要留一个基站开着待机，防止网络彻底断联
        if np.sum(actions) == 0 and num_nodes > 0:
            actions[sorted_indices[0]] = 1

        return actions

class DFSCSAgent:
    """
    复现论文: Collaborative base station sleeping solution design in heterogeneous cellular network
    核心算法: DFSCS (Depth-First Search Collaborative Sleeping) + SBSUF (Single Base Station User Transfer)
    """
    def __init__(self, bs_config):
        self.bs_config = bs_config
        self.idx_to_cap = {
            0: bs_config['4G_macro']['capacity'],
            1: bs_config['5G_macro']['capacity'],
            2: bs_config['4G_micro']['capacity'],
            3: bs_config['5G_micro']['capacity']
        }
        self.idx_to_is_macro = {0: True, 1: True, 2: False, 3: False}

    def select_actions(self, node_features, adj):
        """
        node_features: [N, 6]
        adj: [N, N]
        """
        num_nodes = node_features.shape[0]
        
        # 初始状态：所有基站全部开启
        actions = np.ones(num_nodes, dtype=int) 

        load_ratios = node_features[:, 0]
        types_onehot = node_features[:, 2:6]

        caps = np.zeros(num_nodes)
        is_macro = np.zeros(num_nodes, dtype=bool)

        for i in range(num_nodes):
            t_idx = np.argmax(types_onehot[i])
            caps[i] = self.idx_to_cap[t_idx]
            is_macro[i] = self.idx_to_is_macro[t_idx]

        initial_load_mbps = load_ratios * caps
        
        # 1. 提取所有微基站 (Micro BS)
        mibs_indices = np.where(~is_macro)[0]
        
        # 论文要求按距离宏基站的距离排序。
        # 在我们的图拓扑中，我们可以用初始负载或者邻居数量作为启发式排序。
        # 这里为了最大化节能且贴合 DFS 找最大深度，我们将负载从小到大排序优先尝试休眠
        mibs_sorted = mibs_indices[np.argsort(initial_load_mbps[mibs_indices])]

        # --- SBSUF 可行性检查函数 ---
        def check_feasibility(current_actions):
            """
            模拟当前 actions 状态下的流量转移，如果不掉线(宏基站未超载)，则返回 True
            """
            temp_load = np.copy(initial_load_mbps)
            
            # 遍历所有被设置为休眠的微基站
            sleeping_mibs = np.where(current_actions == 0)[0]
            for i in sleeping_mibs:
                demand = temp_load[i]
                if demand <= 0:
                    continue
                    
                # 寻找活着的邻居
                neighbors = np.where(adj[i] == 1)[0]
                active_neighbors = [n for n in neighbors if current_actions[n] == 1 and n != i]
                
                # 区分相邻的活着的微基站和宏基站
                active_micro_neighbors = [n for n in active_neighbors if not is_macro[n]]
                active_macro_neighbors = [n for n in active_neighbors if is_macro[n]]
                
                # SBSUF 第一步：优先转移给相邻的可用微基站 (Micro BS)
                for n in active_micro_neighbors:
                    if demand <= 0:
                        break
                    room = max(0, caps[n] - temp_load[n])
                    if room > 0:
                        flow = min(demand, room)
                        temp_load[n] += flow
                        demand -= flow
                        
                # SBSUF 第二步：溢出的用户 (Redundant overflow) 转移给宏基站 (Macro BS)
                if demand > 0:
                    for n in active_macro_neighbors:
                        if demand <= 0:
                            break
                        room = max(0, caps[n] - temp_load[n]) # 宏基站的剩余容量 (Nm*)
                        if room > 0:
                            flow = min(demand, room)
                            temp_load[n] += flow
                            demand -= flow
                            
                # 如果经历了微基站和宏基站的转移，demand 还有剩余，说明超过了宏基站阈值 (Nm > Nm*)
                if demand > 0.001: 
                    return False # 方案不可行
                    
            return True

        # 2. 核心 DFSCS 逻辑：逐个尝试休眠并回溯
        for i in mibs_sorted:
            # 尝试深度搜索：让基站 i 休眠 (dp = dp + 1)
            actions[i] = 0 
            
            # 验证 SBSUF 转移约束
            is_feasible = check_feasibility(actions)
            
            if not is_feasible:
                # 超过宏基站阈值 (Nm > Nm*)，执行回溯 (dp = dp - 1)，恢复为开启状态
                actions[i] = 1 
                
        return actions

class REDEEMAgent:
    """
    复现论文: Mitigating Energy Consumption in Heterogeneous Mobile Networks Through Data-Driven Optimization
    核心机制: 能效画像 (Energy Efficiency Profiling) + 5G 流量主动卸载 (5G Traffic Offloading)
    """
    def __init__(self, bs_config):
        self.bs_config = bs_config
        
        # 映射表：0: 4G_macro, 1: 5G_macro, 2: 4G_micro, 3: 5G_micro
        self.idx_to_cap = {}
        self.idx_to_pzero = {}
        self.idx_to_slope = {}
        self.idx_to_psleep = {}
        self.idx_to_is_4g = {0: True, 1: False, 2: True, 3: False}
        
        keys = {0: '4G_macro', 1: '5G_macro', 2: '4G_micro', 3: '5G_micro'}
        for idx, key in keys.items():
            conf = bs_config[key]
            self.idx_to_cap[idx] = conf['capacity']
            self.idx_to_pzero[idx] = conf['p_zero']
            self.idx_to_slope[idx] = conf['slope']
            self.idx_to_psleep[idx] = conf['p_sleep']

    def select_actions(self, node_features, adj):
        num_nodes = node_features.shape[0]
        
        # 1. 解析当前状态特征 (使用 Load_t+1 预测流量实现 Proactive Control)
        predicted_load_ratios = node_features[:, 1] 
        types_onehot = node_features[:, 2:6]

        caps = np.zeros(num_nodes)
        p_zeros = np.zeros(num_nodes)
        slopes = np.zeros(num_nodes)
        p_sleeps = np.zeros(num_nodes)
        is_4g = np.zeros(num_nodes, dtype=bool)
        
        # 理论能效 = 最大容量 / 最大功耗
        efficiencies = np.zeros(num_nodes) 

        for i in range(num_nodes):
            t_idx = np.argmax(types_onehot[i])
            caps[i] = self.idx_to_cap[t_idx]
            p_zeros[i] = self.idx_to_pzero[t_idx]
            slopes[i] = self.idx_to_slope[t_idx]
            p_sleeps[i] = self.idx_to_psleep[t_idx]
            is_4g[i] = self.idx_to_is_4g[t_idx]
            
            max_power = p_zeros[i] + slopes[i]
            efficiencies[i] = caps[i] / max_power if max_power > 0 else 0

        predicted_demand_mbps = predicted_load_ratios * caps
        
        # 划分 4G 和 5G 集合，并按能效从高到低排序 (Algorithm 2)
        idx_4g = np.where(is_4g)[0]
        idx_5g = np.where(~is_4g)[0]
        
        sorted_4g = idx_4g[np.argsort(efficiencies[idx_4g])[::-1]]
        sorted_5g = idx_5g[np.argsort(efficiencies[idx_5g])[::-1]]
        
        total_4g_cap = np.sum(caps[idx_4g])
        
        # --- 内部函数：Mesh 级能效画像与分配 (Algorithm 2) ---
        def profile(L_4G, L_5G):
            temp_actions = np.zeros(num_nodes, dtype=int)
            total_energy = 0.0
            
            # 优先填满高能效的 4G 基站
            rem_4G = L_4G
            for idx in sorted_4g:
                alloc = min(rem_4G, caps[idx])
                rem_4G -= alloc
                if alloc > 0.001:
                    temp_actions[idx] = 1
                    total_energy += p_zeros[idx] + slopes[idx] * (alloc / caps[idx])
                else:
                    total_energy += p_sleeps[idx]
                    
            # 优先填满高能效的 5G 基站
            rem_5G = L_5G
            for idx in sorted_5g:
                alloc = min(rem_5G, caps[idx])
                rem_5G -= alloc
                if alloc > 0.001:
                    temp_actions[idx] = 1
                    total_energy += p_zeros[idx] + slopes[idx] * (alloc / caps[idx])
                else:
                    total_energy += p_sleeps[idx]
                    
            return total_energy, temp_actions

        # 2. 初始化流量分配
        L_4G_current = np.sum(predicted_demand_mbps[idx_4g])
        L_5G_current = np.sum(predicted_demand_mbps[idx_5g])
        
        current_energy, best_actions = profile(L_4G_current, L_5G_current)
        current_efficiency = (L_4G_current + L_5G_current) / current_energy if current_energy > 0 else 0

        # 3. 5G 流量卸载核心逻辑 (Algorithm 3)
        # 为了最大化收益，我们优先尝试把“能效最差”的 5G 基站流量卸载掉
        for idx in reversed(sorted_5g):
            demand_i = predicted_demand_mbps[idx]
            if demand_i <= 0.001:
                continue
                
            # 如果它当前是开启的，并且 4G 整体剩余容量接得住它的流量
            if best_actions[idx] == 1 and (L_4G_current + demand_i) <= total_4g_cap:
                test_L_4G = L_4G_current + demand_i
                test_L_5G = L_5G_current - demand_i
                
                test_energy, test_actions = profile(test_L_4G, test_L_5G)
                test_efficiency = (test_L_4G + test_L_5G) / test_energy if test_energy > 0 else 0
                
                # 如果卸载后，整个 Mesh 的能效提升了，则采纳该卸载方案
                if test_efficiency >= current_efficiency:
                    L_4G_current = test_L_4G
                    L_5G_current = test_L_5G
                    current_efficiency = test_efficiency
                    best_actions = test_actions
                    predicted_demand_mbps[idx] = 0 # 该基站流量已被清空
                    
        return best_actions

class FullSleepAgent:
    """
    (可选) 全关模式：用于测试极端情况，理论上 QoS 会全崩。
    """
    def select_actions(self, node_features, adj):
        num_nodes = node_features.shape[0]
        return np.zeros(num_nodes, dtype=int)