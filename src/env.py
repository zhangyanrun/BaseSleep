import numpy as np
from src.config import PRIORITY_MAP, TYPE_TO_INDEX


class NetworkEnv:
    def __init__(self, processed_data, bs_config, w_energy_saving, qos_alpha, qos_beta, w_drop, global_scale):
        """
        processed_data: build_dataset 生成的数据字典
        bs_config: 物理参数配置字典
        """
        self.data = processed_data
        self.bs_config = bs_config
        self.w_energy_saving = w_energy_saving
        self.qos_alpha = qos_alpha
        self.qos_beta = qos_beta
        self.w_drop = w_drop
        self.global_scale = global_scale
        
        # 缓存当前环境状态
        self.current_mesh_id = None
        self.current_step = 0
        self.max_steps = 0
        
        # 缓存当前 Mesh 的物理属性
        self.real_n = 0
        self.current_types = None      # 字符串类型数组
        self.cap_arr = None            # 容量数组
        self.p_zero_arr = None         # 零载功耗数组
        self.slope_arr = None          # 斜率数组
        self.p_sleep_arr = None        # 休眠功耗数组
        self.priority_scores = None    # 优先级分数数组
        self.type_onehot = None        # 类型 One-Hot 矩阵
        self.current_adj = None

    def reset(self, mesh_id):
        """
        重置环境到指定 Mesh 的第 0 个时间步。
        """
        self.current_mesh_id = mesh_id
        self.mesh_data = self.data[mesh_id]
        self.current_step = 0
        
        # ====================================================
        # 动态计算 real_n，不再依赖字典 Key
        # ====================================================
        if 'bs_ids' in self.mesh_data:
            self.real_n = len(self.mesh_data['bs_ids'])
        elif 'static_info' in self.mesh_data:
            self.real_n = len(self.mesh_data['static_info'])
        else:
            self.real_n = self.mesh_data['traffic_tensor'].shape[1]
            
        self.max_steps = self.mesh_data['traffic_tensor'].shape[0] - 1
        
        # 2. 预加载物理参数 (只加载真实基站部分)
        # static_info 是 DataFrame
        types_series = self.mesh_data['static_info']['Type'].iloc[:self.real_n]
        self.current_types = types_series.values
        
        # 初始化物理参数数组
        self.cap_arr = np.zeros(self.real_n)
        self.p_zero_arr = np.zeros(self.real_n)
        self.slope_arr = np.zeros(self.real_n)
        self.p_sleep_arr = np.zeros(self.real_n)
        self.priority_scores = np.zeros(self.real_n)
        
        # 初始化 One-Hot 矩阵 (N, 4)
        self.type_onehot = np.zeros((self.real_n, 4))
        
        for i, t_str in enumerate(self.current_types):
            cfg = self.bs_config.get(t_str)
            self.cap_arr[i] = cfg.get('capacity')
            self.p_zero_arr[i] = cfg.get('p_zero')
            self.slope_arr[i] = cfg.get('slope')
            self.p_sleep_arr[i] = cfg.get('p_sleep')
            
            # 优先级分数
            self.priority_scores[i] = PRIORITY_MAP.get(t_str, 0)
            
            # One-Hot
            type_idx = TYPE_TO_INDEX.get(t_str)
            if type_idx is None:
                raise ValueError(f"未知基站类型: {t_str}")
            self.type_onehot[i, type_idx] = 1.0

        self.current_adj = self._calculate_directed_geo_adj(self.mesh_data['static_info'])

        # 返回初始状态
        return self._get_state()

    def _calculate_directed_geo_adj(self, static_info):
        """
        构建有向邻接矩阵。
        逻辑: Dist(A, B) + R_eff(A) < R_max(B) => Edge A->B
        """
        # 1. 准备数据 (只取真实基站)
        # static_info 是 DataFrame，确保按索引重置过
        real_n = self.real_n
        locs = static_info[['XLocation', 'YLocation']].values[:real_n] # [N, 2]
        types = static_info['Type'].values[:real_n]                    # [N]
        
        # 2. 向量化获取半径参数
        # 创建 N 维数组
        r_eff_arr = np.zeros(real_n)
        r_max_arr = np.zeros(real_n)
        
        # 查表 (这里用循环查表构建数组，因为类型数量很少，开销可忽略)
        for i, t_str in enumerate(types):
            cfg = self.bs_config.get(t_str)
            r_eff_arr[i] = cfg.get('radius_eff')
            r_max_arr[i] = cfg.get('radius_max')
            
        # 3. 计算距离矩阵 Dist[i, j]
        # 利用广播: [N, 1, 2] - [1, N, 2] = [N, N, 2]
        diff = locs[:, None, :] - locs[None, :, :] 
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1)) # [N, N]
        
        # 4. 应用判断逻辑 (Broadcasting)
        # 公式: dist[i, j] + r_eff[i] < r_max[j]
        # i 是源 (Source), j 是目标 (Target)
        # dist_matrix:    [N, N]
        # r_eff_arr[:, None]: [N, 1] (列向量，对应 i)
        # r_max_arr[None, :]: [1, N] (行向量，对应 j)
        
        condition_mask = (dist_matrix + r_eff_arr[:, None]) < r_max_arr[None, :]
        
        # 转为 float
        adj = condition_mask.astype(float)
        
        # 5. 自环 (Self-loop)
        # 总是允许自己处理自己的流量
        np.fill_diagonal(adj, 1.0)
        
        return adj

    def _get_state(self):
        """
        构建图数据状态。
        返回: node_features, adj_matrix
        """
        # 1. 节点特征 [Real
        # _N, 5] (Load + 4_OneHot)
        current_traffic = self.mesh_data['traffic_tensor'][self.current_step][:self.real_n]
        node_features = np.concatenate([current_traffic, self.type_onehot], axis=1)
        
        # 2. 邻接矩阵 [Real_N, Real_N]
        # adj = self._calculate_directed_geo_adj(self.mesh_data['static_info'])
        
        return node_features, self.current_adj


    def step(self, actions):
        """
        执行动作，进行基于优先级的流量卸载，计算奖励。
        """
        # 1. 还原真实流量需求 (Mbps)input norm_load 是 Load Ratio (0~1)
        norm_load = self.mesh_data['traffic_tensor'][self.current_step][:self.real_n, 0]
        traffic_demand_mbps = norm_load * self.cap_arr
        
        # 初始化状态统计
        actual_load_mbps = traffic_demand_mbps * actions

        #记录掉线的流量数据
        dropped_mbps = 0.0
        
        # ==========================================
        # 2. 核心逻辑: 基于优先级的流量卸载
        # ==========================================
        for i in range(self.real_n):
            if actions[i] == 1: 
                # 激活状态：处理自己的流量
                continue
            else: 
                # 休眠状态：寻找接盘侠
                # 筛选出激活的邻居
                demand = traffic_demand_mbps[i]
                neighbors_idx = np.where(self.current_adj[i] == 1)[0]
                active_neighbors = [n for n in neighbors_idx if actions[n] == 1]
                
                if len(active_neighbors) > 0:
                    # 策略: 优先给高分邻居 (4G Macro > ...)
                    sorted_neighbors = sorted(
                        active_neighbors, 
                        key=lambda x: self.priority_scores[x], 
                        reverse=True
                    )
                    
                    remaining_demand = demand
                    
                    # 贪心填坑 (Water-filling)
                    for target_idx in sorted_neighbors:
                        if remaining_demand <= 0:
                            break
                        
                        # 计算该邻居的剩余容量
                        cap = self.cap_arr[target_idx]
                        curr_load = actual_load_mbps[target_idx]
                        room = cap - curr_load
                        
                        if room > 0:
                            # 能塞多少塞多少
                            flow = min(remaining_demand, room)
                            actual_load_mbps[target_idx] += flow
                            remaining_demand -= flow
                    
                    # 如果还有剩的，强制塞给最好的
                    if remaining_demand > 0:
                        best_idx = sorted_neighbors[0]
                        actual_load_mbps[best_idx] += remaining_demand
                        
                else:
                    # 没邻居开机，直接掉线
                    dropped_mbps += demand

        # ==========================================
        # 3. 计算奖励 (Global Reward)
        # ==========================================
        # --- A. 计算 Baseline (默认全开机模式下的能耗) ---
        # 假设所有基站 action=1，负载就是原始输入的 norm_load
        # 注意 clip 防止原始数据里的异常值超过 1.0
        baseline_load_ratios = np.clip(norm_load, 0, 1.0)
        
        # 计算每个基站的基准功耗 (P_zero + Slope * Load)
        baseline_power_each = self.p_zero_arr + self.slope_arr * baseline_load_ratios
        baseline_total_w = np.sum(baseline_power_each)
        
        # --- B. 计算 Actual (Agent 调度后的能耗) ---
        current_load_ratios = np.divide(actual_load_mbps, self.cap_arr, 
                                      out=np.zeros_like(actual_load_mbps), 
                                      where=self.cap_arr!=0)
        
        # 激活基站用线性公式，休眠基站用 P_sleep
        power_active = self.p_zero_arr + self.slope_arr * np.clip(current_load_ratios, 0, 1.0)
        actual_total_w = np.sum(actions * power_active + (1 - actions) * self.p_sleep_arr)
        
        # --- C. 计算节能增益 (Positive Reward) ---
        # 这里的含义是：相比于全开机，我帮你省了多少电
        saved_power_w = baseline_total_w - actual_total_w
        saved_power_kw = saved_power_w / 1000
        
        r_saving = saved_power_kw * self.w_energy_saving
        
        # --- D. QoS 非线性惩罚 (保持之前的建议) ---        
        # 使用实际负载计算风险
        safe_load = np.clip(current_load_ratios, 0, 2.0) 
        qos_terms = self.qos_alpha * (np.exp(self.qos_beta * safe_load) - 1)
        p_congestion = np.sum(qos_terms)
        
        # --- E. 掉线惩罚 ---
        p_drop = dropped_mbps * self.w_drop
        
        # --- Total Reward ---
        reward = r_saving - p_congestion - p_drop
        reward = reward * self.global_scale
        
        # ==========================================
        # 4. 状态更新
        # ==========================================
        self.current_step += 1
        done = (self.current_step >= self.max_steps)
        
        if not done:
            # 正常获取下一个状态 (元组)
            next_features, next_adj = self._get_state()
        else:
            # 如果结束了，必须返回与 GCN 形状匹配的全 0 数据
            # Feature Dim = 5 (Load + 4_Type)
            next_features = np.zeros((self.real_n, 5)) 
            # Adj Dim = (N, N)
            next_adj = np.zeros((self.real_n, self.real_n))
            
        info = {
            'save_power_w': saved_power_w,
            'dropped_mbps': dropped_mbps,
            'baseline_total_w': baseline_total_w,
            'actual_total_w': actual_total_w,
            'total_demand_mbps': np.sum(traffic_demand_mbps)
        }
        
        # 【修改点】直接返回 5 个扁平的变量，不要用嵌套元组
        return next_features, next_adj, reward, done, info