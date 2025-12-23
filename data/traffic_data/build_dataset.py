import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import pickle

# # ==========================================
# # 1. 基站物理参数配置 (BS Profile)
# # ==========================================
# # 字段说明：
# # - capacity: 最大容量 (Mbps)
# # - radius_eff: 有效覆盖范围 (m) -> 用于计算邻居关系 (adj_matrix)
# # - radius_max: 最大覆盖范围 (m) -> 只有在极端情况下才允许切换到此距离
# # - p_zero: 零负载功耗 (W) -> 基站空闲时的能耗
# # - slope: 动态负载斜率 -> 每增加100%负载增加的功耗
# # - p_sleep: 休眠功率 (W) -> 深度休眠时的能耗

# BS_CONFIG = {
#     '4G_macro': {
#         'capacity': 300.0,
#         'radius_eff': 500.0, 
#         'radius_max': 1500.0,
#         'p_zero': 400.0,      # 800W * 0.6
#         'slope': 200.0,       # 800W - 480W
#         'p_sleep': 80.0
#     },
#     '4G_micro': {
#         'capacity': 100.0,
#         'radius_eff': 200.0,
#         'radius_max': 300.0,
#         'p_zero': 100.0,       # 150W * 0.6
#         'slope': 60.0,        # 150W - 90W
#         'p_sleep': 20.0
#     },
#     '5G_macro': {
#         'capacity': 2000.0,
#         'radius_eff': 300.0,
#         'radius_max': 1000.0,
#         'p_zero': 1000.0,      # 1200W * 0.6
#         'slope': 500.0,       # 1200W - 720W
#         'p_sleep': 150.0
#     },
#     '5G_micro': {
#         'capacity': 800.0,
#         'radius_eff': 500.0,
#         'radius_max': 1000.0,
#         'p_zero': 180.0,      # 300W * 0.6
#         'slope': 120.0,       # 300W - 180W
#         'p_sleep': 40.0
#     }
# }

def load_and_process_traffic(topo_bs_ids):
    """
    读取流量文件并转置，但不生成额外的时间索引。
    """
    traffic_file = 'Final_Real_Data.csv' # 请确保文件名正确
    
    if not os.path.exists(traffic_file):
        print(f"【错误】找不到流量文件: {traffic_file}")
        return None

    print(f"正在读取流量文件: {traffic_file} ...")
    
    # 1. 读取 CSV (假设第一列是 BS_ID)
    df_raw = pd.read_csv(traffic_file, index_col=0)

    # 2. 转置 (Transpose)
    # Index: BS_ID -> Time Steps
    df_pivot = df_raw.T
    
    print(f"数据转置完成。时间片数量: {df_pivot.shape[0]}, 基站数量: {df_pivot.shape[1]}")
    
    # 3. 数据对齐
    df_pivot.columns = df_pivot.columns.astype(int) # 确保列名为整数ID
    
    # 补全缺失基站
    missing_bs = set(topo_bs_ids) - set(df_pivot.columns)
    if missing_bs:
        print(f"警告: {len(missing_bs)} 个基站缺失，填充为 0。")
        for bs_id in missing_bs:
            df_pivot[bs_id] = 0.0
            
    # 只保留拓扑中的基站，并对齐列顺序
    df_pivot = df_pivot[topo_bs_ids].fillna(0.0)
    
    # 【已删除】这里不再生成 pd.date_range 时间索引
    # 索引将保持原样（0, 1, 2... 或 CSV 中的原始表头）
    
    return df_pivot

def build_dataset():
    # 1. 读取拓扑文件
    topo_file = 'BSPartitioned_refactored.csv'
    if not os.path.exists(topo_file):
        print(f"错误：找不到文件 {topo_file}")
        return None
        
    df_topo = pd.read_csv(topo_file)
    
    # 映射物理参数 (Radius用于邻接矩阵, Capacity用于归一化)
    df_topo['radius_eff'] = df_topo['Type'].map(lambda x: BS_CONFIG[x]['radius_eff'])
    df_topo['radius_max'] = df_topo['Type'].map(lambda x: BS_CONFIG[x]['radius_max'])
    
    print(f"拓扑加载完成。共 {len(df_topo)} 个基站。")
    
    # 2. 加载流量
    all_bs_ids = df_topo['ID'].unique()
    df_traffic = load_and_process_traffic(all_bs_ids)
    
    if df_traffic is None:
        return None
    
    # 3. 构建 Dataset
    Dataset = {}
    grouped = df_topo.groupby('MeshID')
    
    print("开始构建 Mesh 数据集...")
    
    for mesh_id, group in grouped:
        # --- A. 静态信息 ---
        # 重置索引以确保后续矩阵对齐
        static_info = group[['ID', 'Type', 'XLocation', 'YLocation']].reset_index(drop=True)
        current_bs_ids = static_info['ID'].values
        
        # --- B. 邻接矩阵 (根据新逻辑修改) ---
        coords = static_info[['XLocation', 'YLocation']].values
        # 计算两两之间的欧氏距离 Matrix (N, N)
        dist_matrix = cdist(coords, coords, metric='euclidean')
        
        # 获取当前 Mesh 中所有基站的参数
        # 注意：必须使用 static_info 中的顺序 'current_bs_ids' 来提取，保证矩阵行列对应
        topo_indexed = group.set_index('ID')
        r_eff_vals = topo_indexed.loc[current_bs_ids, 'radius_eff'].values  # (N,)
        r_max_vals = topo_indexed.loc[current_bs_ids, 'radius_max'].values  # (N,)
        
        # 【关键修改】覆盖判定逻辑:
        # 如果 A (作为接收方/Target) 能够覆盖 B (作为发送方/Source)
        # 公式: Distance(A, B) + Radius_eff(A) < Radius_max(A)
        # 在邻接矩阵 adj[i, j] 中：
        #   - 行 i 代表 Source (B, 待休眠基站)
        #   - 列 j 代表 Target (A, 接管基站)
        # 我们需要判断：列 j 的基站是否满足上述条件来覆盖行 i 的基站
        
        # 广播机制: 
        # dist_matrix: (N, N)
        # r_eff_vals[None, :]: (1, N) -> 将列 j 的 r_eff 广播到该列的所有行
        # r_max_vals[None, :]: (1, N) -> 将列 j 的 r_max 广播到该列的所有行
        
        coverage_condition = (dist_matrix + r_eff_vals[None, :]) < r_max_vals[None, :]
        
        adj_matrix = coverage_condition.astype(float)
        
        # 确保自己一定能覆盖自己 (对角线置1)
        np.fill_diagonal(adj_matrix, 1.0)
        
        # --- C. 动态流量张量 (Traffic Tensor) ---
        # 1. 取值
        mesh_traffic_values = df_traffic[current_bs_ids].values
        
        # 2. 归一化 (Current / Self_Max)
        self_max = np.max(mesh_traffic_values, axis=0)
        self_max[self_max == 0] = 1.0 # 防止除零
        norm_load = mesh_traffic_values / self_max[None, :]
        
        # 3. 维度调整 (T, N) -> (T, N, 1)
        traffic_tensor = np.expand_dims(norm_load, axis=2)
        
        # --- D. 存入字典 ---
        Dataset[mesh_id] = {
            "static_info": static_info,
            "adj_matrix": adj_matrix,
            "traffic_tensor": traffic_tensor,
            "bs_ids": current_bs_ids
        }
        
    print(f"构建完成！共 {len(Dataset)} 个 Mesh。")
    return Dataset

if __name__ == "__main__":
    dataset = build_dataset()
    
    if dataset:
        # 检查
        first_mesh = list(dataset.keys())[0]
        ts_shape = dataset[first_mesh]['traffic_tensor'].shape
        print(f"\n--- 检查 Mesh {first_mesh} ---")
        print(f"Traffic Tensor Shape: {ts_shape}")
        print(f"  -> Time Steps: {ts_shape[0]}")
        print(f"  -> Base Stations: {ts_shape[1]}")
        print(f"  -> Features: {ts_shape[2]} (仅包含 Load Ratio)")

        # 1. 从字典中提取数据
        mesh_data = dataset[first_mesh]
        adj = mesh_data['adj_matrix']
        ids = mesh_data['bs_ids']  # <--- 这里定义 ids，修复报错
        
        print(f"基站数量: {len(ids)}")
        print(f"基站 ID 列表: {ids}")

        # 2. 转为 DataFrame 以便查看 (行=被覆盖方, 列=覆盖方)
        df_adj = pd.DataFrame(adj, index=ids, columns=ids)

        # 设置显示选项，防止打印时折叠
        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 1000)
        pd.set_option('display.precision', 0) # 只显示整数位(0/1)
        
        print("\n--- 邻接矩阵 (1 表示 列j 能覆盖 行i) ---")
        print(df_adj)
        
        # 2. 转为 DataFrame 以便查看 (行=被覆盖方, 列=覆盖方)
        df_adj = pd.DataFrame(adj, index=ids, columns=ids)
        
        with open('network_env_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)
        print("数据集已保存: network_env_dataset.pkl")
        
        with open('bs_config.pkl', 'wb') as f:
            pickle.dump(BS_CONFIG, f)
        print("配置已保存: bs_config.pkl")