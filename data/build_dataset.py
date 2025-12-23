import os
import glob
import pandas as pd
import numpy as np
import pickle

def process_raw_predictions(raw_dir, target_dir, output_filename):
    """
    将 raw_dir 下的 predictions_model_*.csv 合并、清洗、转置，
    生成最终的宽表 CSV (行=节点, 列=时间)。
    """
    file_pattern = os.path.join(raw_dir, "predictions_model_*.csv")
    file_list = glob.glob(file_pattern)
    file_list.sort()

    if not file_list:
        print(f"未找到任何文件: {file_pattern}")
        return None

    print(f"找到 {len(file_list)} 个原始预测文件，开始合并...")

    all_dfs = []
    for file in file_list:
        df = pd.read_csv(file)
        
        # 逻辑: 偶数列=真实值, 奇数列=预测值
        # 如果是 'Final_Real_Data.csv' -> 取偶数列
        # 如果是 'Final_Predicted_Data.csv' -> 取奇数列
        
        if "Real" in output_filename:
            df_subset = df.iloc[:, ::2]  # 真实值
        else:
            df_subset = df.iloc[:, 1::2] # 预测值
            
        # 转置: (Time, Node) -> (Node, Time)
        df_subset_T = df_subset.T
        all_dfs.append(df_subset_T)

    # 垂直堆叠所有节点
    final_df = pd.concat(all_dfs)
    
    # 重置索引并重命名
    final_df.reset_index(drop=True, inplace=True)
    final_df.index.name = 'BS_ID'
    
    # 保存
    output_path = os.path.join(target_dir, output_filename)
    final_df.to_csv(output_path, index=True, header=True)
    print(f"已生成清洗后的数据: {output_path} (Shape: {final_df.shape})")
    
    return output_path

def load_traffic_csv(csv_path, topo_bs_ids):
    """
    读取清洗后的宽表 CSV，并与拓扑文件中的 BS_ID 对齐
    """
    if not os.path.exists(csv_path):
        print(f"找不到流量文件: {csv_path}")
        return None

    print(f"正在读取流量文件: {csv_path} ...")
    # 读取，第一列是 BS_ID
    df = pd.read_csv(csv_path, index_col=0)
    
    # 转置回 (Time, Node) 方便处理时间序列
    df = df.T 
    
    # 确保列名为整数类型以便匹配
    df.columns = df.columns.astype(int)
    
    # 对齐 BS_ID
    # 只保留拓扑中存在的基站，缺失的补 0
    df = df.reindex(columns=topo_bs_ids, fill_value=0.0)
    
    return df

def build_dataset_core(topo_path, traffic_csv_path):
    """
    构建带有mesh信息的数据集
    """
    # 1. 读取拓扑
    if not os.path.exists(topo_path):
        print(f"找不到拓扑文件: {topo_path}")
        return None
    
    df_topo = pd.read_csv(topo_path)
    all_bs_ids = df_topo['ID'].unique()
    print(f"拓扑加载完成。共 {len(df_topo)} 个基站。")

    # 2. 读取并对齐流量数据
    df_traffic = load_traffic_csv(traffic_csv_path, all_bs_ids)
    if df_traffic is None:
        return None

    # 3. 按 Mesh 分组构建
    Dataset = {}
    grouped = df_topo.groupby('MeshID')
    
    print("开始构建数据集...")
    
    for mesh_id, group in grouped:
        # A. 静态信息 (ID, Type, Location)
        # Reset index 很重要，确保内部索引 0~N 对应
        static_info = group[['ID', 'Type', 'XLocation', 'YLocation']].reset_index(drop=True)
        current_bs_ids = static_info['ID'].values
        
        # B. 流量张量 (Traffic Tensor)
        # 从大表中提取当前 Mesh 的基站列
        mesh_traffic_values = df_traffic[current_bs_ids].values # (Time, N)
        
        # 归一化: 除以该基站自身的最大值 (Max-Scaling)
        # 避免除以 0
        self_max = np.max(mesh_traffic_values, axis=0)
        self_max[self_max == 0] = 1.0 
        norm_load = mesh_traffic_values / self_max[None, :]
        
        # 增加 Feature 维度 -> (Time, N, 1)
        traffic_tensor = np.expand_dims(norm_load, axis=2)
        
        # C. 存入字典
        # 【精简】不再存储 adj_matrix，因为 Env 会在 reset 时根据 Loc 动态算
        Dataset[mesh_id] = {
            "bs_ids": current_bs_ids,
            "static_info": static_info,   # 包含 Loc, Type，够算 Adj 了
            "traffic_tensor": traffic_tensor
        }
        
    print(f"数据集构建完成！包含 {len(Dataset)} 个 Mesh。")
    return Dataset