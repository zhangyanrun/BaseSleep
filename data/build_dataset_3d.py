import os
import pandas as pd
import numpy as np
import pickle

def process_csvs_to_tensor(real_csv_path, pred_csv_path, output_path):
    """
    读取真实值和预测值 CSV，合并为 3D Tensor (N, T, 2) 并保存。
    Channel 0: Real (真实值)
    Channel 1: Pred (预测值)
    """
    print(f"正在读取真实流量: {real_csv_path}")
    df_real = pd.read_csv(real_csv_path, index_col=0) # 假设第一列是 BS_ID
    
    print(f"正在读取预测流量: {pred_csv_path}")
    df_pred = pd.read_csv(pred_csv_path, index_col=0)
    
    # --- 1. 数据对齐 ---
    # 取基站 ID 的交集，确保行严格对应
    common_ids = df_real.index.intersection(df_pred.index)
    
    if len(common_ids) != len(df_real) or len(common_ids) != len(df_pred):
        print(f"警告: 基站数量不一致! Real: {len(df_real)}, Pred: {len(df_pred)}. 取交集: {len(common_ids)}")
    
    # 重新索引，确保顺序严格一致
    df_real = df_real.loc[common_ids]
    df_pred = df_pred.loc[common_ids]
    
    # 转为 numpy 数组 (N, T)
    real_val = df_real.values
    pred_val = df_pred.values
    
    # 简单的形状检查
    if real_val.shape != pred_val.shape:
        raise ValueError(f"时间步长度不一致! Real: {real_val.shape}, Pred: {pred_val.shape}")

    # --- 2. 堆叠为 3D Tensor ---
    # 结果形状: (Nodes, Time, 2)
    # axis=2 表示在最后增加一个维度
    traffic_tensor = np.stack([real_val, pred_val], axis=2)
    bs_ids = common_ids.values

    # --- 3. 保存为压缩格式 (中间文件) ---
    np.savez_compressed(output_path, data=traffic_tensor, ids=bs_ids)
    print(f"3D Tensor 构建完成: {output_path}")
    print(f"  - Shape: {traffic_tensor.shape} (Nodes, Time, Channels)")
    
    return output_path

def build_dataset_from_tensor(topo_path, tensor_path):
    """
    读取 3D Tensor，结合拓扑信息构建最终 Dataset 字典。
    """
    # 1. 读取拓扑
    if not os.path.exists(topo_path):
        print(f"找不到拓扑文件: {topo_path}")
        return None
    df_topo = pd.read_csv(topo_path)
    print(f"拓扑加载完成。共 {len(df_topo)} 个基站。")

    # 2. 读取 3D Tensor
    if not os.path.exists(tensor_path):
        print(f"找不到 Tensor 文件: {tensor_path}")
        return None
    
    loaded = np.load(tensor_path, allow_pickle=True)
    raw_data = loaded['data'] # Shape: (All_N, T, 2)
    raw_ids = loaded['ids']   # Shape: (All_N,)
    
    # 建立 ID -> Index 映射，方便快速查找
    id_to_idx = {bs_id: i for i, bs_id in enumerate(raw_ids)}

    # 3. 按 Mesh 分组构建
    Dataset = {}
    grouped = df_topo.groupby('MeshID')
    
    print("开始构建数据集...")
    
    for mesh_id, group in grouped:
        static_info = group[['ID', 'Type', 'XLocation', 'YLocation']].reset_index(drop=True)
        target_ids = static_info['ID'].values
        
        # 提取当前 Mesh 的基站索引
        mesh_indices = []
        valid_bs_ids = []
        for bid in target_ids:
            if bid in id_to_idx:
                mesh_indices.append(id_to_idx[bid])
                valid_bs_ids.append(bid)
        
        if not mesh_indices:
            continue
            
        # 提取数据: (Mesh_N, T, 2)
        mesh_data = raw_data[mesh_indices]
        
        # 转置为 (Time, Mesh_N, 2) -> 符合 DRL/LSTM 的 (Step, Agent, Feature) 习惯
        # Channel 0 = Real, Channel 1 = Pred
        mesh_data_T = mesh_data.transpose(1, 0, 2)
        
        # --- 归一化 (关键步骤) ---
        # 必须使用 Real Data (Channel 0) 来计算最大值，保证尺度统一
        real_vals = mesh_data_T[:, :, 0]
        self_max = np.max(real_vals, axis=0)
        self_max[self_max == 0] = 1.0 # 避免除以0
        
        # 对整个 Tensor (包括 Pred Channel) 进行归一化
        # 这样预测值也会被缩放到同样的比例
        norm_data = mesh_data_T / self_max[None, :, None]
        
        Dataset[mesh_id] = {
            "bs_ids": np.array(valid_bs_ids),
            "static_info": static_info[static_info['ID'].isin(valid_bs_ids)].reset_index(drop=True),
            "traffic_tensor": norm_data, # (Time, N, 2)
            "max_values": self_max       # 保存最大值以便后续反归一化
        }
        
    print(f"数据集构建完成！包含 {len(Dataset)} 个 Mesh。")
    return Dataset