import pickle
import numpy as np

# 数据集路径
DATASET_PATH = "data/dataset_3d.pkl"

def inspect_first_mesh():
    # 1. 加载数据集
    print(f"正在加载数据集: {DATASET_PATH} ...")
    try:
        with open(DATASET_PATH, 'rb') as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {DATASET_PATH}。请先运行 main.py 生成数据。")
        return

    # 2. 获取第一个 Mesh 的 Key
    # dataset 是一个字典，Key 是 MeshID
    if not dataset:
        print("数据集为空！")
        return
        
    first_mesh_id = list(dataset.keys())[0]
    print(f"\n=== 第一个 Mesh ID: {first_mesh_id} ===")
    
    # 3. 获取该 Mesh 的数据对象
    mesh_data = dataset[first_mesh_id]
    
    # mesh_data 结构:
    # {
    #    "bs_ids": array([1001, 1002...]), 
    #    "static_info": DataFrame, 
    #    "traffic_tensor": array(Time, N, 2),
    #    "max_values": array(N,)
    # }

    # 4. 提取流量张量
    traffic_tensor = mesh_data['traffic_tensor']
    bs_ids = mesh_data['bs_ids']
    
    # 5. 打印基本信息
    print(f"包含基站数量 (N): {len(bs_ids)}")
    print(f"基站 IDs: {bs_ids}")
    print(f"流量张量形状 (Time, N, Channels): {traffic_tensor.shape}")
    
    # 6. 打印前 5 个时间步的具体数据 (示例)
    # Channel 0: Real
    # Channel 1: Pred
    print("\n--- 前 5 个时间步的流量数据 (归一化后) ---")
    print(f"{'TimeStep':<10} | {'BS_ID':<10} | {'Real_Load':<10} | {'Pred_Load':<10}")
    print("-" * 50)
    
    # 只打印第一个基站的前 5 行
    target_bs_idx = 0 
    target_bs_id = bs_ids[target_bs_idx]
    
    for t in range(5):
        real_val = traffic_tensor[t, target_bs_idx, 0]
        pred_val = traffic_tensor[t, target_bs_idx, 1]
        print(f"{t:<10} | {target_bs_id:<10} | {real_val:.4f}     | {pred_val:.4f}")

    # 7. (可选) 打印反归一化后的真实值
    # 真实值 = 归一化值 * Max值
    max_val = mesh_data['max_values'][target_bs_idx]
    print(f"\n--- 该基站最大流量值 (Max): {max_val:.2f} ---")
    print("--- 反归一化后的真实流量 (Mbps) ---")
    print(f"{'TimeStep':<10} | {'Real_Mbps':<10} | {'Pred_Mbps':<10}")
    
    for t in range(5):
        real_val = traffic_tensor[t, target_bs_idx, 0] * max_val
        pred_val = traffic_tensor[t, target_bs_idx, 1] * max_val
        print(f"{t:<10} | {real_val:.2f}       | {pred_val:.2f}")

if __name__ == "__main__":
    inspect_first_mesh()