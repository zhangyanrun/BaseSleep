import os
import pickle
from data.build_dataset_3d import process_csvs_to_tensor, build_dataset_from_tensor
import numpy as np
# === 路径配置 ===
TRAFFIC_DIR = "data/traffic_data"
# 请确保这两个文件名和你实际的一致
REAL_CSV = os.path.join(TRAFFIC_DIR, "Real_Data.csv")
PRED_CSV = os.path.join(TRAFFIC_DIR, "Predicted_Data.csv") 

TOPO_FILE = "data/mesh_data/BSPartitioned_refactored.csv"
TENSOR_FILE = os.path.join(TRAFFIC_DIR, "Traffic_3D_All.npz") # 中间文件
OUTPUT_PKL = "data/dataset_3d_test01.pkl" # 最终输出

def main():
    # 1. 检查或生成 3D Tensor 中间文件
    if not os.path.exists(TENSOR_FILE):
        print("未检测到 3D Tensor，开始从 CSV 生成...")
        if not os.path.exists(REAL_CSV) or not os.path.exists(PRED_CSV):
            print(f"错误: 找不到 CSV 文件。\n请检查:\n{REAL_CSV}\n{PRED_CSV}")
            return
        process_csvs_to_tensor(REAL_CSV, PRED_CSV, TENSOR_FILE)
    else:
        print(f"使用现有的 3D Tensor: {TENSOR_FILE}")
    
    # 2. 构建最终数据集
    dataset = build_dataset_from_tensor(TOPO_FILE, TENSOR_FILE)
    
    if dataset:
        # 保存
        with open(OUTPUT_PKL, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"成功保存数据集至: {OUTPUT_PKL}")
        
        # ==========================================
        # 验证并输出最大值 (max_values)
        # ==========================================
        first_mesh = list(dataset.keys())[0]
        shape = dataset[first_mesh]['traffic_tensor'].shape
        max_vals = dataset[first_mesh]['max_values']
        
        print("\n" + "="*50)
        print(f"示例 Mesh {first_mesh} Shape: {shape}")
        print(f"该 Mesh 内各基站的历史最大流量 (max_values):")
        print(np.round(max_vals, 2))  # 保留两位小数打印
        
        # 统计全网所有基站的最大值
        all_max_values = []
        for mesh_id in dataset:
            all_max_values.extend(dataset[mesh_id]['max_values'])
        all_max_values = np.array(all_max_values)
        
        print("\n=== 全网流量峰值统计 (Global Max Values) ===")
        print(f"全网基站总数: {len(all_max_values)}")
        print(f"平均峰值流量: {np.mean(all_max_values):.2f} Mbps")
        print(f"中位数峰值  : {np.median(all_max_values):.2f} Mbps")
        print(f"最小峰值流量: {np.min(all_max_values):.2f} Mbps")
        print(f"最大峰值流量: {np.max(all_max_values):.2f} Mbps")
        
        # 检查是否有离谱的极端大值
        percentile_99 = np.percentile(all_max_values, 99)
        print(f"99% 的基站峰值低于: {percentile_99:.2f} Mbps")
        print("="*50)
        
        print("\n说明: (Time, N, 2)")
        print("  Channel 0: Real Data")
        print("  Channel 1: Pred Data")
if __name__ == "__main__":
    main()