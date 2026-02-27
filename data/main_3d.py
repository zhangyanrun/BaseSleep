import os
import pickle
from build_dataset_3d import process_csvs_to_tensor, build_dataset_from_tensor

# === 路径配置 ===
TRAFFIC_DIR = "data/traffic_data"
# 请确保这两个文件名和你实际的一致
REAL_CSV = os.path.join(TRAFFIC_DIR, "Real_Data.csv")
PRED_CSV = os.path.join(TRAFFIC_DIR, "Predicted_Data.csv") 

TOPO_FILE = "data/mesh_data/BSPartitioned_refactored.csv"
TENSOR_FILE = os.path.join(TRAFFIC_DIR, "Traffic_3D_All.npz") # 中间文件
OUTPUT_PKL = "data/dataset_3d.pkl" # 最终输出

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
        
        # 验证一下数据
        first_mesh = list(dataset.keys())[0]
        shape = dataset[first_mesh]['traffic_tensor'].shape
        print(f"示例 Mesh {first_mesh} Shape: {shape}")
        print("说明: (Time, N, 2)")
        print("  Channel 0: Real Data (用于当前状态 D_t 和 训练/测试的 Reward结算)")
        print("  Channel 1: Pred Data (仅用于测试时的 D_t+1)")

if __name__ == "__main__":
    main()