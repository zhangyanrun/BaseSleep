
import argparse
import os
import pickle
from build_dataset import process_raw_predictions, build_dataset_core

# 定义路径常量
RAW_DATA_DIR = "data/raw_data"
TRAFFIC_DIR = "data/traffic_data" # 原始流量数据文件所在目录
TOPO_FILE = "data/mesh_data/BSPartitioned_refactored.csv" # mesh数据

def main(mode):
    # 1. 确定文件名
    # if args.mode == 'train':
    if mode == 'train':
        target_csv_name = "Real_Data.csv"
        output_pkl_name = "data/train_dataset.pkl"
    else:
        target_csv_name = "Predicted_Data.csv"
        output_pkl_name = "data/test_dataset.pkl"
        
    # 2. 检查中间 CSV 是否存在，不存在则调用清洗函数生成
    csv_path = os.path.join(TRAFFIC_DIR, target_csv_name)
    
    if not os.path.exists(csv_path):
        print(f"目标 CSV {target_csv_name} 不存在，正在从原始文件生成...")
        # 注意：process_raw_predictions 会自动去 RAW_PRED_DIR 下找 predictions_model_*.csv
        generated_path = process_raw_predictions(RAW_DATA_DIR, TRAFFIC_DIR, target_csv_name)
        if not generated_path:
            print("生成 CSV 失败，终止。")
            return
    else:
        print(f"发现已存在的 CSV: {csv_path}，直接使用。")

    # 3. 构建最终的 PKL 数据集
    print(f"开始构建 {mode} 数据集...")
    dataset = build_dataset_core(TOPO_FILE, csv_path)
    
    if dataset:
        # 保存
        with open(output_pkl_name, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"成功保存数据集至: {output_pkl_name}")
        
        # 打印简要信息
        first_mesh = list(dataset.keys())[0]
        print(f"示例 Mesh {first_mesh} 数据形状: {dataset[first_mesh]['traffic_tensor'].shape}")

if __name__ == "__main__":
    main('train')
    main('test')