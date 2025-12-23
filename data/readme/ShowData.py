# import numpy as np
# import logging

# # 1. 配置日志设置
# # filename: 日志文件名
# # level: 记录级别 (INFO 表示记录一般信息)
# # format: 日志的格式 (时间 - 级别 - 内容)
# logging.basicConfig(
#     filename='data_process.log', 
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )

# try:
#     # 2. 加载数据
#     data = np.load('Milan.npy')

#     # 3. 拼接要记录的格式信息
#     # 建议记录：形状、数据类型、占用内存大小、维度
#     log_msg = (
#         f"数据加载成功。\n"
#         f"    - 形状 (Shape): {data.shape}\n"
#         f"    - 类型 (Dtype): {data.dtype}\n"
#         f"    - 维度 (Ndim):  {data.ndim}\n"
#         f"    - 元素总数 (Size): {data.size}"
#     )

#     # 4. 写入日志
#     logging.info(log_msg)
#     print("日志写入完成。")

# except Exception as e:
#     # 如果出错，也可以记录错误日志
#     logging.error(f"读取文件失败: {e}")



### 查看network_env_dataset.pkl
import pickle
import pandas as pd
import numpy as np
import os

def inspect_dataset(file_path='network_env_dataset.pkl'):
    """
    深度检查 PKL 数据集文件的内容、维度和物理意义。
    """
    print(f"\n{'='*40}")
    print(f"🔍 正在检查文件: {file_path}")
    print(f"{'='*40}")

    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 1. 全局概览
    mesh_keys = list(dataset.keys())
    print(f"✅ 读取成功！")
    print(f"📊 数据集包含 Mesh 数量: {len(mesh_keys)}")
    print(f"🔑 Mesh ID 示例: {mesh_keys[:10]} ...")

    # 2. 抽取第一个 Mesh 进行深度检查 (通常是 Mesh 0)
    target_mesh = mesh_keys[0]
    print(f"\n{'-'*20} 正在深入检查 Mesh {target_mesh} {'-'*20}")
    
    data = dataset[target_mesh]
    
    # --- A. 完整性检查 ---
    bs_ids = data['bs_ids']
    num_bs = len(bs_ids)
    print(f"1. [基站列表] (数量: {num_bs})")
    print(f"   IDs: {bs_ids}")

    # --- B. 流量张量检查 ---
    tensor = data['traffic_tensor']
    print(f"\n2. [流量张量 Traffic Tensor]")
    print(f"   Shape: {tensor.shape} (Time, BS, Feature)")
    
    # 维度校验
    if tensor.shape[1] != num_bs:
        print(f"   ❌ 警告：张量第2维 ({tensor.shape[1]}) 与基站数量 ({num_bs}) 不一致！(可能有脏数据)")
    else:
        print(f"   ✅ 维度校验通过：时间步={tensor.shape[0]}, 基站数={tensor.shape[1]}")
        
    # 数值范围检查
    print(f"   Max Load: {np.max(tensor):.4f}, Min Load: {np.min(tensor):.4f} (应在 0~1 之间)")

    # --- C. 邻接矩阵检查 ---
    adj = data['adj_matrix']
    print(f"\n3. [邻接矩阵 Adjacency Matrix]")
    print(f"   Shape: {adj.shape}")
    
    # 转为 DataFrame 方便查看
    df_adj = pd.DataFrame(adj, index=bs_ids, columns=bs_ids)
    
    # 设置 Pandas 显示选项 (防止打印不全)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 0)
    
    print("\n   --- 矩阵预览 (1=列覆盖行, 0=不可覆盖) ---")
    print(df_adj)
    
    # --- D. 覆盖逻辑统计 ---
    # 计算每个基站被多少个“其他基站”覆盖 (Row Sum - 1)
    covered_counts = np.sum(adj, axis=1) - 1
    avg_coverage = np.mean(covered_counts)
    
    print(f"\n   --- 覆盖统计 ---")
    print(f"   平均每个基站可被 {avg_coverage:.2f} 个邻居接管")
    
    # 找出孤立点
    isolated = np.where(covered_counts == 0)[0]
    if len(isolated) > 0:
        print(f"   ⚠️ 注意：有 {len(isolated)} 个基站无法被任何邻居覆盖 (只能自己覆盖自己):")
        print(f"   -> IDs: {bs_ids[isolated]}")
    else:
        print(f"   ✅ 所有基站至少有一个邻居可以接管它。")

    # --- E. 静态信息预览 ---
    print(f"\n4. [静态信息 Static Info]")
    print(data['static_info'].head())

if __name__ == "__main__":
    inspect_dataset()