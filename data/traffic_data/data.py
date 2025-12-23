import pandas as pd
import glob
import os

# 1. 获取所有目标 CSV 文件
# 根据截图，文件名模式是 predictions_model_*.csv
file_list = glob.glob("predictions_model_*.csv")
file_list.sort()  # 确保按顺序处理 (dataset_1, dataset_2...)

print(f"找到 {len(file_list)} 个文件: {file_list}")

# 用于存放处理后的数据片段
all_real_nodes = []
all_pred_nodes = []

for file in file_list:
    print(f"正在处理文件: {file} ...")
    
    # 读取数据
    df = pd.read_csv(file)
    
    # -------------------------------------------------------
    # 步骤 A: 分离交替列 (Slicing)
    # -------------------------------------------------------
    # 假设结构是：Actual_0, Predicted_0, Actual_1, Predicted_1 ...
    # ::2  -> 从第0列开始，每隔2列取一次 (偶数列：0, 2, 4...) -> 真实值
    # 1::2 -> 从第1列开始，每隔2列取一次 (奇数列：1, 3, 5...) -> 预测值
    
    df_real = df.iloc[:, ::2]   
    df_pred = df.iloc[:, 1::2]  
    
    # -------------------------------------------------------
    # 步骤 B: 转置 (Transpose)
    # -------------------------------------------------------
    # 用户要求：每一行为一个节点，每一列为一个时间片
    # 原数据：行是时间，列是节点
    # 操作：使用 .T 进行转置
    
    df_real_T = df_real.T
    df_pred_T = df_pred.T
    
    # (可选) 清理索引名称，如果你希望行名保持原始的 "Actual_Node_X" 可以跳过这步
    # 这里不做修改，保留原始 Node 名称作为行索引方便核对
    
    # 添加到列表中
    all_real_nodes.append(df_real_T)
    all_pred_nodes.append(df_pred_T)

# -------------------------------------------------------
# 步骤 C: 垂直聚合 (Concatenate)
# -------------------------------------------------------
# 将4个文件的节点数据垂直堆叠在一起
# 结果将是：(4个文件 * 400节点) 行，(时间片数量) 列

final_real_df = pd.concat(all_real_nodes)
final_pred_df = pd.concat(all_pred_nodes)

# ==== 【关键修改点】 ====
# 1. 强制重置索引 (Reset Index)
#    这会丢弃原本的 "Actual_Node_xx" 字符串索引，生成新的 0, 1, 2... 纯整数索引
final_real_df.reset_index(drop=True, inplace=True)
final_pred_df.reset_index(drop=True, inplace=True)

# 2. 设置索引列的名称为 'BS_ID'
#    这样保存为 CSV 时，第一列的表头就会显示为 BS_ID
final_real_df.index.name = 'BS_ID'
final_pred_df.index.name = 'BS_ID'

print("-" * 30)
print("处理完成！")
print(f"真实数据最终形状 (行=节点, 列=时间): {final_real_df.shape}")
print(f"预测数据最终形状 (行=节点, 列=时间): {final_pred_df.shape}")

# -------------------------------------------------------
# 步骤 D: 保存文件
# -------------------------------------------------------
# header=False/True 根据你是否需要保留时间片序号（0,1,2...）
# index=True 保留节点名称（如 Actual_Node_0）作为第一列

final_real_df.to_csv("Final_Real_Data.csv", index=True, header=True)
final_pred_df.to_csv("Final_Predicted_Data.csv", index=True, header=True)

print("文件已保存：")
print("1. Final_Real_Data.csv")
print("2. Final_Predicted_Data.csv")


###查看NPZ的数据格式
# import numpy as np

# # 1. 加载 npz 文件
# # 替换成你的实际文件名
# data = np.load('/mnt/data/zyr/Experiment/version4/data/Milan0/cluster1/1/test.npz')

# # ------------------------------------------
# # 维度一：查看里面包含多少个数组（变量）
# # ------------------------------------------
# print(f"--- 归档文件包含 {len(data)} 个数组 ---")
# print(f"数组名称 (Keys): {data.files}") 

# # ------------------------------------------
# # 维度二：查看每个数组具体的形状和数据量
# # ------------------------------------------
# print("\n--- 每个数组的详细信息 ---")
# for key in data.files:
#     array_content = data[key]
#     print(f"变量名: {key}")
#     print(f"  - 形状 (Shape): {array_content.shape}") # 最常用：看行数就知道有多少样本
#     print(f"  - 元素总数 (Size): {array_content.size}") 
#     print(f"  - 数据类型 (Dtype): {array_content.dtype}")
#     print("-" * 20)

# # 记得关闭文件（虽然 Python 会自动回收，但手动关闭是好习惯）
# data.close()