import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def prepare_datasets():
    """准备训练、验证和测试数据集"""
    print("开始数据集划分...")
    
    # 加载原始数据
    data = pd.read_csv('data/processed/xgboost/processed_data.csv')
    features = pd.read_csv('data/processed/xgboost/train_features.csv')
    
    print(f"原始数据大小: {len(data)}")
    print(f"特征数据大小: {len(features)}")
    
    # 确保数据对齐
    # 使用特征数据的索引来过滤原始数据
    if 'index' in features.columns:
        data = data.iloc[features['index'].values]
    else:
        # 如果没有索引列，则取前N条数据
        data = data.iloc[:len(features)]
    
    # 再次验证长度
    if len(data) != len(features):
        raise ValueError(f"数据长度不匹配: 原始数据={len(data)}, 特征数据={len(features)}")
    
    print(f"对齐后数据大小: {len(data)}")
    
    # 1. 首先划分出测试集（20%）
    X_train_val, X_test, y_train_val, y_test, data_train_val, data_test = train_test_split(
        features,
        data['等待时间'].values,
        data,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    
    # 2. 将剩余数据划分为训练集（75%）和验证集（25%）
    X_train, X_val, y_train, y_val, data_train, data_val = train_test_split(
        X_train_val,
        y_train_val,
        data_train_val,
        test_size=0.25,
        random_state=42,
        shuffle=True
    )
    
    # 创建保存目录
    save_dir = 'data/processed/xgboost'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存数据集
    datasets = {
        'train': (X_train, data_train),
        'validation': (X_val, data_val),
        'test': (X_test, data_test)
    }
    
    for name, (features_df, data_df) in datasets.items():
        # 保存特征数据
        features_df.to_csv(f'{save_dir}/{name}_features.csv', index=False)
        # 保存原始数据
        data_df.to_csv(f'{save_dir}/{name}_data.csv', index=False)
    
    # 打印数据集大小
    print("\n数据集划分完成:")
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"\n数据已保存到: {save_dir}")
    
    return {
        'train': (X_train, y_train, data_train),
        'validation': (X_val, y_val, data_val),
        'test': (X_test, y_test, data_test)
    }

if __name__ == "__main__":
    prepare_datasets()