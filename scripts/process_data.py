# scripts/process_data.py

import os
import sys
import pandas as pd
from datetime import datetime

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data.data_processor import DataProcessor
from src.data.feature_engineer import FeatureEngineer

def main():
    """主执行函数"""
    try:
        # 1. 初始化数据处理器和特征工程器
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        
        # 2. 加载原始数据
        raw_data_path = 'data/raw/queue_data_offset_sorted_utf8.csv'
        print("开始加载数据...")
        raw_data = data_processor.load_data(raw_data_path)
        
        # 3. 数据预处理
        print("开始数据预处理...")
        processed_data = data_processor.prepare_data(raw_data)
        
        # 4. 保存处理后的数据
        processed_data_path = 'data/processed/processed_data.csv'
        data_processor.save_processed_data(processed_data_path)
        
        # 5. 分割训练集和测试集
        print("分割训练集和测试集...")
        train_df, test_df = data_processor.split_train_test(processed_data)
        print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")
        
        # 6. 特征工程 - 训练集
        print("开始生成训练集特征...")
        train_features = feature_engineer.combine_features(
            train_df, 
            for_training=True
        )
        
        # 7. 特征工程 - 测试集
        print("开始生成测试集特征...")
        test_features = feature_engineer.combine_features(
            test_df, 
            for_training=True
        )
        
        # 8. 保存特征数据
        train_features_path = 'data/processed/train_features.csv'
        test_features_path = 'data/processed/test_features.csv'
        
        train_features.to_csv(train_features_path, index=False)
        test_features.to_csv(test_features_path, index=False)
        
        print("\n数据处理和特征工程完成！")
        print(f"训练特征形状: {train_features.shape}")
        print(f"测试特征形状: {test_features.shape}")
        
        # 9. 特征统计信息
        print("\n特征列表:")
        for col in train_features.columns:
            print(f"- {col}")
            
        # 10. 测试实时预测特征生成
        print("\n测试实时预测特征生成...")
        sample_prediction = feature_engineer.prepare_features_for_prediction(
            order_id='TP2102024101600xx',
            coal_type='YT02WX',
            queue_time='2024-10-16 07:05:00',
            historical_data=processed_data
        )
        print("实时预测特征生成成功！")
        print(f"预测特征形状: {sample_prediction.shape}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()