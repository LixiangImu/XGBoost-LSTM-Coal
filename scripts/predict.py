import os
import sys
import pandas as pd
from datetime import datetime
import glob

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.xgboost_model import XGBoostPredictor
from src.data.feature_engineer import FeatureEngineer

def get_latest_model_path():
    """获取最新的模型路径"""
    # 查找所有实验目录
    experiment_dirs = glob.glob('models/xgboost/experiment_*')
    if not experiment_dirs:
        raise ValueError("找不到任何模型文件")
    
    # 获取最新的实验目录
    latest_dir = max(experiment_dirs)
    model_path = f"{latest_dir}/final_model/model.json"
    
    if not os.path.exists(model_path):
        raise ValueError(f"模型文件不存在: {model_path}")
    
    return model_path

def predict_wait_time(
    order_id: str,
    coal_type: str,
    queue_time: str,
    model_path: str = None
):
    """预测等待时间
    Args:
        order_id: 提煤单号
        coal_type: 煤种编号
        queue_time: 排队时间 (格式: 'YYYY-MM-DD HH:MM:SS')
        model_path: 模型文件路径
    Returns:
        float: 预测的等待时间（分钟）
    """
    try:
        # 如果没有指定模型路径，获取最新的模型
        if model_path is None:
            model_path = get_latest_model_path()
        
        print("加载历史数据...")
        # 加载历史数据（用于特征生成）
        historical_data = pd.read_csv('data/processed/processed_data.csv')
        
        # 确保历史数据的时间列是datetime类型
        historical_data['排队时间'] = pd.to_datetime(historical_data['排队时间'])
        
        print("初始化特征工程器...")
        # 初始化特征工程器
        feature_engineer = FeatureEngineer()
        
        print("生成预测特征...")
        # 创建预测数据
        current_time = pd.to_datetime(queue_time)
        prediction_data = pd.DataFrame({
            '提煤单号': [order_id],
            '煤种编号': [coal_type],
            '排队时间': [current_time],
            '等待时间': [0]  # 添加一个占位的等待时间列
        })
        
        # 合并历史数据以生成特征
        data = pd.concat([historical_data, prediction_data], ignore_index=True)
        
        # 生成特征
        features = feature_engineer.combine_features(
            data,
            for_training=False,
            current_time=current_time
        )
        
        # 获取最后一行（预测数据的特征）
        prediction_features = features.iloc[-1:].copy()
        
        print(f"加载模型: {model_path}")
        # 加载模型
        model = XGBoostPredictor()
        model.load_model(model_path)
        
        print("进行预测...")
        # 预测
        prediction = model.predict(prediction_features)[0]
        
        return prediction
        
    except Exception as e:
        print(f"预测失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 示例预测
    print("\n=== 等待时间预测系统 ===")
    
    # 使用当前时间作为示例
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    wait_time = predict_wait_time(
        order_id='TP2024031200001',  # 示例订单号
        coal_type='YM01',  # 示例煤种
        queue_time=current_time  # 使用当前时间
    )
    print(f"\n预测等待时间: {wait_time:.2f} 分钟")