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
    experiment_dirs = glob.glob('models/xgboost/experiment_*')
    if not experiment_dirs:
        raise ValueError("找不到任何模型文件")
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
    """预测等待时间"""
    try:
        # 获取最新模型路径
        if model_path is None:
            model_path = get_latest_model_path()
        
        print("加载历史数据...")
        historical_data = pd.read_csv('data/processed/processed_data.csv')
        historical_data['排队时间'] = pd.to_datetime(historical_data['排队时间'])
        historical_data['叫号时间'] = pd.to_datetime(historical_data['叫号时间'])
        
        # 创建预测数据
        current_time = pd.to_datetime(queue_time)
        prediction_data = pd.DataFrame({
            '提煤单号': [order_id],
            '煤种编号': [coal_type],
            '排队时间': [current_time],
            '叫号时间': [current_time],  # 临时设置，用于特征生成
            '处理时间': [0],            # 临时设置，用于特征生成
            '等待时间': [0]             # 临时设置，用于特征生成
        })
        
        # 合并数据并按时间排序
        all_data = pd.concat([historical_data, prediction_data])
        all_data = all_data.sort_values('排队时间').reset_index(drop=True)
        
        # 初始化特征工程器
        feature_engineer = FeatureEngineer()
        
        print("生成预测特征...")
        # 生成所有特征，包括滚动特征
        features = feature_engineer.combine_features(
            all_data,
            for_training=True,  # 设置为True以生成所有特征
            current_time=current_time
        )
        
        # 确保所有需要的特征都存在
        expected_features = [
            '煤种编号_encoded', 'hour', 'weekday', 'is_weekend', 'shift', 
            'is_peak_hour', 'coal_type_wait_mean', 'coal_type_wait_std', 
            'coal_type_wait_median', 'coal_type_process_mean', 'coal_type_process_std',
            'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
            'current_queue_length', 'same_coal_type_queue', 'peak_coal_type',
            'queue_wait_ratio', 'type_queue_ratio', 'wait_time_rolling_mean',
            'wait_time_rolling_std', 'wait_time_rolling_mean_3', 'wait_time_rolling_std_3',
            'hour_rolling_mean_3', 'shift_rolling_mean_3', 'wait_time_rolling_mean_5',
            'wait_time_rolling_std_5', 'hour_rolling_mean_5', 'shift_rolling_mean_5',
            'wait_time_rolling_mean_7', 'wait_time_rolling_std_7', 'hour_rolling_mean_7',
            'shift_rolling_mean_7', 'wait_time_ewm_mean', 'wait_time_ewm_std',
            'time_since_last_same_type', 'queue_change_rate', 'type_queue_change_rate',
            'peak_queue_ratio', 'hour_queue_length', 'type_wait_queue',
            'shift_afternoon_queue_ratio', 'shift_morning_queue_ratio',
            'shift_night_queue_ratio', 'hour_sin_peak', 'hour_cos_peak'
        ]
        
        # 检查是否缺少特征
        missing_features = set(expected_features) - set(features.columns)
        if missing_features:
            print("\n警告: 缺少以下特征:")
            for feat in missing_features:
                print(f"- {feat}")
                features[feat] = 0  # 使用0填充缺失的特征
        
        # 确保特征顺序一致
        features = features[expected_features]
        
        # 获取最后一行作为预测数据
        prediction_features = features.iloc[-1:].copy()
        
        print(f"\n加载模型: {model_path}")
        model = XGBoostPredictor()
        model.load_model(model_path)
        
        print("\n特征维度:", prediction_features.shape)
        print("特征列表:")
        for col in prediction_features.columns:
            print(f"- {col}")
        
        print("\n进行预测...")
        prediction = model.predict(prediction_features)[0]
        
        return prediction
        
    except Exception as e:
        print(f"预测失败: {str(e)}")
        raise

if __name__ == "__main__":
    print("\n=== 等待时间预测系统 ===")
    
    # 使用历史数据中的最后一个时间点后的时间
    historical_data = pd.read_csv('data/processed/processed_data.csv')
    last_time = pd.to_datetime(historical_data['排队时间'].max())
    predict_time = (last_time + pd.Timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"\n预测时间点: {predict_time}")
    wait_time = predict_wait_time(
        order_id='TP2024031200001',
        coal_type='YM01',
        queue_time=predict_time
    )
    print(f"\n预测等待时间: {wait_time:.2f} 分钟")