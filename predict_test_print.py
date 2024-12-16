import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import random
import json

def calculate_actual_wait_time(row):
    """计算实际等待时间(分钟)"""
    queue_time = pd.to_datetime(f"{row['排队日期']} {row['排队时间']}")
    call_time = pd.to_datetime(f"{row['叫号日期']} {row['叫号时间']}")
    wait_minutes = (call_time - queue_time).total_seconds() / 60
    return wait_minutes

def prepare_features(row, historical_data):
    """根据特征重要性准备模型特征"""
    # 定义模型期望的特征顺序
    expected_features = [
        '煤种编号_encoded', 'hour', 'weekday', 'is_weekend', 'shift', 'is_peak_hour',
        'coal_type_wait_mean', 'coal_type_wait_std', 'coal_type_wait_median',
        'coal_type_process_mean', 'coal_type_process_std', 'hour_sin', 'hour_cos',
        'weekday_sin', 'weekday_cos', 'current_queue_length', 'same_coal_type_queue',
        'peak_coal_type', 'queue_wait_ratio', 'type_queue_ratio', 'wait_time_rolling_mean',
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
    
    queue_time = pd.to_datetime(f"{row['排队日期']} {row['排队时间']}")
    coal_type = row['煤种编号']
    hour = queue_time.hour
    weekday = queue_time.weekday()
    
    # 计算基础特征
    current_queue = historical_data[
        (pd.to_datetime(historical_data['排队日期'] + ' ' + historical_data['排队时间']) <= queue_time) &
        (pd.to_datetime(historical_data['叫号日期'] + ' ' + historical_data['叫号时间']) > queue_time)
    ]
    same_type_queue = current_queue[current_queue['煤种编号'] == coal_type]
    recent_window = historical_data.tail(50)
    coal_type_history = recent_window[recent_window['煤种编号'] == coal_type]
    
    # 初始化特征字典
    features = {}
    
    # 基础时间特征
    features['煤种编号_encoded'] = hash(coal_type) % 10
    features['hour'] = hour
    features['weekday'] = weekday
    features['is_weekend'] = 1 if weekday >= 5 else 0
    features['shift'] = (hour // 8) + 1
    features['is_peak_hour'] = 1 if (6 <= hour <= 9) or (16 <= hour <= 19) else 0
    
    # 煤种统计特征
    features['coal_type_wait_mean'] = coal_type_history['actual_wait'].mean() if len(coal_type_history) > 0 else 0
    features['coal_type_wait_std'] = coal_type_history['actual_wait'].std() if len(coal_type_history) > 0 else 0
    features['coal_type_wait_median'] = coal_type_history['actual_wait'].median() if len(coal_type_history) > 0 else 0
    features['coal_type_process_mean'] = features['coal_type_wait_mean']
    features['coal_type_process_std'] = features['coal_type_wait_std']
    
    # 周期特征
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    features['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    
    # 队列特征
    features['current_queue_length'] = len(current_queue)
    features['same_coal_type_queue'] = len(same_type_queue)
    features['peak_coal_type'] = 1 if len(same_type_queue) > 5 else 0
    
    # 队列比率特征
    features['queue_wait_ratio'] = len(current_queue) / max(len(same_type_queue), 1)
    features['type_queue_ratio'] = len(same_type_queue) / max(len(current_queue), 1)
    
    # 滚动统计特征
    for window in [3, 5, 7]:
        recent_data = historical_data.tail(window)
        prefix = f'wait_time_rolling_{window}'
        features[f'{prefix}_mean'] = recent_data['actual_wait'].mean()
        features[f'{prefix}_std'] = recent_data['actual_wait'].std()
        features[f'hour_rolling_mean_{window}'] = recent_data['actual_wait'].mean()
        features[f'shift_rolling_mean_{window}'] = recent_data['actual_wait'].mean()
    
    # 指数加权特征
    features['wait_time_ewm_mean'] = historical_data['actual_wait'].ewm(span=12).mean().iloc[-1]
    features['wait_time_ewm_std'] = historical_data['actual_wait'].ewm(span=12).std().iloc[-1]
    features['wait_time_rolling_mean'] = recent_window['actual_wait'].mean()
    features['wait_time_rolling_std'] = recent_window['actual_wait'].std()
    
    # 时间间隔特征
    if len(coal_type_history) > 0:
        last_same_type = pd.to_datetime(coal_type_history.iloc[-1]['排队日期'] + ' ' + 
                                      coal_type_history.iloc[-1]['排队时间'])
        features['time_since_last_same_type'] = (queue_time - last_same_type).total_seconds() / 60
    else:
        features['time_since_last_same_type'] = 0
    
    # 队列变化特征
    queue_history = historical_data.tail(20)
    if len(queue_history) > 1:
        queue_change = len(current_queue) - len(queue_history)
        type_queue_change = (len(same_type_queue) - 
                           len(queue_history[queue_history['煤种编号'] == coal_type]))
        features['queue_change_rate'] = queue_change / len(queue_history)
        features['type_queue_change_rate'] = type_queue_change / max(len(queue_history), 1)
    else:
        features['queue_change_rate'] = 0
        features['type_queue_change_rate'] = 0
    
    # 高峰期特征
    features['peak_queue_ratio'] = len(current_queue) / max(features['current_queue_length'], 1)
    features['hour_queue_length'] = len(historical_data[historical_data['hour'] == hour])
    features['type_wait_queue'] = len(same_type_queue)
    
    # 班次队列比率
    shift_data = historical_data[historical_data['shift'] == features['shift']]
    features['shift_morning_queue_ratio'] = len(shift_data[shift_data['hour'].between(6, 14)]) / max(len(shift_data), 1)
    features['shift_afternoon_queue_ratio'] = len(shift_data[shift_data['hour'].between(14, 22)]) / max(len(shift_data), 1)
    features['shift_night_queue_ratio'] = len(shift_data[~shift_data['hour'].between(6, 22)]) / max(len(shift_data), 1)
    
    # 高峰期周期特征
    features['hour_sin_peak'] = features['hour_sin'] if features['is_peak_hour'] else 0
    features['hour_cos_peak'] = features['hour_cos'] if features['is_peak_hour'] else 0
    
    # 确保所有特征都存在并按期望顺序排列
    ordered_features = pd.Series(index=expected_features)
    for feature in expected_features:
        ordered_features[feature] = features.get(feature, 0)
    
    return ordered_features

def evaluate_predictions():
    """评估预测结果"""
    # 加载原始数据
    raw_data = pd.read_csv('data/raw/queue_data_offset_sorted_utf8.csv')
    
    # 计算实际等待时间
    raw_data['actual_wait'] = raw_data.apply(calculate_actual_wait_time, axis=1)
    raw_data['shift'] = raw_data['排队时间'].apply(lambda x: int(x.split(':')[0]) // 8 + 1)
    raw_data['hour'] = raw_data['排队时间'].apply(lambda x: int(x.split(':')[0]))
    
    # 从数据集中选择连续的20条记录
    start_index = 5000  # 从第1000条记录开始
    sample_size = 1000
    sample_data = raw_data.iloc[start_index:start_index + sample_size]
    
    # 准备特征并预测
    results = []
    for i, row in sample_data.iterrows():
        print("\n" + "="*50)
        print(f"模拟预测: {row['提煤单号']}")
        
        # 准备请求数据
        order_data = {
            'order_id': row['提煤单号'],
            'coal_type': row['煤种编号'],
            'queue_time': f"{row['排队日期']} {row['排队时间']}"
        }
        print("\n输入数据:")
        print(json.dumps(order_data, indent=2, ensure_ascii=False))
        
        # 准备特征
        features = prepare_features(row, raw_data.loc[:i-1])
        
        # 创建DMatrix时使用有序特征
        dmatrix = xgb.DMatrix(
            data=[features.values],
            feature_names=features.index.tolist()
        )
        
        # 预测
        predicted_wait = float(model.predict(dmatrix)[0])
        actual_wait = row['actual_wait']
        error = predicted_wait - actual_wait
        bias_direction = '偏长' if error > 10 else '偏短'
        
        # 计算预计叫号时间
        queue_dt = pd.to_datetime(f"{row['排队日期']} {row['排队时间']}")
        predicted_call_time = queue_dt + pd.Timedelta(minutes=predicted_wait)
        actual_call_time = queue_dt + pd.Timedelta(minutes=actual_wait)
        
        # 获取队列信息
        current_queue = raw_data[
            (pd.to_datetime(raw_data['排队日期'] + ' ' + raw_data['排队时间']) <= queue_dt) &
            (pd.to_datetime(raw_data['叫号日期'] + ' ' + raw_data['叫号时间']) > queue_dt)
        ]
        same_type_queue = current_queue[current_queue['煤种编号'] == row['煤种编号']]
        
        # 打印预测结果
        print("\n=== 等待时间预测结果 ===")
        print(f"提煤单号: {row['提煤单号']}")
        print(f"煤种编号: {row['煤种编号']}")
        print(f"排队时间: {row['排队日期']} {row['排队时间']}")
        print(f"预计等待: {format_time_period(predicted_wait)}")
        print(f"实际等待: {format_time_period(actual_wait)}")
        print(f"预测偏差: {format_time_period(abs(error))} ({bias_direction})\n")
        print(f"预计叫号: {predicted_call_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"实际叫号: {actual_call_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n=== 队列详情 ===")
        print(f"当前总队列: {len(current_queue)} 辆")
        print(f"同煤种队列: {len(same_type_queue)} 辆")
        print(f"基础处理时间: {format_time_period(25)}")  # 假设基础处理时间为25分钟
        
        # 保存结果
        results.append({
            'order_id': row['提煤单号'],
            'coal_type': row['煤种编号'],
            'queue_time': f"{row['排队日期']} {row['排队时间']}",
            'predicted_wait': predicted_wait,
            'actual_wait': actual_wait,
            'error': error,
            'abs_error': abs(error)
        })
        
        print("="*50)
    
    # 转换为DataFrame并计算统计指标
    results_df = pd.DataFrame(results)
    
    # 打印统计指标
    print("\n=== 预测评估统计 ===")
    print(f"样本数量: {sample_size}")
    print(f"平均绝对误差 (MAE): {results_df['abs_error'].mean():.2f} 分钟")
    print(f"均方根误差 (RMSE): {np.sqrt((results_df['error'] ** 2).mean()):.2f} 分钟")
    print(f"平均绝对百分比误差 (MAPE): {(results_df['abs_error'] / results_df['actual_wait'] * 100).mean():.2f}%")
    
    # 误差分布
    print("\n误差分布:")
    error_ranges = [0, 5, 10, 15, 20, float('inf')]
    error_counts = []
    total_samples = len(results_df)
    
    for i in range(len(error_ranges)-1):
        count = len(results_df[
            (results_df['abs_error'] >= error_ranges[i]) & 
            (results_df['abs_error'] < error_ranges[i+1])
        ])
        percentage = (count / total_samples) * 100
        print(f"误差 {error_ranges[i]}-{error_ranges[i+1]}分钟: {count} 个 ({percentage:.1f}%)")

def format_time_period(minutes):
    """格式化时间段"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}小时{mins}分钟"
    return f"{mins}分钟"

if __name__ == "__main__":
    # 加载模型
    model = xgb.Booster()
    model.load_model('models/xgboost/experiment_20241213_010025/final_model/model.json')
    
    # 运行评估
    evaluate_predictions()