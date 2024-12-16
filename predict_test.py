import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
import random
import json
import os

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
    
    # 1. 优化时间窗口选择
    recent_window = historical_data.tail(25)
    last_hour_data = historical_data[
        pd.to_datetime(historical_data['排队日期'] + ' ' + historical_data['排队时间']) >= 
        queue_time - pd.Timedelta(hours=1)
    ]
    
    # 2. 计算更精确的当前队列
    current_queue = historical_data[
        (pd.to_datetime(historical_data['排队日期'] + ' ' + historical_data['排队时间']) <= queue_time) &
        (pd.to_datetime(historical_data['叫号日期'] + ' ' + historical_data['叫号时间']) > queue_time)
    ]
    same_type_queue = current_queue[current_queue['煤种编号'] == coal_type]
    
    # 3. 优化煤种统计特征
    coal_type_history = recent_window[recent_window['煤种编号'] == coal_type]
    if len(coal_type_history) > 0:
        features = {
            'coal_type_wait_mean': coal_type_history['actual_wait'].mean(),
            'coal_type_wait_std': coal_type_history['actual_wait'].std(),
            'coal_type_wait_median': coal_type_history['actual_wait'].median(),
            'coal_type_process_mean': coal_type_history['actual_wait'].mean(),
            'coal_type_process_std': coal_type_history['actual_wait'].std()
        }
    else:
        recent_stats = recent_window['actual_wait'].agg(['mean', 'std', 'median'])
        features = {
            'coal_type_wait_mean': recent_stats['mean'],
            'coal_type_wait_std': recent_stats['std'],
            'coal_type_wait_median': recent_stats['median'],
            'coal_type_process_mean': recent_stats['mean'],
            'coal_type_process_std': recent_stats['std']
        }
    
    # 4. 优化时间特征
    features.update({
        '煤种编号_encoded': hash(coal_type) % 10,
        'hour': hour,
        'weekday': weekday,
        'is_weekend': 1 if weekday >= 5 else 0,
        'shift': (hour // 8) + 1,
        'is_peak_hour': 1 if (6 <= hour <= 9) or (16 <= hour <= 19) else 0,
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'weekday_sin': np.sin(2 * np.pi * weekday / 7),
        'weekday_cos': np.cos(2 * np.pi * weekday / 7)
    })
    
    # 5. 优化队列特征
    queue_position = len(same_type_queue)
    total_queue = len(current_queue)
    features.update({
        'current_queue_length': total_queue,
        'same_coal_type_queue': queue_position,
        'peak_coal_type': 1 if queue_position > 5 else 0,
        'queue_wait_ratio': min(queue_position * 25 / max(features['coal_type_wait_mean'], 25), 2.5),
        'type_queue_ratio': queue_position / max(total_queue, 1)
    })
    
    # 6. 优化滚动统计特征
    for window in [3, 5, 7]:
        recent_data = last_hour_data.tail(window)
        if len(recent_data) > 0:
            features.update({
                f'wait_time_rolling_mean_{window}': recent_data['actual_wait'].mean(),
                f'wait_time_rolling_std_{window}': recent_data['actual_wait'].std(),
                f'hour_rolling_mean_{window}': recent_data['actual_wait'].mean(),
                f'shift_rolling_mean_{window}': recent_data['actual_wait'].mean()
            })
        else:
            features.update({
                f'wait_time_rolling_mean_{window}': features['coal_type_wait_mean'],
                f'wait_time_rolling_std_{window}': features['coal_type_wait_std'],
                f'hour_rolling_mean_{window}': features['coal_type_wait_mean'],
                f'shift_rolling_mean_{window}': features['coal_type_wait_mean']
            })
    
    # 7. 优化高峰期特征
    is_peak = features['is_peak_hour']
    features.update({
        'peak_queue_ratio': len(current_queue) / max(features['current_queue_length'], 1) if is_peak else 0.5,
        'hour_queue_length': len(last_hour_data),
        'type_wait_queue': queue_position,
        'hour_sin_peak': features['hour_sin'] if is_peak else 0,
        'hour_cos_peak': features['hour_cos'] if is_peak else 0
    })
    
    # 8. 添加班次队列比率
    shift_data = historical_data[historical_data['shift'] == features['shift']]
    features.update({
        'shift_morning_queue_ratio': len(shift_data[shift_data['hour'].between(6, 14)]) / max(len(shift_data), 1),
        'shift_afternoon_queue_ratio': len(shift_data[shift_data['hour'].between(14, 22)]) / max(len(shift_data), 1),
        'shift_night_queue_ratio': len(shift_data[~shift_data['hour'].between(6, 22)]) / max(len(shift_data), 1)
    })
    
    # 确保所有特征都存在并按期望顺序排列
    ordered_features = pd.Series(index=expected_features)
    for feature in expected_features:
        ordered_features[feature] = features.get(feature, 0)
    
    return ordered_features

def evaluate_predictions():
    """评估预测结果并生成预测报告"""
    # 创建预测报告目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_dir = f'predictions/prediction_{timestamp}'
    os.makedirs(report_dir, exist_ok=True)
    
    # 加载原始数据
    raw_data = pd.read_csv('data/raw/queue_data_offset_sorted_utf8.csv')
    raw_data['actual_wait'] = raw_data.apply(calculate_actual_wait_time, axis=1)
    raw_data['shift'] = raw_data['排队时间'].apply(lambda x: int(x.split(':')[0]) // 8 + 1)
    raw_data['hour'] = raw_data['排队时间'].apply(lambda x: int(x.split(':')[0]))
    
    # 从数据集中选择连续的记录
    start_index = 1
    sample_size = 200
    sample_data = raw_data.iloc[start_index:start_index + sample_size]
    
    # 准备特征并预测
    results = []
    
    # 创建详细预测报告文件
    with open(f'{report_dir}/detailed_predictions.txt', 'w', encoding='utf-8') as f:
        for i, row in sample_data.iterrows():
            # 写入分隔线
            f.write("="*50 + "\n")
            
            # 写入基本信息
            order_data = {
                'order_id': row['提煤单号'],
                'coal_type': row['煤种编号'],
                'queue_time': f"{row['排队日期']} {row['排队时间']}"
            }
            f.write(f"模拟预测: {order_data['order_id']}\n\n")
            f.write("输入数据:\n")
            f.write(json.dumps(order_data, indent=2, ensure_ascii=False) + "\n\n")
            
            # 准备特征和预测
            features = prepare_features(row, raw_data.loc[:i-1])
            dmatrix = xgb.DMatrix(
                data=[features.values],
                feature_names=features.index.tolist()
            )
            
            predicted_wait = float(model.predict(dmatrix)[0])
            actual_wait = row['actual_wait']
            error = predicted_wait - actual_wait
            
            # 计算时间
            queue_dt = pd.to_datetime(f"{row['排队日期']} {row['排队时间']}")
            predicted_call_time = queue_dt + pd.Timedelta(minutes=predicted_wait)
            actual_call_time = queue_dt + pd.Timedelta(minutes=actual_wait)
            
            # 获取队列信息
            current_queue = raw_data[
                (pd.to_datetime(raw_data['排队日期'] + ' ' + raw_data['排队时间']) <= queue_dt) &
                (pd.to_datetime(raw_data['叫号日期'] + ' ' + raw_data['叫号时间']) > queue_dt)
            ]
            same_type_queue = current_queue[current_queue['煤种编号'] == row['煤种编号']]
            
            # 写入预测结果
            f.write("=== 等待时间预测结果 ===\n")
            f.write(f"提煤单号: {row['提煤单号']}\n")
            f.write(f"煤种编号: {row['煤种编号']}\n")
            f.write(f"排队时间: {row['排队日期']} {row['排队时间']}\n")
            f.write(f"预计等待: {format_time_period(predicted_wait)}\n")
            f.write(f"实际等待: {format_time_period(actual_wait)}\n")
            f.write(f"预测偏差: {format_time_period(abs(error))} \n")
            f.write(f"预计叫号: {predicted_call_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实际叫号: {actual_call_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 写入队列详情
            f.write("=== 队列详情 ===\n")
            f.write(f"当前总队列: {len(current_queue)} 辆\n")
            f.write(f"同煤种队列: {len(same_type_queue)} 辆\n")
            f.write(f"基础处理时间: {format_time_period(25)}\n")
            f.write("="*50 + "\n\n")
            
            # 保存结果用于统计
            results.append({
                'order_id': row['提煤单号'],
                'coal_type': row['煤种编号'],
                'queue_time': f"{row['排队日期']} {row['排队时间']}",
                'predicted_wait': predicted_wait,
                'actual_wait': actual_wait,
                'error': error,
                'abs_error': abs(error)
            })
    
    # 转换为DataFrame并计算统计指标
    results_df = pd.DataFrame(results)
    
    # 创建统计报告文件
    with open(f'{report_dir}/statistics_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== 预测评估统计 ===\n")
        f.write(f"样本数量: {sample_size}\n")
        f.write(f"平均绝对误差 (MAE): {results_df['abs_error'].mean():.2f} 分钟\n")
        f.write(f"均方根误差 (RMSE): {np.sqrt((results_df['error'] ** 2).mean()):.2f} 分钟\n")
        f.write(f"平均绝对百分比误差 (MAPE): {(results_df['abs_error'] / results_df['actual_wait'] * 100).mean():.2f}%\n\n")
        
        # 误差分布
        f.write("误差分布:\n")
        error_ranges = [0, 5, 10, 15, 20, float('inf')]
        for i in range(len(error_ranges)-1):
            count = len(results_df[
                (results_df['abs_error'] >= error_ranges[i]) & 
                (results_df['abs_error'] < error_ranges[i+1])
            ])
            percentage = (count / len(results_df)) * 100
            f.write(f"误差 {error_ranges[i]}-{error_ranges[i+1]}分钟: {count} 个 ({percentage:.1f}%)\n")
    
    # 保存原始结果数据
    results_df.to_csv(f'{report_dir}/raw_results.csv', index=False)
    
    print(f"预测报告已生成在目录: {report_dir}")

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