import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.models.xgboost_model import XGBoostPredictor
from src.data.xgb_feature_engineer import FeatureEngineer

def calculate_queue_state(data, current_idx, window_size=10):
    """计算当前队列状态
    Args:
        data: DataFrame, 包含历史数据
        current_idx: int, 当前预测位置
        window_size: int, 考虑的历史窗口大小
    Returns:
        dict: 队列状态信息
    """
    if current_idx < window_size:
        history_data = data.iloc[:current_idx]
    else:
        history_data = data.iloc[current_idx-window_size:current_idx]
    
    current_row = data.iloc[current_idx]
    
    # 计算队列状态
    queue_state = {
        'current_queue_length': len(history_data[history_data['等待时间'] > 0]),
        'same_coal_type_queue': len(history_data[history_data['煤种编号'] == current_row['煤种编号']]),
        'recent_wait_times': history_data['等待时间'].mean(),
        'recent_wait_times_std': history_data['等待时间'].std(),
        'same_type_wait_times': history_data[history_data['煤种编号'] == current_row['煤种编号']]['等待时间'].mean(),
        'current_hour_queue': len(history_data[history_data['排队时间'].apply(lambda x: x.hour) == current_row['排队时间'].hour])
    }
    
    return queue_state

def generate_evaluation_report(validation_data, model):
    """生成评估报告"""
    try:
        # 1. 创建结果目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_dir = f'evaluation_results/experiment_{timestamp}'
        os.makedirs(result_dir, exist_ok=True)
        print(f"\n结果将保存在目录: {result_dir}")
        
        # 2. 预处理数据
        print("正在预处理数据...")
        validation_data = validation_data.copy()
        validation_data['排队时间'] = pd.to_datetime(validation_data['排队时间'])
        if '排队日期' in validation_data.columns:
            validation_data['排队日期'] = pd.to_datetime(validation_data['排队日期'])
        
        # 3. 逐条进行预测
        print("正在进行预测...")
        predictions = []
        queue_states = []
        
        for idx in range(len(validation_data)):
            if idx % 100 == 0:
                print(f"正在处理第 {idx}/{len(validation_data)} 条数据...")
            
            try:
                # 获取当前队列状态
                queue_state = calculate_queue_state(validation_data, idx)
                queue_states.append(queue_state)
                
                # 准备特征
                current_data = validation_data.iloc[[idx]].copy()
                feature_engineer = FeatureEngineer()
                features = feature_engineer.combine_features(current_data, for_training=True)
                
                # 移除新增的队列状态特征
                features = features.drop(['recent_wait_times', 'recent_wait_times_std', 
                                       'same_type_wait_times', 'current_hour_queue'], 
                                      axis=1, errors='ignore')
                
                # 确保所有必需的特征都存在
                required_features = ['shift_afternoon_queue_ratio', 'shift_morning_queue_ratio', 
                                  'shift_night_queue_ratio']
                for feature in required_features:
                    if feature not in features.columns:
                        features[feature] = 0
                
                # 确保特征顺序与训练时一致
                expected_features = ['煤种编号_encoded', 'hour', 'weekday', 'is_weekend', 'shift', 
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
                                  'shift_night_queue_ratio', 'hour_sin_peak', 'hour_cos_peak']
                
                features = features.reindex(columns=expected_features, fill_value=0)
                
                # 处理分类特征
                categorical_features = ['shift']
                for col in categorical_features:
                    if col in features.columns:
                        features[col] = features[col].astype('category')
                
                # 确保所有非分类特征为数值类型
                for col in features.columns:
                    if col not in categorical_features:
                        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
                
                # 创建DMatrix时启用分类特征
                feature_types = ['c' if col in categorical_features else 'q' for col in features.columns]
                dval = xgb.DMatrix(features, 
                                 feature_types=feature_types,
                                 enable_categorical=True)
                
                pred = model.predict(dval)[0]
                predictions.append(pred)
                
            except Exception as e:
                print(f"处理第 {idx} 条数据时出错: {str(e)}")
                queue_states.append({
                    'current_queue_length': 0,
                    'same_coal_type_queue': 0,
                    'recent_wait_times': 0,
                    'recent_wait_times_std': 0,
                    'same_type_wait_times': 0,
                    'current_hour_queue': 0
                })
                predictions.append(0)
        
        # 4. 创建结果DataFrame
        results = pd.DataFrame({
            '提煤单号': validation_data['提煤单号'],
            '车牌号': validation_data['车牌号'],
            '煤种编号': validation_data['煤种编号'],
            '排队日期': validation_data['排队日期'],
            '排队时间': validation_data['排队时间'],
            '实际等待时间': validation_data['等待时间'],
            '预测等待时间': np.round(predictions, 2),
            '当前队列长度': [state['current_queue_length'] for state in queue_states],
            '同煤种队列': [state['same_coal_type_queue'] for state in queue_states],
            '最近平均等待时间': [round(state['recent_wait_times'], 2) for state in queue_states],
            '最近等待时间标准差': [round(state['recent_wait_times_std'], 2) if not np.isnan(state['recent_wait_times_std']) else 0 for state in queue_states],
            '同煤种平均等待时间': [round(state['same_type_wait_times'], 2) if not np.isnan(state['same_type_wait_times']) else 0 for state in queue_states],
            '当前时段队列长度': [state['current_hour_queue'] for state in queue_states]
        })
        
        # 计算误差
        results['预测误差'] = results['预测等待时间'] - results['实际等待时间']
        results['绝对误差'] = abs(results['预测误差'])
        results['相对误差%'] = (results['绝对误差'] / results['实际等待时间'] * 100).round(2)
        
        # 5. 按车牌号统计
        vehicle_stats = results.groupby('车牌号').agg({
            '实际等待时间': ['count', 'mean'],
            '预测等待时间': 'mean',
            '绝对误差': ['mean', 'std'],
            '相对误差%': ['mean', 'std']
        }).round(2)
        
        vehicle_stats.columns = ['预测次数', '平均实际等待时间', '平均预测等待时间', 
                               '平均绝对误差', '绝对误差标准差',
                               '平均相对误差%', '相对误差标准差%']
        
        # 6. 计算整体评估指标
        metrics = {
            'MSE': mean_squared_error(results['实际等待时间'], results['预测等待时间']),
            'RMSE': np.sqrt(mean_squared_error(results['实际等待时间'], results['预测等待时间'])),
            'MAE': mean_absolute_error(results['实际等待时间'], results['预测等待时间']),
            'R2': r2_score(results['实际等待时间'], results['预测等待时间']),
            '平均相对误差%': results['相对误差%'].mean()
        }
        
        # 7. 生成误差分布
        error_distribution = pd.cut(
            results['绝对误差'],
            bins=[0, 5, 10, 15, 30, float('inf')],
            labels=['0-5分钟', '5-10分钟', '10-15分钟', '15-30分钟', '30分钟以上']
        )
        error_stats = error_distribution.value_counts().sort_index()
        error_percentages = (error_stats / len(results) * 100).round(2)
        
        # 8. 保存所有结果
        # 保存详细预测结果
        results.to_csv(f'{result_dir}/详细预测结果.csv', index=False)
        
        # 保存车辆统计信息
        vehicle_stats.to_csv(f'{result_dir}/车辆统计信息.csv')
        
        # 保存整体评估指标
        pd.DataFrame([metrics]).to_csv(f'{result_dir}/整体评估指标.csv', index=False)
        
        # 保存误差分布
        pd.DataFrame({
            '误差范围': error_stats.index,
            '预测数量': error_stats.values,
            '占比%': error_percentages.values
        }).to_csv(f'{result_dir}/误差分布.csv', index=False)
        
        # 9. 打印关键指标
        print("\n=== 模型评估报告 ===")
        print(f"\n整体评估指标:")
        print(f"均方误差(MSE): {metrics['MSE']:.2f}")
        print(f"均方根误差(RMSE): {metrics['RMSE']:.2f}")
        print(f"平均绝对误差(MAE): {metrics['MAE']:.2f}")
        print(f"R2分数: {metrics['R2']:.4f}")
        print(f"平均相对误差: {metrics['平均相对误差%']:.2f}%")
        
        print("\n误差分布:")
        for range_name, count, percentage in zip(error_stats.index, error_stats, error_percentages):
            print(f"{range_name}: {count}次 ({percentage}%)")
        
        print(f"\n车辆统计信息示例 (前5辆):")
        print(vehicle_stats.head())
        
        print(f"\n所有结果已保存至目录: {result_dir}")
        
        return results, vehicle_stats, metrics, error_stats
        
    except Exception as e:
        print(f"生成评估报告时出错: {str(e)}")
        raise

def main():
    try:
        # 1. 加载数据
        print("正在加载验证数据...")
        validation_data = pd.read_csv('data/processed/xgboost/validation_data.csv')
        
        # 2. 加载模型
        print("正在加载模型...")
        model_path = 'models/xgboost/experiment_20241220_080512/final_model/model.json'
        model = xgb.Booster()
        model.load_model(model_path)
        
        # 3. 生成评估报告
        print("正在生成评估报告...")
        results, vehicle_stats, metrics, error_stats = generate_evaluation_report(validation_data, model)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()