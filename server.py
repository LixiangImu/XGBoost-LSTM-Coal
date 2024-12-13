import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import xgboost as xgb
import math

app = Flask(__name__)

class QueueManager:
    def __init__(self):
        # 加载模型和特征重要性
        print("初始化模型和特征工程器...")
        self.model = xgb.Booster()
        model_path = 'models/xgboost/experiment_20241213_010025/final_model/model.json'
        self.model.load_model(model_path)
        print(f"加载模型: {model_path}")
        
        # 加载特征重要性
        self.feature_importance = pd.read_csv('models/xgboost/experiment_20241213_010025/final_model/feature_importance.csv')
        print("加载特征重要性完成")
        
        # 从原始数据中获取煤种列表
        raw_data = pd.read_csv('data/raw/queue_data_offset_sorted_utf8.csv')
        self.coal_types = set(raw_data['煤种编号'].unique())
        print(f"可用煤种类型: {self.coal_types}")
        
        # 加载训练数据以获取统计特征
        self.train_data = pd.read_csv('data/processed/train_features.csv')
        print("加载训练数据完成")
        
        # 煤种处理时间配置
        self.coal_process_times = {
            'YT02WX': 25,  # 分钟
            'YT04MM': 30,
            'YT02HK': 28,
            'YT05MM': 27,
            'YT02XX': 26
        }
        
        # 队列状态
        self.current_queue = []
        self.queue_timeout = 180  # 清理过期订单的时间阈值（分钟）
        
        # 初始化历史统计数据
        self.historical_stats = {
            'shift_stats': {},  # 班次统计
            'hour_stats': {},   # 小时统计
            'coal_type_stats': {}  # 煤种统计
        }
        
    def update_historical_stats(self, order_data):
        """更新历史统计数据"""
        dt = pd.to_datetime(order_data['queue_time'])
        hour = dt.hour
        shift = self.get_shift(hour)
        coal_type = order_data['coal_type']
        wait_time = order_data['predicted_wait']
        
        # 更新班次统计
        if shift not in self.historical_stats['shift_stats']:
            self.historical_stats['shift_stats'][shift] = []
        self.historical_stats['shift_stats'][shift].append(wait_time)
        
        # 更新小时统计
        if hour not in self.historical_stats['hour_stats']:
            self.historical_stats['hour_stats'][hour] = []
        self.historical_stats['hour_stats'][hour].append(wait_time)
        
        # 更新煤种统计
        if coal_type not in self.historical_stats['coal_type_stats']:
            self.historical_stats['coal_type_stats'][coal_type] = []
        self.historical_stats['coal_type_stats'][coal_type].append(wait_time)
        
        # 保持最近100条记录
        max_history = 100
        for stat_type in self.historical_stats.values():
            for key in stat_type:
                stat_type[key] = stat_type[key][-max_history:]
    
    def get_rolling_stats(self, stat_type, key, window_size):
        """获取滚动统计值"""
        if key not in self.historical_stats[stat_type]:
            return 0
        data = self.historical_stats[stat_type][key]
        if len(data) < window_size:
            return np.mean(data) if data else 0
        return np.mean(data[-window_size:])
    
    def prepare_features(self, order_id, coal_type, queue_time):
        """准备模型输入特征"""
        dt = pd.to_datetime(queue_time)
        hour = dt.hour
        weekday = dt.weekday()
        shift = self.get_shift(hour)
        
        features = {
            # 高重要性特征
            'shift_rolling_mean_3': self.get_rolling_stats('shift_stats', shift, 3),
            'hour_rolling_mean_3': self.get_rolling_stats('hour_stats', hour, 3),
            'hour_rolling_mean_5': self.get_rolling_stats('hour_stats', hour, 5),
            'shift': shift,
            'shift_rolling_mean_5': self.get_rolling_stats('shift_stats', shift, 5),
            'hour_cos': math.cos(2 * math.pi * hour / 24),
            'coal_type_wait_median': np.median(self.historical_stats['coal_type_stats'].get(coal_type, [25])),
            
            # 时间特征
            'hour': hour,
            'weekday': weekday,
            'is_weekend': 1 if weekday >= 5 else 0,
            'is_peak_hour': self.is_peak_hour(hour),
            'hour_sin': math.sin(2 * math.pi * hour / 24),
            'weekday_sin': math.sin(2 * math.pi * weekday / 7),
            'weekday_cos': math.cos(2 * math.pi * weekday / 7),
            
            # 煤种编码
            '煤种编号_encoded': self.encode_coal_type(coal_type),
            
            # 队列特征
            'current_queue_length': len(self.current_queue),
            'same_coal_type_queue': len([x for x in self.current_queue if x['coal_type'] == coal_type])
        }
        
        # 确保所有特征都存在
        df = pd.DataFrame([features])
        df = df.reindex(columns=self.train_data.columns, fill_value=0)
        
        return xgb.DMatrix(df)
    
    def encode_coal_type(self, coal_type):
        """简单的标签编码"""
        coal_types_list = sorted(list(self.coal_types))
        return coal_types_list.index(coal_type)
    
    def get_shift(self, hour):
        """获取班次"""
        if 6 <= hour < 14:
            return 0  # 早班
        elif 14 <= hour < 22:
            return 1  # 中班
        else:
            return 2  # 晚班
    
    def is_peak_hour(self, hour):
        """判断是否高峰时段"""
        return 1 if (8 <= hour < 11) or (14 <= hour < 17) else 0
    
    def validate_request(self, order_id, coal_type, queue_time):
        """验证请求参数"""
        if not (order_id.startswith('TP') or order_id.startswith('HP')):
            raise ValueError("提煤单号格式错误")
            
        if coal_type not in self.coal_types:
            raise ValueError(f"未知的煤种编号: {coal_type}")
            
        try:
            datetime.strptime(queue_time, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError("时间格式错误")

    def clean_expired_orders(self, current_time):
        """清理过期订单"""
        current_dt = pd.to_datetime(current_time)
        self.current_queue = [
            order for order in self.current_queue
            if (current_dt - pd.to_datetime(order['queue_time'])).total_seconds() / 60 < self.queue_timeout
        ]
    
    def calculate_queue_wait_time(self, coal_type, queue_time):
        """计算实际队列等待时间"""
        same_type_queue = [
            order for order in self.current_queue
            if order['coal_type'] == coal_type
        ]
        
        base_process_time = self.coal_process_times.get(coal_type, 25)
        queue_length = len(same_type_queue)
        
        # ��础等待时间
        wait_time = base_process_time * (queue_length + 1)
        
        # 考虑煤种切换的额外时间
        if queue_length == 0 and len(self.current_queue) > 0:
            last_order = self.current_queue[-1]
            if last_order['coal_type'] != coal_type:
                wait_time += 10  # 煤种切换额外等待时间
        
        return wait_time
    
    def predict_wait_time(self, order_id, coal_type, queue_time):
        """预测等待时间"""
        # 清理过期订单
        self.clean_expired_orders(queue_time)
        
        # 计算基础队列等待时间
        queue_wait_time = self.calculate_queue_wait_time(coal_type, queue_time)
        
        # 特征工程处理
        features = self.prepare_features(order_id, coal_type, queue_time)
        
        # 使用模型预测基础等待时间
        base_wait_time = float(self.model.predict(features)[0])
        
        # 根据特征重要性加权计算最终等待时间
        importance_sum = self.feature_importance['importance'].sum()
        top_features = self.feature_importance.nlargest(5, 'importance')
        
        # 计算加权系数
        weight_model = 0.6
        weight_queue = 0.3
        weight_base = 0.1
        
        final_wait_time = (
            weight_model * base_wait_time +
            weight_queue * queue_wait_time +
            weight_base * self.coal_process_times.get(coal_type, 25)
        )
        
        # 更新队列和历史统计
        order_data = {
            'order_id': order_id,
            'coal_type': coal_type,
            'queue_time': queue_time,
            'predicted_wait': final_wait_time
        }
        
        self.current_queue.append(order_data)
        self.update_historical_stats(order_data)
        
        # 计算预计叫号时间
        queue_dt = datetime.strptime(queue_time, '%Y-%m-%d %H:%M:%S')
        call_time = queue_dt + timedelta(minutes=final_wait_time)
        
        return {
            'wait_minutes': round(final_wait_time, 2),
            'call_time': call_time.strftime('%Y-%m-%d %H:%M:%S'),
            'queue_info': {
                'total_queue': len(self.current_queue),
                'same_type_queue': len([x for x in self.current_queue if x['coal_type'] == coal_type]),
                'base_process_time': self.coal_process_times.get(coal_type, 25),
                'queue_wait_time': round(queue_wait_time, 2),
                'model_prediction': round(base_wait_time, 2),
                'top_features': top_features.to_dict('records')
            }
        }

# 创建全局队列管理器实例
queue_manager = QueueManager()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 验证输入
        queue_manager.validate_request(
            data['order_id'],
            data['coal_type'],
            data['queue_time']
        )
        
        # 预测等待时间
        result = queue_manager.predict_wait_time(
            data['order_id'],
            data['coal_type'], 
            data['queue_time']
        )
        
        return jsonify({
            'status': 'success',
            'data': {
                'order_id': data['order_id'],
                'coal_type': data['coal_type'],
                'queue_time': data['queue_time'],
                'wait_minutes': result['wait_minutes'],
                'call_time': result['call_time'],
                'queue_info': result['queue_info']
            }
        })
        
    except Exception as e:
        import traceback
        print("错误详情:")
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    print("启动等待时间预测服务...")
    app.run(host='0.0.0.0', port=5000)