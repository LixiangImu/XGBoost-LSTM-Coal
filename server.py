import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import xgboost as xgb
import math
import warnings
from werkzeug.serving import WSGIRequestHandler

app = Flask(__name__)

# 忽略开发服务器警告
warnings.filterwarnings('ignore', message='This is a development server.')
# 或者完全禁用 Werkzeug 警告
WSGIRequestHandler.protocol_version = "HTTP/1.1"

class QueueManager:
    def __init__(self):
        # 加载模型和特征重要性
        print("初始化模型和特征工程器...")
        self.model = xgb.Booster()
        model_path = 'models/xgboost/experiment_20241220_080512/final_model/model.json'
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
        self.train_data = pd.read_csv('data/processed/xgboost/train_features.csv')
        print("加载训练数据完成")
        
        # 每个煤种独立的队列
        self.coal_type_queues = {
            'YT02WX': [],
            'YT04MM': [],
            'YT02HK': [],
            'YT05MM': [],
            'YT02XX': []
        }
        
        # 每个煤种的装煤站处理时间
        self.coal_process_times = {
            'YT02WX': 25,
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
        
        # 添加新的配置参数
        self.error_threshold = 10  # 预测误差阈值（分钟）
        self.weight_history = []   # 存储权重调整历史
        self.prediction_errors = [] # 存储预测误差
        self.abnormal_threshold = 30  # 异常判断阈值（分钟）
        
        # 初始权重配置
        self.default_weights = {
            'model': 0.6,
            'queue': 0.3,
            'base': 0.1
        }
        self.current_weights = self.default_weights.copy()

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
        """计算队列等待时间"""
        # 获取该煤种的队列
        coal_queue = self.coal_type_queues.get(coal_type, [])
        
        # 基础处理时间
        base_process_time = self.coal_process_times.get(coal_type, 25)
        
        # 如果是该煤种队列的第一辆车
        if len(coal_queue) == 0:
            return base_process_time * 0.5  # 只需要基础装煤时间
            
        # 计算前面车辆的等待时间
        queue_wait_time = base_process_time * len(coal_queue)
        
        return queue_wait_time
    
    def calculate_weights(self, hour, queue_length, coal_type):
        """动态权重计算系统"""
        weights = self.current_weights.copy()
        
        # 1. 队列长度影响
        if queue_length > 10:
            weights['queue'] += 0.1
            weights['model'] -= 0.1
        elif queue_length > 20:
            weights['queue'] += 0.2
            weights['model'] -= 0.2
            
        # 2. 时间段影响
        if 6 <= hour <= 9 or 17 <= hour <= 19:  # 高峰时段
            weights['model'] += 0.1
            weights['queue'] -= 0.1
            
        # 3. 煤种特殊处理
        same_type_count = len([x for x in self.current_queue if x['coal_type'] == coal_type])
        if same_type_count > 5:  # 同煤种积压
            weights['queue'] += 0.15
            weights['model'] -= 0.15
            
        # 确保权重和为1
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def is_abnormal_pattern(self):
        """检测异常模式"""
        if len(self.prediction_errors) < 5:
            return False
            
        # 检查最近5次预测误差
        recent_errors = self.prediction_errors[-5:]
        avg_error = np.mean(recent_errors)
        
        # 检查队列异常增长
        queue_growth_rate = len(self.current_queue) - len(self.current_queue[:-5])
        
        return avg_error > self.abnormal_threshold or queue_growth_rate > 10

    def handle_abnormal_prediction(self, coal_type, queue_time):
        """处理异常情况的预测"""
        # 1. 使用更保守的预测方式
        base_time = self.coal_process_times.get(coal_type, 25)
        queue_length = len(self.current_queue)
        
        # 2. 增加安全边际
        safety_margin = 1.5
        
        # 3. 基于最近实际等待时间调整
        recent_waits = [order['actual_wait'] for order in self.current_queue[-5:] 
                       if 'actual_wait' in order]
        
        if recent_waits:
            avg_recent_wait = np.mean(recent_waits) * safety_margin
            return max(avg_recent_wait, base_time * queue_length)
        
        return base_time * queue_length * safety_margin

    def fallback_prediction(self, coal_type):
        """降级预测机制"""
        # 使用最简单的预测方式
        base_time = self.coal_process_times.get(coal_type, 25)
        queue_length = len(self.current_queue)
        return base_time * (queue_length + 1)

    def update_model_weights(self, order_id, actual_wait_time):
        """更新模型权重（自适应学习）"""
        # 找到对应的预测记录
        order = next((x for x in self.current_queue if x['order_id'] == order_id), None)
        if not order:
            return
            
        predicted_time = order['predicted_wait']
        error = abs(actual_wait_time - predicted_time)
        
        # 存储预测误差
        self.prediction_errors.append(error)
        
        # 如果误差超过阈值，调整权重
        if error > self.error_threshold:
            # 计算权重调整方向
            if actual_wait_time > predicted_time:
                # 预测偏低，增加队列权重
                self.current_weights['queue'] = min(0.5, self.current_weights['queue'] + 0.05)
                self.current_weights['model'] = max(0.4, self.current_weights['model'] - 0.05)
            else:
                # 预测偏高，增加模型权重
                self.current_weights['model'] = min(0.7, self.current_weights['model'] + 0.05)
                self.current_weights['queue'] = max(0.2, self.current_weights['queue'] - 0.05)
            
            # 记录权重调整
            self.weight_history.append({
                'timestamp': datetime.now(),
                'weights': self.current_weights.copy(),
                'error': error
            })

    def predict_wait_time(self, order_id, coal_type, queue_time):
        """预测等待时间（整合新功能）"""
        try:
            # 清理过期订单
            self.clean_expired_orders(queue_time)
            
            # 计算等待时间
            queue_wait_time = self.calculate_queue_wait_time(coal_type, queue_time)
            features = self.prepare_features(order_id, coal_type, queue_time)
            base_wait_time = float(self.model.predict(features)[0])
            
            # 动态权重
            weights = self.calculate_weights(
                pd.to_datetime(queue_time).hour,
                len(self.coal_type_queues[coal_type]),  # 只考虑同煤种队列长度
                coal_type
            )
            
            # 计算最终等待时间
            wait_time = (
                weights['model'] * base_wait_time +
                weights['queue'] * queue_wait_time +
                weights['base'] * self.coal_process_times.get(coal_type, 25)
            )
            
            # 更新该煤种的队列
            order_data = {
                'order_id': order_id,
                'coal_type': coal_type,
                'queue_time': queue_time,
                'predicted_wait': wait_time
            }
            self.coal_type_queues[coal_type].append(order_data)
            
            # 计算预计叫号时间
            queue_dt = datetime.strptime(queue_time, '%Y-%m-%d %H:%M:%S')
            call_time = queue_dt + timedelta(minutes=wait_time)
            
            return {
                'wait_minutes': round(wait_time, 2),
                'call_time': call_time.strftime('%Y-%m-%d %H:%M:%S'),
                'queue_info': {
                    'total_queue': sum(len(q) for q in self.coal_type_queues.values()),
                    'same_type_queue': len(self.coal_type_queues[coal_type]),
                    'base_process_time': self.coal_process_times.get(coal_type, 25),
                    'queue_wait_time': round(queue_wait_time, 2),
                    'model_prediction': round(base_wait_time, 2),
                    'weights_used': weights,
                    'prediction_confidence': 'normal'
                }
            }
            
        except Exception as e:
            print(f"预测过程发生错误: {str(e)}")
            return self.fallback_prediction(coal_type, queue_time)

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

@app.route('/update_wait_time', methods=['POST'])
def update_wait_time():
    try:
        data = request.get_json()
        order_id = data['order_id']
        actual_wait_time = data['actual_wait_time']
        
        queue_manager.update_model_weights(order_id, actual_wait_time)
        
        return jsonify({
            'status': 'success',
            'message': '等待时间更新成功'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    print("启动等待时间预测服务...")
    try:
        app.run(host='0.0.0.0', port=5001)
    except OSError as e:
        if "Address already in use" in str(e):
            print("端口5000已被占用,尝试使用其他端口...")
            # 尝试其他端口
            for port in range(5001, 5010):
                try:
                    app.run(host='0.0.0.0', port=port)
                    break
                except OSError:
                    continue
        else:
            raise e