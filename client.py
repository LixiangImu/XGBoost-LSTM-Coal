import requests
import json
from datetime import datetime, timedelta
import time
import random
import pandas as pd

# 服务器配置
SERVER_HOST = '127.0.0.1'  
SERVER_PORT = 5000

def get_coal_types():
    """从原始数据中获取煤种列表"""
    raw_data = pd.read_csv('data/processed/xgboost/validation_data.csv')
    coal_types = list(raw_data['煤种编号'].unique())
    print(f"可用煤种类型: {coal_types}")
    return coal_types

def generate_order_id():
    """生成提煤单号"""
    # 生成日期部分
    date_str = datetime.now().strftime('%y%m%d')
    # 生成5位序号
    sequence = str(random.randint(1, 99999)).zfill(5)
    # 组合成完整的提煤单号
    prefix = random.choice(['TP', 'HP'])
    return f"{prefix}{date_str}{sequence}"

def format_time_period(minutes):
    """格式化时间段"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if hours > 0:
        return f"{hours}小时{mins}分钟"
    return f"{mins}分钟"

def simulate_driver_checkin():
    """模拟司机打卡"""
    url = f'http://{SERVER_HOST}:{SERVER_PORT}/predict'
    
    # 生成模拟数据
    order_id = generate_order_id()
    coal_type = random.choice(COAL_TYPES)
    queue_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 准备请求数据
    data = {
        'order_id': order_id,
        'coal_type': coal_type,
        'queue_time': queue_time
    }
    
    print("\n" + "="*50)
    print(f"模拟司机打卡: {order_id}")
    print("\n发送请求数据:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result['status'] == 'success':
                print("\n=== 等待时间预测结果 ===")
                print(f"提煤单号: {result['data']['order_id']}")
                print(f"煤种编号: {result['data']['coal_type']}")
                print(f"排队时间: {result['data']['queue_time']}")
                print(f"预计等待: {format_time_period(result['data']['wait_minutes'])}")
                print(f"预计叫号: {result['data']['call_time']}")
                
                print("\n=== 队列详情 ===")
                queue_info = result['data']['queue_info']
                print(f"当前总队列: {queue_info['total_queue']} 辆")
                print(f"同煤种队列: {queue_info['same_type_queue']} 辆")
                print(f"基础处理时间: {format_time_period(queue_info['base_process_time'])}")
                
                if 'queue_wait_time' in queue_info:
                    print(f"队列等待时间: {format_time_period(queue_info['queue_wait_time'])}")
                if 'model_prediction' in queue_info:
                    print(f"模型基础预测: {format_time_period(queue_info['model_prediction'])}")
                if 'prediction_confidence' in queue_info:
                    print(f"预测置信度: {queue_info['prediction_confidence']}")
                
                if queue_info.get('is_abnormal'):
                    print("\n警告: 检测到异常模式")
                    
            else:
                print(f"\n预测失败: {result.get('message', '未知错误')}")
                
    except Exception as e:
        print(f"\n发送请求失败: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        print("="*50)

def main():
    """主函数"""
    print("\n=== 启动司机打卡模拟程序 ===")
    
    try:
        # 获取煤种列表
        global COAL_TYPES
        COAL_TYPES = get_coal_types()
        
        while True:
            simulate_driver_checkin()
            # 随机等待5-15秒
            wait_time = random.randint(5, 15)
            print(f"\n等待 {wait_time} 秒后发送下一个请求...")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"\n程序异常: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()