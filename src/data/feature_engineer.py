import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..config.config import FEATURE_CONFIG

class FeatureEngineer:
    """特征工程类，负责特征的提取、转换和生成"""
    
    def __init__(self):
        """初始化特征工程器"""
        self.window_size = FEATURE_CONFIG['WINDOW_SIZE']
        self.time_features = FEATURE_CONFIG['TIME_FEATURES']
        self.primary_features = FEATURE_CONFIG['PRIMARY_FEATURES']
        self.feature_cache = {}  # 用于存储计算的特征
        
    def extract_time_features(self, df):
        """
        提取时间特征
        Args:
            df: 包含时间列的DataFrame
        Returns:
            DataFrame: 时间特征
        """
        try:
            features = pd.DataFrame()
            
            # 提取小时
            features['hour'] = df['排队时间'].dt.hour
            
            # 提取星期
            features['weekday'] = df['排队时间'].dt.weekday
            
            # 是否周末
            features['is_weekend'] = features['weekday'].isin([5, 6]).astype(int)
            
            # 时间段划分（早班/中班/晚班）- 使用字符串类型
            features['shift'] = pd.cut(
                features['hour'],
                bins=[-1, 7, 15, 23],
                labels=['night', 'morning', 'afternoon']
            ).astype(str)  # 转换为字符串类型
            
            # 是否高峰时段（根据历史数据统计得出的高峰期）
            peak_hours = [7, 8, 9, 13, 14, 15]
            features['is_peak_hour'] = features['hour'].isin(peak_hours).astype(int)
            
            return features
            
        except Exception as e:
            print(f"提取时间特征失败: {str(e)}")
            raise
    
    def create_window_features(self, df, current_time=None):
        """
        创建滑动时间窗口特征
        Args:
            df: 数据DataFrame
            current_time: 当前时间点（用于实时预测）
        Returns:
            DataFrame: 时间窗口特征
        """
        try:
            # 按时间排序
            df = df.sort_values('排队时间')
            
            if current_time is not None:
                # 只使用当前时间之前的数据
                df = df[df['排队时间'] <= current_time]
            
            # 计算滑动窗口统计特征
            window_features = df.groupby(
                pd.Grouper(key='排队时间', freq=self.window_size)
            ).agg({
                '等待时间': ['mean', 'std', 'count'],
                '煤种编号': 'nunique',
                '处理时间': ['mean', 'std']
            }).fillna(0)
            
            # 重命名列
            window_features.columns = [
                'wait_time_mean', 'wait_time_std', 'vehicle_count',
                'coal_type_count', 'process_time_mean', 'process_time_std'
            ]
            
            # 计算趋势特征
            window_features['wait_time_trend'] = window_features['wait_time_mean'].pct_change()
            window_features['vehicle_count_trend'] = window_features['vehicle_count'].pct_change()
            
            return window_features
            
        except Exception as e:
            print(f"创建时间窗口特征失败: {str(e)}")
            raise
    
    def create_queue_features(self, df, current_time=None):
        """
        创建队列特征
        Args:
            df: 数据DataFrame
            current_time: 当前时间点（用于实时预测）
        Returns:
            DataFrame: 队列特征
        """
        try:
            queue_features = pd.DataFrame()
            
            if current_time is not None:
                # 获取当前排队状态
                current_queue = df[
                    (df['排队时间'] <= current_time) & 
                    (df['叫号时间'] > current_time)
                ]
            else:
                current_queue = df
            
            # 计算当前队列长度
            queue_features['current_queue_length'] = len(current_queue)
            
            # 计算同煤种排队数量
            queue_features['same_coal_type_count'] = current_queue.groupby(
                '煤种编号'
            ).cumcount()
            
            # 计算各煤种的等待车辆数
            coal_type_counts = current_queue['煤种编号'].value_counts()
            queue_features['coal_type_queue_length'] = coal_type_counts
            
            return queue_features
            
        except Exception as e:
            print(f"创建队列特征失败: {str(e)}")
            raise
    
    def create_coal_type_features(self, df, current_time=None):
        """
        创建煤种相关特征
        Args:
            df: 数据DataFrame
            current_time: 当前时间点（用于实时预测）
        Returns:
            DataFrame: 煤种特征
        """
        try:
            if current_time is not None:
                historical_data = df[df['排队时间'] < current_time]
            else:
                historical_data = df
            
            # 使用编码后的煤种编号进行分组
            coal_type_stats = historical_data.groupby('煤种编号_encoded').agg({
                '等待时间': ['mean', 'std', 'median'],
                '处理时间': ['mean', 'std']
            })
            
            # 展平多级索引列名
            coal_type_stats.columns = [
                'coal_type_wait_mean', 'coal_type_wait_std', 
                'coal_type_wait_median', 'coal_type_process_mean',
                'coal_type_process_std'
            ]
            
            return coal_type_stats
            
        except Exception as e:
            print(f"创建煤种特征失败: {str(e)}")
            raise
    
    def combine_features(self, df, for_training=True, current_time=None):
        """
        组合所有特征
        Args:
            df: 原始数据DataFrame
            for_training: 是否用于训练
            current_time: 当前时间点（用于实时预测）
        Returns:
            DataFrame: 组合后的特征
        """
        try:
            # 提取时间特征
            time_features = self.extract_time_features(df)
            
            # 将shift列转换为字符串类型，避免分类变量的问题
            time_features['shift'] = time_features['shift'].astype(str)
            
            # 创建基础特征集
            features = pd.DataFrame(index=df.index)
            features['煤种编号_encoded'] = df['煤种编号_encoded']
            features['等待时间'] = df['等待时间']  # 添加等待时间列
            features = pd.concat([features, time_features], axis=1)
            
            # 添加煤种统计特征
            coal_type_features = self.create_coal_type_features(df, current_time)
            coal_type_features = coal_type_features.reset_index()
            coal_type_features = coal_type_features.rename(columns={'index': '煤种编号_encoded'})
            
            # 使用煤种编号_encoded进行合并
            features = features.merge(
                coal_type_features,
                on='煤种编号_encoded',
                how='left'
            )
            
            # 添加新的时间特征
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
            features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
            
            # 计算当前队列长度
            if for_training:
                features['current_queue_length'] = df.groupby(
                    pd.Grouper(key='排队时间', freq=self.window_size)
                ).cumcount()
                
                # 计算同煤种队列长度
                features['same_coal_type_queue'] = df.groupby(
                    ['煤种编号_encoded', pd.Grouper(key='排队时间', freq=self.window_size)]
                ).cumcount()
            else:
                # 对于预测，使用最近的统计值
                features['current_queue_length'] = df.groupby(
                    pd.Grouper(key='排队时间', freq=self.window_size)
                ).size().iloc[-1] if len(df) > 0 else 0
                
                features['same_coal_type_queue'] = df.groupby(
                    ['煤种编号_encoded', pd.Grouper(key='排队时间', freq=self.window_size)]
                ).size().iloc[-1] if len(df) > 0 else 0
            
            # 添加交互特征
            features['peak_coal_type'] = features['is_peak_hour'] * features['coal_type_wait_mean']
            features['queue_wait_ratio'] = features['current_queue_length'] / (features['coal_type_wait_mean'] + 1)
            features['type_queue_ratio'] = features['same_coal_type_queue'] / (features['current_queue_length'] + 1)
            
            # 添加统计特征
            if for_training:
                features['wait_time_rolling_mean'] = df.groupby('煤种编号_encoded')['等待时间'].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
                features['wait_time_rolling_std'] = df.groupby('煤种编号_encoded')['等待时间'].transform(
                    lambda x: x.rolling(window=3, min_periods=1).std()
                )
            
            # 添加多个时间窗口的统计特征
            if for_training:
                # 多窗口滚动统计
                for window in [3, 5, 7]:
                    # 按煤种分组的滚动统计
                    features[f'wait_time_rolling_mean_{window}'] = df.groupby('煤种编号_encoded')['等待时间'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    features[f'wait_time_rolling_std_{window}'] = df.groupby('煤种编号_encoded')['等待时间'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                    
                    # 按时间段的滚动统计
                    features[f'hour_rolling_mean_{window}'] = features.groupby('hour')['等待时间'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    
                    # 按班次的滚动统计
                    features[f'shift_rolling_mean_{window}'] = features.groupby('shift')['等待时间'].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                
                # 指数加权移动平均
                features['wait_time_ewm_mean'] = df.groupby('煤种编号_encoded')['等待时间'].transform(
                    lambda x: x.ewm(span=5).mean()
                )
                features['wait_time_ewm_std'] = df.groupby('煤种编号_encoded')['等待时间'].transform(
                    lambda x: x.ewm(span=5).std()
                )
                
                # 时间差特征
                features['time_since_last_same_type'] = df.groupby('煤种编号_encoded')['排队时间'].transform(
                    lambda x: x.diff().dt.total_seconds() / 60
                )
                
                # 队列变化率
                features['queue_change_rate'] = features['current_queue_length'].diff()
                features['type_queue_change_rate'] = features['same_coal_type_queue'].diff()
            
            # 增强的交互特征
            features['peak_queue_ratio'] = features['is_peak_hour'] * features['queue_wait_ratio']
            features['hour_queue_length'] = features['hour'] * features['current_queue_length']
            features['type_wait_queue'] = features['coal_type_wait_mean'] * features['current_queue_length']
            
            # 为每个班次创建单独的队列比率特征
            shift_dummies = pd.get_dummies(features['shift'], prefix='shift')
            for shift_col in shift_dummies.columns:
                features[f'{shift_col}_queue_ratio'] = shift_dummies[shift_col] * features['queue_wait_ratio']
            
            # 时间周期性特征的交互
            features['hour_sin_peak'] = features['hour_sin'] * features['is_peak_hour']
            features['hour_cos_peak'] = features['hour_cos'] * features['is_peak_hour']
            
            # 填充缺失值
            features = features.fillna(0)
            
            # 删除等待时间列（因为这是标签）
            if '等待时间' in features.columns:
                features = features.drop('等待时间', axis=1)
            
            return features
            
        except Exception as e:
            print(f"组合特征失败: {str(e)}")
            raise
    
    def prepare_features_for_prediction(self, order_id, coal_type, queue_time, historical_data):
        """
        为实时预测准备特征
        Args:
            order_id: 提煤单号
            coal_type: 煤种编号
            queue_time: 排队时间
            historical_data: 历史数据DataFrame
        Returns:
            DataFrame: 用于预测的特征
        """
        try:
            # 创建当前记录的DataFrame
            current_record = pd.DataFrame({
                '提煤单号': [order_id],
                '煤种编号': [coal_type],
                '排队时间': [pd.to_datetime(queue_time)]
            })
            
            # 合并历史数据
            df = pd.concat([historical_data, current_record])
            
            # 生成特征
            features = self.combine_features(
                df,
                for_training=False,
                current_time=pd.to_datetime(queue_time)
            )
            
            return features.iloc[-1:]
            
        except Exception as e:
            print(f"准备预测特征失败: {str(e)}")
            raise
    
    def save_feature_cache(self):
        """保存特征缓存"""
        pass
    
    def load_feature_cache(self):
        """加载特征缓存"""
        pass
