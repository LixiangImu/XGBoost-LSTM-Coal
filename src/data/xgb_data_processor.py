import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ..config.xgboost_config import FEATURE_CONFIG, DATA_CONFIG

class DataProcessor:
    """数据处理类，负责数据的加载、清洗和预处理"""
    
    def __init__(self):
        """初始化数据处理器"""
        self.coal_type_encoder = LabelEncoder()
        self.processed_data = None
        self.feature_columns = None
        self.target_column = FEATURE_CONFIG['TARGET']
        
    def load_data(self, file_path):
        """
        加载原始数据
        Args:
            file_path: 数据文件路径
        Returns:
            DataFrame: 加载的原始数据
        """
        try:
            df = pd.read_csv(file_path)
            print(f"成功加载数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            raise
    
    def process_raw_data(self, df):
        """
        处理原始数据，包括时间转换和等待时间计算
        Args:
            df: 原始数据DataFrame
        Returns:
            DataFrame: 处理后的数据
        """
        try:
            # 复制数据，避免修改原始数据
            df = df.copy()
            
            # 转换时间格式
            df['排队时间'] = pd.to_datetime(df['排队日期'] + ' ' + df['排队时间'])
            df['叫号时间'] = pd.to_datetime(df['叫号日期'] + ' ' + df['叫号时间'])
            df['入口时间'] = pd.to_datetime(df['入口操作日期'] + ' ' + df['入口操作时间'])
            df['出口时间'] = pd.to_datetime(df['出口操作日期'] + ' ' + df['出口操作时间'])
            
            # 计算等待时间（分钟）- 从排队到叫号的时间
            df['等待时间'] = (df['叫号时间'] - df['排队时间']).dt.total_seconds() / 60
            
            # 计算处理时间（分钟）- 从入口到出口的时间
            df['处理时间'] = (df['出口时间'] - df['入口时间']).dt.total_seconds() / 60
            
            # 移除异常值
            df = self._remove_outliers(df)
            
            return df
            
        except Exception as e:
            print(f"处理原始数据失败: {str(e)}")
            raise
    
    def _remove_outliers(self, df):
        """
        移除异常值
        Args:
            df: DataFrame
        Returns:
            DataFrame: 清理后的数据
        """
        # 移除等待时间小于0或异常大的记录
        df = df[df['等待时间'] >= 0]
        df = df[df['等待时间'] <= 1440]  # 最大等待时间24小时
        
        # 移除处理时间异常的记录
        df = df[df['处理时间'] >= 0]
        df = df[df['处理时间'] <= 240]  # 最大处理时间4小时
        
        return df
    
    def prepare_data(self, df):
        """
        准备数据，包括特征处理和编码
        Args:
            df: 原始数据DataFrame
        Returns:
            DataFrame: 处理后的数据
        """
        try:
            # 处理原始数据
            processed_df = self.process_raw_data(df)
            
            # 对煤种编号进行编码
            processed_df['煤种编号_encoded'] = self.coal_type_encoder.fit_transform(
                processed_df['煤种编号']
            )
            
            # 提取提煤单号中的批次信息
            processed_df['批次'] = processed_df['提煤单号'].str[:4]
            
            # 保存处理后的数据
            self.processed_data = processed_df
            
            return processed_df
            
        except Exception as e:
            print(f"准备数据失败: {str(e)}")
            raise
    
    def split_train_test(self, df):
        """
        分割训练集和测试集
        Args:
            df: 处理后的DataFrame
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        try:
            # 按时间顺序排序
            df = df.sort_values('排队时间')
            
            # 分割训练集和测试集
            train_size = int(len(df) * (1 - DATA_CONFIG['TEST_SIZE']))
            
            # 时间序列分割
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            return train_df, test_df
            
        except Exception as e:
            print(f"分割训练测试集失败: {str(e)}")
            raise
    
    def get_feature_names(self):
        """
        获取特征列名
        Returns:
            list: 特征列名列表
        """
        if self.feature_columns is None:
            raise ValueError("特征列未初始化，请先处理数据")
        return self.feature_columns
    
    def save_processed_data(self, output_path):
        """
        保存处理后的数据
        Args:
            output_path: 输出文件路径
        """
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            print(f"处理后的数据已保存到: {output_path}")
        else:
            print("没有可保存的处理后数据")
    
    def load_processed_data(self, file_path):
        """
        加载已处理的数据
        Args:
            file_path: 文件路径
        Returns:
            DataFrame: 处理后的数据
        """
        try:
            df = pd.read_csv(file_path)
            # 转换时间列
            time_columns = ['排队时间', '叫号时间', '入口时间', '出口时间']
            for col in time_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            self.processed_data = df
            return df
            
        except Exception as e:
            print(f"加载处理后的数据失败: {str(e)}")
            raise
