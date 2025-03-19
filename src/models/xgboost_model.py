import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import os
import json

from ..config.xgboost_config import XGBOOST_PARAMS

class XGBoostPredictor:
    """XGBoost模型类"""
    
    def __init__(self):
        """初始化XGBoost模型"""
        self.model = None
        self.feature_importance = None
        self.params = XGBOOST_PARAMS
        
    def train(self, train_features, train_labels, eval_features=None, eval_labels=None):
        """
        训练模型
        Args:
            train_features: 训练特征
            train_labels: 训练标签
            eval_features: 评估特征
            eval_labels: 评估标签
        Returns:
            dict: 训练评估指标
        """
        try:
            print("开始训练XGBoost模型...")
            
            # 打印GPU配置信息
            print(f"\nXGBoost配置:")
            print(f"树方法: {self.params.get('tree_method', 'default')}")
            print(f"GPU ID: {self.params.get('gpu_id', 'Not specified')}")
            print(f"预测器: {self.params.get('predictor', 'default')}")
            
            # 预处理特征
            train_features = self._preprocess_features(train_features)
            if eval_features is not None:
                eval_features = self._preprocess_features(eval_features)
            
            # 确保shift列是分类类型
            train_features['shift'] = train_features['shift'].astype('category')
            if eval_features is not None:
                eval_features['shift'] = eval_features['shift'].astype('category')
            
            # 创建DMatrix数据结构，启用分类特征支持
            dtrain = xgb.DMatrix(
                train_features, 
                label=train_labels,
                enable_categorical=True  # 启用分类特征支持
            )
            
            # 设置训练参数
            train_params = self.params.copy()
            if 'n_estimators' in train_params:
                del train_params['n_estimators']
            num_boost_round = train_params.pop('num_boost_round', 1000)
            
            # 根据是否有验证集来设置不同的训练参数
            if eval_features is not None and eval_labels is not None:
                deval = xgb.DMatrix(
                    eval_features, 
                    label=eval_labels,
                    enable_categorical=True  # 启用分类特征支持
                )
                eval_set = [(dtrain, 'train'), (deval, 'eval')]
                early_stopping_rounds = 50
            else:
                # 如果没有验证集，就不使用早停
                eval_set = [(dtrain, 'train')]
                early_stopping_rounds = None
            
            print("\n开始GPU训练...")
            training_start_time = datetime.now()
            
            # 训练模型
            self.model = xgb.train(
                params=train_params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=10  # 每10轮打印一次评估结果
            )
            
            training_end_time = datetime.now()
            training_duration = (training_end_time - training_start_time).total_seconds()
            print(f"\n训练完成！用时: {training_duration:.2f} 秒")
            
            # 计算特征重要性
            importance_type = 'gain'
            importance_scores = self.model.get_score(importance_type=importance_type)
            self.feature_importance = pd.DataFrame(
                {'feature': list(importance_scores.keys()),
                 'importance': list(importance_scores.values())}
            ).sort_values('importance', ascending=False)
            
            # 计算评估指标
            train_preds = self.model.predict(dtrain)
            metrics = {
                'train': self._calculate_metrics(train_labels, train_preds)
            }
            
            if eval_features is not None and eval_labels is not None:
                eval_preds = self.model.predict(deval)
                metrics['eval'] = self._calculate_metrics(eval_labels, eval_preds)
            
            return metrics
            
        except Exception as e:
            print(f"训练模型失败: {str(e)}")
            raise
    
    def _preprocess_features(self, features):
        """
        预处理特征
        Args:
            features: 特征DataFrame
        Returns:
            DataFrame: 预处理后的特征
        """
        features = features.copy()
        
        # 将shift列转换为分类类型，并指定可能的类别
        if 'shift' in features.columns:
            # 先填充缺失值为一个已知类别
            features['shift'] = features['shift'].fillna('unknown')
            # 确保所有可能的类别都被包含
            categories = ['night', 'morning', 'afternoon', 'unknown']
            features['shift'] = pd.Categorical(
                features['shift'],
                categories=categories
            )
        
        # 确保其他列都是数值类型并填充缺失值
        numeric_columns = [col for col in features.columns if col != 'shift']
        for col in numeric_columns:
            features[col] = pd.to_numeric(features[col], errors='coerce')
            features[col] = features[col].fillna(0)
        
        return features
    
    def predict(self, features):
        """
        预测等待时间
        Args:
            features: 特征DataFrame
        Returns:
            numpy.ndarray: 预测的等待时间
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练，请先训练模型")
            
            # 预处理特征
            features = self._preprocess_features(features)
            
            # 转换为DMatrix
            dtest = xgb.DMatrix(
                features,
                enable_categorical=True,
                feature_types=['float' if col != 'shift' else 'c' for col in features.columns]
            )
            
            # 预测
            predictions = self.model.predict(dtest)
            
            return predictions
            
        except Exception as e:
            print(f"预测失败: {str(e)}")
            raise
    
    def _calculate_metrics(self, true_values, predictions):
        """
        计算评估指标
        Args:
            true_values: 真实值
            predictions: 预测值
        Returns:
            dict: 评估指标
        """
        return {
            'mae': mean_absolute_error(true_values, predictions),
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'r2': r2_score(true_values, predictions)
        }
    
    def save_model(self, save_dir=None):
        """
        保存模型和相关结果
        Args:
            save_dir: 保存目录路径，如果为None则使用默认路径
        """
        try:
            if self.model is None:
                raise ValueError("模型未训练，无法保存")
            
            # 如果未指定保存目录，则创建时间戳目录
            if save_dir is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_dir = f'models/xgboost/experiment_{timestamp}'
            
            # 确保目录存在
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存模型
            model_path = f'{save_dir}/model.json'
            self.model.save_model(model_path)
            print(f"模型已保存到: {model_path}")
            
            # 保存特征重要性
            if self.feature_importance is not None:
                importance_path = f'{save_dir}/feature_importance.csv'
                self.feature_importance.to_csv(importance_path, index=False)
                print(f"特征重要性已保存到: {importance_path}")
            
            # 保存模型参数
            params_path = f'{save_dir}/model_params.json'
            with open(params_path, 'w', encoding='utf-8') as f:
                json.dump(self.params, f, indent=4, ensure_ascii=False)
            print(f"模型参数已保存到: {params_path}")
            
        except Exception as e:
            print(f"保存模型失败: {str(e)}")
            raise
    
    def load_model(self, model_path):
        """
        加载模型
        Args:
            model_path: 模型文件路径
        """
        try:
            self.model = xgb.Booster()
            self.model.load_model(model_path)
            print(f"成功加载模型: {model_path}")
            
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            raise