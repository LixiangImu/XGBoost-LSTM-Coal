import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
import json

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.xgboost_model import XGBoostPredictor

def save_training_results(metrics, save_dir):
    """保存训练结果"""
    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存交叉验证结果
    cv_results = {
        'mae_mean': float(np.mean(metrics['mae'])),
        'mae_std': float(np.std(metrics['mae'])),
        'rmse_mean': float(np.mean(metrics['rmse'])),
        'rmse_std': float(np.std(metrics['rmse'])),
        'r2_mean': float(np.mean(metrics['r2'])),
        'r2_std': float(np.std(metrics['r2']))
    }
    
    results_path = f'{save_dir}/cv_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(cv_results, f, indent=4, ensure_ascii=False)
    print(f"\n交叉验证结果已保存到: {results_path}")

def main():
    """主训练函数"""
    try:
        # 加载特征数据
        print("加载特征数据...")
        features = pd.read_csv('data/processed/xgboost/train_features.csv')
        
        # 加载原始处理后的数据
        processed_data = pd.read_csv('data/processed/xgboost/train_data.csv')
        labels = processed_data['等待时间'].values
        
        print(f"特征数据大小: {len(features)}")
        print(f"标签数据大小: {len(labels)}")
        
        # 确保数据对齐
        if len(features) != len(labels):
            raise ValueError(f"数据长度不匹配: 特征={len(features)}, 标签={len(labels)}")
            
        # 1. 首先划分出测试集（20%）
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, labels,
            test_size=0.2,
            random_state=42,
            shuffle=True
        )
        
        # 2. 将剩余数据划分为训练集（75%）和验证集（25%）
        # 这样最终比例约为：训练集60%，验证集20%，测试集20%
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=0.25,
            random_state=42,
            shuffle=True
        )
        
        print(f"数据集大小:")
        print(f"训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 创建时间戳文件夹
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = f'models/xgboost/experiment_{timestamp}'
        os.makedirs(experiment_dir, exist_ok=True)
        
        # === 第一部分：在训练集上进行交叉验证 ===
        print("\n开始交叉验证...")
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_scores = {
            'mae': [],
            'rmse': [],
            'r2': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"\n训练折次 {fold + 1}/{n_splits}")
            
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # 初始化模型
            model = XGBoostPredictor()
            
            # 训练模型
            metrics = model.train(
                train_features=X_fold_train,
                train_labels=y_fold_train,
                eval_features=X_fold_val,
                eval_labels=y_fold_val
            )
            
            cv_scores['mae'].append(metrics['eval']['mae'])
            cv_scores['rmse'].append(metrics['eval']['rmse'])
            cv_scores['r2'].append(metrics['eval']['r2'])
            
            # 保存每个折的结果
            fold_dir = f'{experiment_dir}/fold_{fold + 1}'
            os.makedirs(fold_dir, exist_ok=True)
            model.save_model(fold_dir)
        
        # 打印交叉验证结果
        print("\n交叉验证结果:")
        print(f"平均 MAE: {np.mean(cv_scores['mae']):.4f} ± {np.std(cv_scores['mae']):.4f}")
        print(f"平均 RMSE: {np.mean(cv_scores['rmse']):.4f} ± {np.std(cv_scores['rmse']):.4f}")
        print(f"平均 R2: {np.mean(cv_scores['r2']):.4f} ± {np.std(cv_scores['r2']):.4f}")
        
        # === 第二部分：在验证集上评估 ===
        print("\n在验证集上评估...")
        val_model = XGBoostPredictor()
        val_metrics = val_model.train(
            train_features=X_train,
            train_labels=y_train,
            eval_features=X_val,
            eval_labels=y_val
        )
        
        print("\n验证集结果:")
        print(f"MAE: {val_metrics['eval']['mae']:.4f}")
        print(f"RMSE: {val_metrics['eval']['rmse']:.4f}")
        print(f"R2: {val_metrics['eval']['r2']:.4f}")
        
        # === 第三部分：训练最终模型并在测试集上评估 ===
        print("\n训练最终模型...")
        final_model = XGBoostPredictor()
        
        # 使用训练集和验证集的组合训练最终模型
        final_metrics = final_model.train(
            train_features=X_train_val,
            train_labels=y_train_val
        )
        
        # 在测试集上评估
        test_predictions = final_model.predict(X_test)
        test_metrics = final_model._calculate_metrics(y_test, test_predictions)
        
        print("\n测试集结果:")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"R2: {test_metrics['r2']:.4f}")
        
        # 保存最终模型
        final_model_dir = f'{experiment_dir}/final_model'
        os.makedirs(final_model_dir, exist_ok=True)
        final_model.save_model(final_model_dir)
        
        # 保存评估结果
        results = {
            'cross_validation': {
                'mae_mean': float(np.mean(cv_scores['mae'])),
                'mae_std': float(np.std(cv_scores['mae'])),
                'rmse_mean': float(np.mean(cv_scores['rmse'])),
                'rmse_std': float(np.std(cv_scores['rmse'])),
                'r2_mean': float(np.mean(cv_scores['r2'])),
                'r2_std': float(np.std(cv_scores['r2']))
            },
            'validation': val_metrics['eval'],
            'test': test_metrics
        }
        
        with open(f'{final_model_dir}/evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\n实验结果已保存到: {experiment_dir}")
        print(f"最终模型已保存到: {final_model_dir}")
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()