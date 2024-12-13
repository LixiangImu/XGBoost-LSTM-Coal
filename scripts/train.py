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
        train_features = pd.read_csv('data/processed/train_features.csv')
        test_features = pd.read_csv('data/processed/test_features.csv')
        
        # 加载原始处理后的数据
        processed_data = pd.read_csv('data/processed/processed_data.csv')
        
        # 使用与特征生成时相同的训练集索引
        train_size = len(train_features)
        train_data = processed_data.iloc[:train_size]
        train_labels = train_data['等待时间'].values
        
        # 验证长度匹配
        if len(train_features) != len(train_labels):
            print(f"特征和标签长度不匹配: 特征长度={len(train_features)}, 标签长度={len(train_labels)}")
            raise ValueError("特征和标签长度必须匹配")
        
        print(f"训练数据大小: 特征={train_features.shape}, 标签={train_labels.shape}")
        
        # 创建时间戳文件夹
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = f'models/xgboost/experiment_{timestamp}'
        os.makedirs(experiment_dir, exist_ok=True)
        
        # === 第一部分：交叉验证 ===
        print("\n开始交叉验证...")
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        mae_scores = []
        rmse_scores = []
        r2_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
            print(f"\n训练折次 {fold + 1}/{n_splits}")
            
            X_train, X_val = train_features.iloc[train_idx], train_features.iloc[val_idx]
            y_train, y_val = train_labels[train_idx], train_labels[val_idx]
            
            # 初始化模型
            model = XGBoostPredictor()
            
            # 训练模型
            metrics = model.train(
                train_features=X_train,
                train_labels=y_train,
                eval_features=X_val,
                eval_labels=y_val
            )
            
            mae_scores.append(metrics['eval']['mae'])
            rmse_scores.append(metrics['eval']['rmse'])
            r2_scores.append(metrics['eval']['r2'])
            
            # 保存每个折的结果
            fold_dir = f'{experiment_dir}/fold_{fold + 1}'
            os.makedirs(fold_dir, exist_ok=True)
            model.save_model(fold_dir)
        
        # 打印并保存交叉验证结果
        print("\n交叉验证结果:")
        print(f"平均 MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
        print(f"平均 RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
        print(f"平均 R2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
        
        # 保存交叉验证结果
        save_training_results(
            {
                'mae': mae_scores,
                'rmse': rmse_scores,
                'r2': r2_scores
            },
            experiment_dir
        )
        
        # === 第二部分：训练最终模型 ===
        print("\n开始训练最终模型...")
        final_model = XGBoostPredictor()
        
        # 使用全部训练数据训练最终模型
        final_metrics = final_model.train(
            train_features=train_features,
            train_labels=train_labels
        )
        
        # 保存最终模型
        final_model_dir = f'{experiment_dir}/final_model'
        os.makedirs(final_model_dir, exist_ok=True)
        final_model.save_model(final_model_dir)
        
        # 在测试集上评估最终模型
        test_predictions = final_model.predict(test_features)
        test_labels = processed_data['等待时间'].values[train_size:]
        test_metrics = final_model._calculate_metrics(test_labels, test_predictions)
        
        print("\n最终模型在测试集上的表现:")
        print(f"MAE: {test_metrics['mae']:.4f}")
        print(f"RMSE: {test_metrics['rmse']:.4f}")
        print(f"R2: {test_metrics['r2']:.4f}")
        
        # 保存最终模型的测试结果
        with open(f'{final_model_dir}/test_metrics.json', 'w', encoding='utf-8') as f:
            json.dump(test_metrics, f, indent=4, ensure_ascii=False)
        
        print(f"\n实验结果已保存到: {experiment_dir}")
        print(f"最终模型已保存到: {final_model_dir}")
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()