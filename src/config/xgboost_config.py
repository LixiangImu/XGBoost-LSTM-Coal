"""配置文件"""

# 特征工程配置
FEATURE_CONFIG = {
    # 主要输入特征（实时预测时需要的）
    'PRIMARY_FEATURES': ['提煤单号', '煤种编号', '排队时间'],
    
    # 时间窗口配置
    'WINDOW_SIZE': '1h',  # 1小时的时间窗口
    
    # 时间特征配置
    'TIME_FEATURES': ['hour', 'weekday', 'shift'],
    
    # 目标变量
    'TARGET': '等待时间'
}

# XGBoost模型参数
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.03,
    'num_boost_round': 1000,
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.2,
    'reg_alpha': 0.2,
    'reg_lambda': 1.2,
    'random_state': 42,
    'tree_method': 'gpu_hist',
    'gpu_id': 6,
    'predictor': 'gpu_predictor',
    'categorical_feature': ['shift']
}

# 数据处理配置
DATA_CONFIG = {
    'TEST_SIZE': 0.2,  # 测试集比例
    'RANDOM_STATE': 42  # 随机种子
}