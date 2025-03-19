import pandas as pd

# 读取数据文件
df = pd.read_csv('/home/lx/XGBOOST-LSTM-COAL/data/raw/queue_data_offset_sorted_utf8.csv')

# 打印列名
print("数据文件列名:")
print(df.columns.tolist())

# 打印前几行数据
print("\n数据预览:")
print(df.head())