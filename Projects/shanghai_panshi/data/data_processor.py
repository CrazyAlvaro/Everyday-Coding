import pandas as pd

# 读取多个CSV文件
df_ego = pd.read_csv("ego.csv", index_col='ts', parse_dates=True)
df_obj = pd.read_csv("obj.csv", index_col='ts', parse_dates=True)
# ... 其他传感器文件

# 合并DataFrame
df_merged = pd.concat([df_ego, df_obj], axis=1)

# 处理缺失值（例如填充为0）
df_merged.fillna(0, inplace=True)

# 查看合并后的DataFrame
print(df_merged.head)