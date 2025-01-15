import pandas as pd

# 读取数据
data = pd.read_csv('obj.csv', sep=',')

# # 转换时间戳为秒级
# data['ts'] = data['ts'] / 1e9

# 对数据进行排序
data.sort_values(by='ts', inplace=True)

# 初始化一个新的DataFrame来存储对齐后的数据
aligned_data = pd.DataFrame()

# 设置初始时间戳
start_time = data['ts'].min()

# 遍历数据，按照0.1秒的间隔进行对齐
while start_time < data['ts'].max():
    # 计算下一个时间戳
    next_ts = start_time + 0.1
    # 找到与当前时间戳最接近的数据
    current_data = data.loc[(data['ts'] - start_time).abs().min()]

    # 将当前时间点的数据添加到对齐后的数据中
    aligned_data = pd.concat([aligned_data, current_data.to_frame().T], ignore_index=True)

    # 更新时间戳，增加0.1秒
    start_time = current_data["ts"]

# 保存对齐后的数据
aligned_data.to_csv('aligned_data.csv', index=False)