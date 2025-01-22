import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 读取 CSV 文件
def read_data(file_path):
    """读取 CSV 文件并返回 DataFrame"""
    return pd.read_csv(file_path)


# 读取主车和其他车辆轨迹数据
main_vehicle_file = "ego.csv"  # 主车轨迹文件
other_vehicles_file = "valid_trajectories.csv"  # 其他车辆轨迹文件

# 读取数据
main_vehicle_data = read_data(main_vehicle_file)
other_vehicles_data = read_data(other_vehicles_file)

# 将时间戳转换为 datetime 类型
main_vehicle_data['timestamp'] = pd.to_datetime(main_vehicle_data['timestamp'], unit='s')
other_vehicles_data['timestamp'] = pd.to_datetime(other_vehicles_data['timestamp'], unit='s')

# 确保数据按车辆 ID 和时间戳排序
main_vehicle_data = main_vehicle_data.sort_values(by=['vehicle_id', 'timestamp'])
other_vehicles_data = other_vehicles_data.sort_values(by=['vehicle_id', 'timestamp'])

# 合并主车和其他车辆的数据
data = pd.concat([main_vehicle_data, other_vehicles_data], ignore_index=True)

# 分组按车辆 ID
vehicles = data.groupby("vehicle_id")

# 初始化动画画布
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Vehicle Trajectories (Animated)")
ax.set_xlabel("X Position (meters)")
ax.set_ylabel("Y Position (meters)")

# 设置动态范围
x_min, x_max = data["x"].min(), data["x"].max()
y_min, y_max = data["y"].min(), data["y"].max()
ax.set_xlim(x_min - 1, x_max + 1)
ax.set_ylim(y_min - 1, y_max + 1)
ax.grid()

# 每辆车的数据存储
lines = {}  # 每辆车的轨迹线
points = {}  # 每辆车的当前点
texts = {}  # 每辆车的 ID 文本
vehicle_trajectories = {}

# 初始化每辆车的轨迹
for vehicle_id, vehicle_data in vehicles:
    # 确保轨迹数据是列表
    x_coords = vehicle_data["x"].dropna().tolist()  # 去掉空值并转换为列表
    y_coords = vehicle_data["y"].dropna().tolist()  # 去掉空值并转换为列表

    # 跳过没有有效数据的车辆
    if len(x_coords) < 2 or len(y_coords) < 2:
        continue

    # 保存每辆车的轨迹数据
    vehicle_trajectories[vehicle_id] = (x_coords, y_coords)

    # 如果是主车，设置特殊样式
    # print(str(main_vehicle_data['id'].iloc[0]))
    if str(vehicle_id) == str(main_vehicle_data['vehicle_id'].iloc[0]):  # 假设主车ID为文件中的唯一ID
        # 主车使用红色，线条更粗，点为大圆点
        line, = ax.plot([], [], label=f"Main Vehicle ID: {vehicle_id}", color='red', linewidth=2)
        point, = ax.plot([], [], 'o', markersize=10, color='red')
    else:
        # 其他车辆使用不同颜色，默认设置
        line, = ax.plot([], [], label=f"Vehicle ID: {vehicle_id}")
        point, = ax.plot([], [], 'o')

    # 设置标签，显示车辆ID
    text = ax.text(x_coords[0], y_coords[0], f"ID: {vehicle_id}", color="blue", fontsize=10)

    lines[vehicle_id] = line
    points[vehicle_id] = point
    texts[vehicle_id] = text  # 保存每辆车的文本对象

# 添加图例
ax.legend()


# 动画更新函数
def update(frame):
    for vehicle_id, (x_coords, y_coords) in vehicle_trajectories.items():
        if frame < len(x_coords):  # 确保 frame 不超出数据长度
            # 更新轨迹线
            lines[vehicle_id].set_data(x_coords[:frame + 1], y_coords[:frame + 1])
            # 更新当前点
            points[vehicle_id].set_data([x_coords[frame]], [y_coords[frame]])  # 确保点是可迭代的
            # 更新文本位置
            texts[vehicle_id].set_position((x_coords[frame], y_coords[frame]))
    return list(lines.values()) + list(points.values()) + list(texts.values())


# 动画帧数
max_frames = max(len(coords[0]) for coords in vehicle_trajectories.values())

# 创建动画
ani = FuncAnimation(fig, update, frames=max_frames, interval=500, blit=True)

# 显示动画
plt.show()
