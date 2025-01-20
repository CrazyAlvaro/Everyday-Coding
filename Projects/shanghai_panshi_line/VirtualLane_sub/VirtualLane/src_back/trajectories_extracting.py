import csv
import json
import math
from datetime import datetime

# 读取 JSON 文件
with open("路口.json", "r", encoding="utf-8") as file:
    data = json.load(file)
# 提取数据
road_users = data.get("road_user")
ego_vehicle = data.get("ego")
start_unix_timestamp = int(datetime.strptime(data["environment"]["time"], "%Y-%m-%d %H:%M:%S").timestamp())


# 计算纵向和横向的速度和加速度
def decompose_velocity_acceleration(v, a, yaw):
    """分解速度和加速度为纵向和横向分量"""
    yaw = float(yaw)  # 转换 yaw 为浮动数值
    # 计算纵向速度 vx 和横向速度 vy
    vx = v * math.cos(math.radians(yaw))
    vy = v * math.sin(math.radians(yaw))

    # 计算纵向加速度 ax 和横向加速度 ay
    ax = a * math.cos(math.radians(yaw))
    ay = a * math.sin(math.radians(yaw))

    return vx, vy, ax, ay


# 初始化 CSV 文件数据
def generate_trajectory_data(vehicles, timestamp):
    csv_data = [["timestamp", "vehicle_id", "length", "width", "height", "x", "y", "yaw", "yaw_rate", "vx", "vy", "ax", "ay"]]

    for vehicle in vehicles:
        user_id = vehicle["id"]
        length, width, height = map(float, vehicle["dimension"])
        initial_position, initial_orientation = map(list, [vehicle["initial_position"], vehicle["initial_orientation"]])
        vertices = vehicle["trajectory"]["vertices"]

        # 初始位置与速度
        x, y, yaw, yaw_rate = initial_position[0], initial_position[1], initial_orientation[0], initial_orientation[1]
        v, a = (float(vertices[0][-3]), float(vertices[0][-2])) if vertices else (0, 0)

        # 计算纵向和横向速度及加速度
        vx, vy, ax, ay = decompose_velocity_acceleration(v, a, yaw)

        # 第一帧数据
        csv_data.append([timestamp, user_id, length, width, height, x, y, yaw, yaw_rate, vx, vy, ax, ay])

        # 轨迹点数据
        for vertex in vertices:
            x, y, yaw, yaw_rate = map(float, vertex[:4])
            # v, a = float(vertex[-3]), float(vertex[-2])
            vx, vy, ax, ay = decompose_velocity_acceleration(v, a, yaw)
            delta_t = float(vertex[-1]) * 100  # 时间间隔
            current_time = timestamp + int(delta_t)
            csv_data.append([current_time, user_id, length, width, height, x, y, yaw, yaw_rate, vx, vy, ax, ay])

    return csv_data


# 生成 road_user 和 ego 数据
road_user_csv_data = generate_trajectory_data(road_users, start_unix_timestamp)
ego_csv_data = generate_trajectory_data([ego_vehicle], start_unix_timestamp) if ego_vehicle else []

# 排序 road_user 数据
road_user_csv_data_sorted = [road_user_csv_data[0]] + sorted(road_user_csv_data[1:], key=lambda row: row[0])


# 写入 CSV 文件
def write_csv(file_name, data):
    with open(file_name, "w", encoding="utf-8", newline="") as csvfile:
        csv.writer(csvfile).writerows(data)


write_csv("object.csv", road_user_csv_data)
# write_csv("object_sorted_trajectory.csv", road_user_csv_data_sorted)
write_csv("ego.csv", ego_csv_data)

