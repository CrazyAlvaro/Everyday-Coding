import csv
import json
import math
import pandas as pd
from datetime import datetime

class TrajectoryExtractor:
    def __init__(self, json_file):
        self.data = self.load_json(json_file)
        self.road_users = self.data.get("road_user")
        self.ego_vehicle = self.data.get("ego")
        self.start_unix_timestamp = int(datetime.strptime(self.data["environment"]["time"], "%Y-%m-%d %H:%M:%S").timestamp())

    def load_json(self, json_file):
        with open(json_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def decompose_velocity_acceleration(self, v, a, yaw):
        """分解速度和加速度为纵向和横向分量"""
        yaw = float(yaw)  # 转换 yaw 为浮动数值
        # 计算纵向速度 vx 和横向速度 vy
        vx = v * math.cos(math.radians(yaw))
        vy = v * math.sin(math.radians(yaw))

        # 计算纵向加速度 ax 和横向加速度 ay
        ax = a * math.cos(math.radians(yaw))
        ay = a * math.sin(math.radians(yaw))

        return vx, vy, ax, ay

    def generate_trajectory_data(self, vehicles, timestamp):
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
            vx, vy, ax, ay = self.decompose_velocity_acceleration(v, a, yaw)

            # 第一帧数据
            csv_data.append([timestamp, user_id, length, width, height, x, y, yaw, yaw_rate, vx, vy, ax, ay])

            # 轨迹点数据
            for vertex in vertices:
                x, y, yaw, yaw_rate = map(float, vertex[:4])
                vx, vy, ax, ay = self.decompose_velocity_acceleration(v, a, yaw)
                delta_t = float(vertex[-1]) * 100  # 时间间隔
                current_time = timestamp + int(delta_t)
                csv_data.append([current_time, user_id, length, width, height, x, y, yaw, yaw_rate, vx, vy, ax, ay])

        return csv_data

    def extract_data(self):
        road_user_csv_data = self.generate_trajectory_data(self.road_users, self.start_unix_timestamp)
        ego_csv_data = self.generate_trajectory_data([self.ego_vehicle], self.start_unix_timestamp) if self.ego_vehicle else []

        # 排序 road_user 数据
        road_user_csv_data_sorted = [road_user_csv_data[0]] + sorted(road_user_csv_data[1:], key=lambda row: row[0])
        return road_user_csv_data, ego_csv_data
    
    # 写入 CSV 文件
    def write_csv(self, file_name, data):
        with open(file_name, "w", encoding="utf-8", newline="") as csvfile:
            csv.writer(csvfile).writerows(data)

    def extract_lane_boundaries_and_save(self, lane_boundaries_file):
        """提取 road_network 下 vertex_descriptor 中的 lane_boundary"""
        lane_boundaries = []
        for boundary in self.data['road_network'][0]['vertex_descriptor']['lane_boundary']:
            for point in boundary['boundary_point']:
                lane_boundaries.append({
                    'boundary_id': boundary['id'],
                    'x': point[0],
                    'y': point[1]
                })
        df = pd.DataFrame(lane_boundaries)
        df.to_csv(lane_boundaries_file, index=False, encoding='utf-8')

    def extract_centerline_and_save(self, centerline_file):
        """提取 lane 数据"""
        lanes_data = []
        lane_coordinates = {}
        for road in self.data.get("road_network", []):
            for lane in road.get("vertex_descriptor", {}).get("lane", []):
                lane_id = str(lane["id"])
                if lane_id not in lane_coordinates:
                    lane_coordinates[lane_id] = {"x": [], "y": []}
                for point in lane["centerline"]:
                    x, y = map(float, point)
                    lanes_data.append([lane_id, x, y])
                    lane_coordinates[lane_id]["x"].append(x)
                    lane_coordinates[lane_id]["y"].append(y)
        with open(centerline_file, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["id", "x", "y"])  # 表头
            writer.writerows(lanes_data)      # 数据
        print(f"Lane 数据已保存到 {centerline_file}")

    def align_obj_data(self, obj_data_input, obj_data_output):
        """将 obj 数据按照时间戳对齐"""
        # 读取数据
        data = pd.read_csv(obj_data_input, sep=',')

        # 转换时间戳为秒级
        data['ts'] = data['ts'] / 1e9

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
        aligned_data.to_csv(obj_data_output, index=False)
    

if __name__ == "__main__":
    extractor = TrajectoryExtractor("../data/路口.json")

    # 提取 road_user 和 ego 数据并保存为 CSV 文件
    road_user_csv_data, ego_csv_data = extractor.extract_data()
    extractor.write_csv("../result/object.csv", road_user_csv_data)
    extractor.write_csv("../result/ego.csv", ego_csv_data)

    # 提取 lane_boundary 并保存为 CSV 文件
    extractor.extract_lane_boundaries_and_save("../result/lane_boundaries.csv")

    # 提取 lane_centerline 并保存为 CSV 文件
    extractor.extract_centerline_and_save("../result/lane_centerline.csv")

    # 对齐 obj 数据并保存为 CSV 文件
    extractor.align_obj_data("../data/obj.csv", "../result/aligned_data.csv")
