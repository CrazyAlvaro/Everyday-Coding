import json
import csv
import matplotlib.pyplot as plt

# 读取 JSON 文件
with open("路口.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 提取 lane 数据
lanes_data = []
lane_coordinates = {}  # 用于可视化存储每条车道的点
for road in data.get("road_network", []):
    for lane in road.get("vertex_descriptor", {}).get("lane", []):
        # 将 lane_id 转换为字符串，避免 KeyError
        lane_id = str(lane["id"])  # 强制将 lane_id 转换为字符串
        if lane_id not in lane_coordinates:
            lane_coordinates[lane_id] = {"x": [], "y": []}
        for point in lane["centerline"]:
            x, y = map(float, point)
            lanes_data.append([lane_id, x, y])
            lane_coordinates[lane_id]["x"].append(x)
            lane_coordinates[lane_id]["y"].append(y)

# 写入 CSV 文件
output_file = "lane_centerline.csv"
with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "x", "y"])  # 表头
    writer.writerows(lanes_data)      # 数据

print(f"Lane 数据已保存到 {output_file}")

# 可视化车道中心线
plt.figure(figsize=(10, 6))
for lane_id, points in lane_coordinates.items():
    plt.plot(points["x"], points["y"], label=f"Lane {lane_id}", marker='o')

# 图形美化
plt.title("Lane Centerlines Visualization")
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.grid(True)
plt.legend()
plt.show()
