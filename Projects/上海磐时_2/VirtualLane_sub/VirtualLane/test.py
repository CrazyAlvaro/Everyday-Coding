from src.TrajectoryExtractor import TrajectoryExtractor
from src.trajectory_filtering import TrajectoryProcessor
from src.trajectory_clustering import TrajectoryClustering
from src.VirtualLane import VirtualLane


# #----数据提取----#
# extractor = TrajectoryExtractor("./data/路口.json")

# # 提取 road_user 和 ego 数据并保存为 CSV 文件
# road_user_csv_data, ego_csv_data = extractor.extract_data()
# extractor.write_csv("./result/object.csv", road_user_csv_data)
# extractor.write_csv("./result/ego.csv", ego_csv_data)

# # 提取 lane_boundary 并保存为 CSV 文件
# extractor.extract_lane_boundaries_and_save("./result/lane_boundaries.csv")

# # 提取 lane_centerline 并保存为 CSV 文件
# extractor.extract_centerline_and_save("./result/lane_centerline.csv")

# # 对齐 obj 数据并保存为 CSV 文件
# # extractor.align_obj_data("./data/obj.csv", "./result/aligned_data.csv")

# print("数据提取完成！")

#----轨迹过滤----#
config_file_path = './cfg/config.json'  # JSON 配置文件路径
processor = TrajectoryProcessor(config_file_path)
processor.process_data()
print("轨迹过滤完成！")

#----轨迹聚类----#
clustering = TrajectoryClustering(config_file_path)
clustering.run()
print("轨迹聚类完成！")

#----虚拟车道----#
# 创建 TrajectoryFitting 对象
virtual_lane = VirtualLane(config_file_path)
# 拟合车道中心线
virtual_lane.fit_lane_centrelines()
# 对轨迹进行聚类，并对同类别内的轨迹求标准线
virtual_lane.cluster_trajectories_and_refit()
# 计算车道边界
virtual_lane.get_lane_boundary()
# # 计算虚拟车道线与感知车道线的偏移量并进行平移
# virtual_lane.cal_offset_with_detected_line_and_translate()
# 绘制拟合的车道中心线
virtual_lane.plot_virtual_center_lines()
# 绘制车道边界
virtual_lane.plot_lane_boundaries()
print("虚拟车道生成完成！")
