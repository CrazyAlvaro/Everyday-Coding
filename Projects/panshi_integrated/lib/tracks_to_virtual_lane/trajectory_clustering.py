import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import json


class TrajectoryClustering:
    def __init__(self, config_path,casename):
        """
        初始化 TrajectoryClustering 类，从配置文件加载参数。

        :param config_path: JSON 配置文件路径
        """
        self.config = self.load_config(config_path)
        # self.case_name = self.config['case_name']  # 案例名称
        self.file_path = "./results/"+ casename + "/" + self.config['valid_trajectories_file']  # 输入文件路径
        self.eps = self.config['params']['eps']  # 邻域半径
        self.min_samples = self.config['params']['min_samples']  # 核心点的最少样本数
        self.output_file = "./results/"+ casename + "/" + self.config['vehicle_with_cluster_id']  # 输出文件路径
        self.data = None
        self.segments = []
        self.vehicle_ids = []
        self.cluster_labels = None

    @staticmethod
    def load_config(config_path):
        """加载 JSON 配置文件并返回配置信息"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def load_data(self):
        """加载轨迹数据。"""
        self.data = pd.read_csv(self.file_path)
        print(f"成功加载数据：{len(self.data)} 条记录")

    def partition_trajectory(self, trajectory):
        """将轨迹分割为线段。"""
        return [(trajectory.iloc[i], trajectory.iloc[i + 1]) for i in range(len(trajectory) - 1)]

    def segment_to_feature_vector(self, segment):
        """将线段转换为特征向量。"""
        return [segment[0]['x'], segment[0]['y'], segment[1]['x'], segment[1]['y']]

    def process_trajectories(self):
        """处理所有车辆的轨迹，将其分割为线段。"""
        self.segments = []
        self.vehicle_ids = []
        for vehicle_id, vehicle_trajectory in self.data.groupby('vehicle_id'):
            vehicle_segments = self.partition_trajectory(vehicle_trajectory[['x', 'y']])
            self.segments.extend(vehicle_segments)
            self.vehicle_ids.extend([vehicle_id] * len(vehicle_segments))
        print(f"成功分割出 {len(self.segments)} 个线段")

    def cluster_segments(self):
        """对线段进行 DBSCAN 聚类。"""
        feature_vectors = [self.segment_to_feature_vector(segment) for segment in self.segments]
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(feature_vectors)
        self.cluster_labels = db.labels_
        print(f"DBSCAN 聚类完成，共识别出 {len(np.unique(self.cluster_labels))} 个簇（包含噪声）")

    def assign_cluster_labels(self):
        """将聚类标签映射回原始数据。"""
        vehicle_to_cluster = {self.vehicle_ids[i]: self.cluster_labels[i] for i in range(len(self.vehicle_ids))}
        self.data['cluster_id'] = self.data['vehicle_id'].map(vehicle_to_cluster)
        print("\n聚类结果（每辆车的 ID 和对应的聚类 ID）:")
        for vehicle_id, cluster_id in vehicle_to_cluster.items():
            print(f"ID: {vehicle_id}, Cluster ID: {cluster_id}")

    def plot_trajectories(self):
        """可视化聚类结果。"""
        plt.figure(figsize=(10, 6))
        unique_labels = np.unique(self.cluster_labels)
        for label in unique_labels:
            if label == -1:
                color = 'k'  # 噪声点
            else:
                color = plt.cm.jet(label / float(max(unique_labels)))
            class_member_mask = (self.data['cluster_id'] == label)
            xy = self.data[class_member_mask]
            plt.plot(xy['x'], xy['y'], 'o', markerfacecolor=color, markersize=5, label=f'Cluster {label}')
        plt.title('Clustered Trajectories')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.show()

    def save_results(self):
        """保存结果到 CSV 文件。"""
        self.data.to_csv(self.output_file, index=False)
        print(f"\n聚类结果已保存到 '{self.output_file}'")

    def run(self):
        """运行完整的聚类流程。"""
        self.load_data()
        self.process_trajectories()
        self.cluster_segments()
        self.assign_cluster_labels()
        self.plot_trajectories()
        self.save_results()


# 使用示例
if __name__ == "__main__":
    config_path = '../cfg/config.json'  # 配置文件路径
    clustering = TrajectoryClustering(config_path)
    clustering.run()
