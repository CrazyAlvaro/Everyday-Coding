import json
import os
import pandas as pd
import numpy as np


class TrajectoryProcessor:
    def __init__(self, config_path, casename):
        """
        初始化 TrajectoryProcessor 类，加载配置参数。
        :param config_path: JSON 配置文件路径
        """
        self.config = self.load_config(config_path)
        self.filepath = "./results/"+ casename + "/" + self.config['tracks_file']  # 输入文件路径
        self.T1 = self.config['params']['T1']  # 轨迹点数阈值 T1
        self.T2 = self.config['params']['T2']  # 横纵坐标方差阈值 T2，T3
        self.T3 = self.config['params']['T3']
        self.variability_threshold = self.config['params']['variability_threshold']  # 轨迹稳定性阈值
        self.lateral_threshold = self.config['params']['lateral_threshold']  # 侧偏阈值
        self.yaw_rate_threshold = self.config['params']['yaw_rate_threshold']  # 航向角变化速率阈值
        self.ay_threshold = self.config['params']['ay_threshold']  # 纵向加速度阈值
        self.outputfile = "./results/"+ casename + "/" + self.config['valid_trajectories_file']  # 输出文件路径
        self.df = None

    @staticmethod
    def load_config(file_path):
        """加载 JSON 配置文件并返回配置信息"""
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def read_data(file_path):
        """读取 CSV 文件并返回 DataFrame"""
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        return pd.read_csv(file_path)

    @staticmethod
    def save_results(df, output_file):
        """将筛选后的数据保存为 CSV 文件"""
        df.to_csv(output_file, index=False)
        print(f"结果已保存到: {output_file}")

    @staticmethod
    def filter_by_length(df, T1):
        """根据轨迹点数阈值 T1 筛选有效轨迹"""
        trajectory_counts = df.groupby('vehicle_id').size()
        valid_ids = trajectory_counts[trajectory_counts >= T1].index
        return df[df['vehicle_id'].isin(valid_ids)]

    @staticmethod
    def calculate_variability(group):
        """计算位置和速度的标准差"""
        position_variability = np.sqrt((group['x'].diff() ** 2 + group['y'].diff() ** 2)).fillna(0).std()
        speed = np.sqrt((group['x'].diff() ** 2 + group['y'].diff() ** 2)).fillna(0)
        speed_variability = speed.diff().std()
        return pd.Series({'position_variability': position_variability, 'speed_variability': speed_variability})

    def filter_by_stability(self, df, variability_threshold):
        """过滤不稳定的轨迹"""
        grouped = df.groupby('vehicle_id')
        variability = grouped.apply(self.calculate_variability)
        unstable_ids = variability[(variability['position_variability'] > variability_threshold) |
                                   (variability['speed_variability'] > variability_threshold)].index
        return df[~df['vehicle_id'].isin(unstable_ids)]

    @staticmethod
    def calculate_variances(group):
        """计算每条轨迹的横纵坐标方差"""
        return pd.Series({'Vx': group['x'].var(), 'Vy': group['y'].var()})

    def filter_by_variance(self, df, T2, T3):
        """根据横纵坐标方差筛选有效轨迹"""
        grouped = df.groupby('vehicle_id')
        variances = grouped.apply(self.calculate_variances)
        valid_ids = variances[(variances['Vx'] > T2) | (variances['Vy'] > T3)].index
        return df[df['vehicle_id'].isin(valid_ids)]

    @staticmethod
    def filter_by_lane_change(df, lateral_threshold, yaw_rate_threshold, ay_threshold):
        """根据横向偏移、航向角变化速率、横向加速度检测变道轨迹"""
        vehicles_to_remove = []
        for vehicle_id, group in df.groupby('vehicle_id'):
            delta_x = group['x'].diff()
            delta_y = group['y'].diff()
            yaw = group['yaw'].diff()
            lateral_displacement = (-np.sin(yaw) * delta_x + np.cos(yaw) * delta_y).abs().sum()
            yaw_rate_change = group['yaw_rate'].diff().abs().max()
            ay_change = group['ay'].diff().abs().sum()

            if (lateral_displacement > lateral_threshold and
                yaw_rate_change > yaw_rate_threshold and
                ay_change > ay_threshold):
                vehicles_to_remove.append(vehicle_id)

        return df[~df['vehicle_id'].isin(vehicles_to_remove)]

    def process_data(self):
        """主流程：执行数据读取、筛选、计算和保存"""
        # 读取数据
        self.df = self.read_data(self.filepath)

        # 执行筛选步骤
        self.df = self.filter_by_length(self.df, self.T1)
        self.df = self.filter_by_stability(self.df, self.variability_threshold)
        self.df = self.filter_by_variance(self.df, self.T2, self.T3)
        self.df = self.filter_by_lane_change(self.df,
                                             self.lateral_threshold,
                                             self.yaw_rate_threshold,
                                             self.ay_threshold)

        # 保存结果
        self.save_results(self.df, self.outputfile)
        print(f"有效轨迹总数: {len(self.df['vehicle_id'].unique())}")


# 使用示例
if __name__ == "__main__":
    config_file_path = '../cfg/config.json'  # JSON 配置文件路径
    processor = TrajectoryProcessor(config_file_path)
    processor.process_data()
