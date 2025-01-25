import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os

class LaneProcessor:
    def __init__(self, file_path, filer=None, lastfit_degree=3, read_interval = 2, max_class_num = 200, plot=False):
        self.file_path = file_path
        self.df = None
        self.filter = filer
        self.lastfit_degree = lastfit_degree
        self.read_interval = read_interval
        self.max_class_num = max_class_num
        self.abs_x = np.array([])  # 绝对坐标系下的x坐标
        self.abs_y = np.array([])  # 绝对坐标系下的y坐标
        self.line = pd.DataFrame()
        self.plot = plot

    def read_csv(self):
        df = pd.read_csv(self.file_path)
        self.df = df.drop_duplicates(subset=['c0', 'c1', 'c2', 'c3'])  # 剔除重复行

    def cluster_and_fit(self):
        """
            construct self.abs_x, self.abs_y

            first, check if label is noisy data "-1", otherwise processing

            self.abs_x = np.array([])
            self.abs_y = np.array([])
        """
        if self.df is None:
            raise ValueError("Dataframe is empty. Please read the CSV file first.")

        # 按ts分组
        grouped = self.df.groupby('ts')

        self.abs_x = np.array([])  # 绝对坐标系下的x坐标
        self.abs_y = np.array([])

        cnt = 0
        for ts, group in grouped:
            # print(cnt % 50)
            # print(cnt)
            if cnt % self.read_interval != 0:
                cnt += 1
                print('skip {} data'.format(cnt))
                continue
            else:
                cnt += 1
            # 获取三次多项式的系数
            coefficients = group[['c0', 'c1', 'c2', 'c3']].values

            # 使用DBSCAN进行聚类
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            labels = dbscan.fit_predict(coefficients)
            # print("number of clusters: ", len(set(labels)))

            ego_x, ego_y, ego_yaw = group.iloc[0][['ego_x', 'ego_y', 'ego_yaw']] # 车辆坐标系的原点和方向
            # 对每个类别内的数据进行三次多项式拟合
            unique_labels = set(labels)
            for cluster in unique_labels:

                # Check if data can be seen as noisy point
                if cluster == -1:
                    # continue

                    indices = np.where(labels == cluster)[0]
                    for idx in indices:
                        start = group.iloc[idx]['LD_Start']
                        end = group.iloc[idx]['LD_End']
                        if (end - start) < self.filter:
                            # 忽略噪声点
                            # print("ignore noise point")
                            continue
                        else:
                            x = np.linspace(start, end, 50)
                            y = group.iloc[idx]['c0'] + group.iloc[idx]['c1'] * x + group.iloc[idx]['c2'] * x**2 + group.iloc[idx]['c3'] * x**3
                            # 拟合三次多项式
                            coeffs = np.polyfit(x, y, 3)
                            x_fit = np.linspace(min(x) + (max(x)-min(x))/1, max(x) -(max(x)-min(x))/1, 100)
                            # x_fit = np.linspace(-10, 10, 50)
                            y_fit = np.polyval(coeffs, x_fit)

                            # plt.plot(x_fit, y_fit, label=f'TS: {ts}, idx: {idx}')

                            # 将车辆坐标系下的坐标转换为绝对坐标系下的坐标
                            abs_x = ego_x + x_fit * np.cos(ego_yaw) - y_fit * np.sin(ego_yaw)
                            abs_y = ego_y + x_fit * np.sin(ego_yaw) + y_fit * np.cos(ego_yaw)
                            self.abs_x = np.concatenate([self.abs_x, abs_x])
                            self.abs_y = np.concatenate([self.abs_y, abs_y])
                    continue

                x_all = np.array([])
                y_all = np.array([])
                true_indices = np.where(labels == cluster)[0]
                for idx in true_indices:
                    x = np.linspace(group.iloc[idx]['LD_Start'], group.iloc[idx]['LD_End'], 100)
                    y = group.iloc[idx]['c0'] + group.iloc[idx]['c1'] * x + group.iloc[idx]['c2'] * x**2 + group.iloc[idx]['c3'] * x**3
                    x_all = np.concatenate([x_all, x])
                    y_all = np.concatenate([y_all, y])

                # 拟合三次多项式
                coeffs = np.polyfit(x_all, y_all, 3)
                x_fit = np.linspace(min(x_all) + (max(x_all)-min(x_all))/1, max(x_all) -(max(x_all)-min(x_all))/1, 100)  # 考虑到越远越不准，只取感知范围的一半
                y_fit = np.polyval(coeffs, x_fit)

                # 将车辆坐标系下的坐标转换为绝对坐标系下的坐标
                abs_x = ego_x + x_fit * np.cos(ego_yaw) - y_fit * np.sin(ego_yaw)
                abs_y = ego_y + x_fit * np.sin(ego_yaw) + y_fit * np.cos(ego_yaw)
                self.abs_x = np.concatenate([self.abs_x, abs_x])
                self.abs_y = np.concatenate([self.abs_y, abs_y])

    def coordinate_transform_and_concatenate(self):
        '''未进行任何处理，直接原始数据转为绝对坐标然后拼接'''
        for time_id, time_data in self.df.groupby('ts'):
            for track_id, group in time_data.groupby('track_id'):
                # 获取三次多项式的系数
                c0, c1, c2, c3 = group.iloc[0][['c0', 'c1', 'c2', 'c3']]

                # 获取起止点
                LD_Start = group.iloc[0]['LD_Start']
                LD_End = group.iloc[0]['LD_End']

                # 生成x坐标
                x = np.linspace(LD_Start, LD_End, 50)

                # 计算y坐标
                y = c0 + c1 * x + c2 * x**2 + c3 * x**3

                # 获取车辆坐标系的原点和方向
                ego_x, ego_y, ego_yaw = group.iloc[0][['ego_x', 'ego_y', 'ego_yaw']]

                # 将车辆坐标系下的坐标转换为绝对坐标系下的坐标
                abs_x = ego_x + x * np.cos(ego_yaw) - y * np.sin(ego_yaw)
                abs_y = ego_y + x * np.sin(ego_yaw) + y * np.cos(ego_yaw)

                self.abs_x = np.concatenate([self.abs_x, abs_x])
                self.abs_y = np.concatenate([self.abs_y, abs_y])

        # plt.scatter(x_all, y_all)
        # plt.xlabel('Abosulte X')
        # plt.ylabel('Abosulte Y')
        # plt.legend()
        # # plt.title('Before')
        # plt.show()
        # # plt.savefig(f'./figures/before/{time_id}.png')
        # plt.close()

    def dbscan_on_absolute_coordinates(self, eps=0.5, min_samples=10):
        """
            output: self.line
                include (x,y,line_id)

                line.append({'x': x, 'y': y, 'cluster': cluster})
        """
        if self.abs_x is None or self.abs_y is None:
            raise ValueError("Absolute coordinates are not available. Please run cluster_and_fit first.")

        # 组合绝对坐标
        coordinates = np.vstack((self.abs_x, self.abs_y)).T

        # 使用DBSCAN进行聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(coordinates)

        if self.plot:
            plt.figure(figsize=(10, 8))

        # 绘制聚类结果
        unique_labels = set(labels)
        for cluster in unique_labels:
            if cluster == -1:
                # 忽略噪声点
                continue
            cluster_points = coordinates[labels == cluster]
            if self.plot:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1])

        # plt.xlabel('Absolute X')
        # plt.ylabel('Absolute Y')
        # plt.legend()
        # plt.show()

        # 筛选包含200个点以上的类别，并进行多项式拟合
        unique_labels = set(labels)
        line = []
        for cluster in unique_labels:
            if cluster == -1:
                # 忽略噪声点
                continue
            cluster_points = coordinates[labels == cluster]
            if len(cluster_points) < self.max_class_num:
                continue

            # # 拟合三次多项式
            # x_all = cluster_points[:, 0]
            # y_all = cluster_points[:, 1]
            # coeffs = np.polyfit(x_all, y_all, 3)
            # x_fit = np.linspace(min(x_all), max(x_all), 50)
            # y_fit = np.polyval(coeffs, x_fit)

            # # 绘制拟合曲线
            # plt.plot(x_fit, y_fit, 'k', label=f'Cluster: {cluster}')

             # 选择x最大和最小的两个点作为x轴
            max_point = cluster_points[np.argmax(cluster_points[:, 0])]
            min_point = cluster_points[np.argmin(cluster_points[:, 0])]
            dx = max_point[0] - min_point[0]
            dy = max_point[1] - min_point[1]
            # print(max_point, min_point)
            angle = np.arctan2(dy, dx)

            # 旋转坐标系，使x轴对齐
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rotated_points = np.dot(cluster_points - min_point, rotation_matrix)

            # 拟合三次多项式
            x_all = rotated_points[:, 0]
            y_all = rotated_points[:, 1]
            coeffs = np.polyfit(x_all, y_all, self.lastfit_degree)
            x_fit = np.linspace(min(x_all), max(x_all), 50)
            y_fit = np.polyval(coeffs, x_fit)

            # 将坐标转化回来
            rotated_fit = np.vstack((x_fit, y_fit)).T
            inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
            original_fit = np.dot(rotated_fit, inverse_rotation_matrix) + min_point
            for x, y in original_fit:
                line.append({'x': x, 'y': y, 'cluster': cluster})

            # 绘制拟合曲线
            if self.plot:
                plt.plot(original_fit[:, 0], original_fit[:, 1], 'k', label=f'Cluster: {cluster}')
                mid_point = original_fit[len(original_fit) // 2]
                plt.text(mid_point[0], mid_point[1], cluster, fontsize=9, ha='center')

        if self.plot:
            plt.xlabel('Absolute X')
            plt.ylabel('Absolute Y')
            plt.axis('equal')
            plt.legend()

            folder_path = os.path.dirname(self.file_path)
            plt.savefig('results/extracted_line.png', dpi=600)
            plt.show()
            plt.close()

        self.line = pd.DataFrame(line)

if __name__ == '__main__':
    # 使用示例
    '''
    Ganzhide
    Duochulai
    Xianyousuanfa
    '''
    # case = "Newcase2"
    case = "Ganzhide"
    line_case_path = '../data/'+ case +'/line.csv'
    line_file_path = 'data/line.csv'
    # 确定超参数
    _filter = 1        # LD_end - LD_start := filter
    lastfit_degree=3        # 最后拟合车道边界的时候使用的多项式次数
    read_interval = 1       # 读取数据的间隔
    # max_class_num = 200     # 选择的类别中最少包含的点数
    max_class_num = 10
    processor = LaneProcessor(line_file_path, _filter, lastfit_degree, read_interval, max_class_num, plot=True)
    processor.read_csv()
    processor.cluster_and_fit()
    # processor.coordinate_transform_and_concatenate()
    processor.dbscan_on_absolute_coordinates()
    # processor.line.to_csv('../data/'+ case +'/line_processed.csv', index=False)
    processor.line.to_csv('./results/line_processed.csv', index=False)