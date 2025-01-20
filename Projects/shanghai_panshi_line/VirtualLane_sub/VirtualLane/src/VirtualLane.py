import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class VirtualLane:
    def __init__(self, config_path, degree=10, smooth_degree=7, max_curvature=0.1):
        self.config = self.load_config(config_path)
        self.file_path = self.config['vehicle_with_cluster_id']  # 输入文件路径
        self.dected_lane_path = self.config['lane_boundaries']  # 检测到的车道边线路径
        self.degree = degree    # 多项式拟合的阶数
        self.smooth_degree = smooth_degree  # 再次拟合的阶数
        self.max_curvature = max_curvature  # 曲率阈值
        self.trajectories = self.read_data(self.file_path) # 车辆轨迹数据
        self.lane_num = self.count_cluster_ids() #车道数量
        self.lane_width = self.cal_lane_width() #车道宽度
        self.lane_centrelines = {}  #拟合的车辆中心线_所有信息
        self.lane_center_xy = pd.DataFrame() #拟合的车辆中心线_坐标信息
        self.lane_std_line = pd.DataFrame() #标准线坐标点
        self.lane_boundary = pd.DataFrame() #车道边界坐标点

    @staticmethod
    def load_config(file_path):
        """加载 JSON 配置文件并返回配置信息"""
        with open(file_path, 'r') as f:
            return json.load(f)

    def read_data(self, file_path=None):
        """读取 CSV 文件并返回 DataFrame"""
        return pd.read_csv(file_path)
    
    def fit_polynomial(self, x, y, degree=3):
        """使用多项式拟合数据点"""
        coeffs = np.polyfit(x, y, degree)
        return np.poly1d(coeffs)

    def calculate_distances(self, poly1, poly2, x_values):
        """计算两个多项式曲线在给定 x 值上的距离"""
        y1 = poly1(x_values)
        y2 = poly2(x_values)
        distances = np.sqrt((x_values - x_values)**2 + (y1 - y2)**2)
        return distances
    
    def cal_lane_width(self):
        """计算车道宽度"""
        data = self.read_data(self.dected_lane_path)
    
        # 获取所有 boundary_id
        boundary_ids = data['boundary_id'].unique()
        
        # 存储每个 boundary_id 的多项式拟合结果
        polynomials = {}
        
        for boundary_id, boundary_data in data.groupby('boundary_id'):
            x = boundary_data['x'].values
            y = boundary_data['y'].values
            polynomials[boundary_id] = self.fit_polynomial(x, y)
        
        # 计算相邻 boundary_id 车道线上各点的距离，并求平均距离
        average_distances = []
        for i in range(len(boundary_ids) - 1):
            boundary_id1 = boundary_ids[i]
            boundary_id2 = boundary_ids[i + 1]
            poly1 = polynomials[boundary_id1]
            poly2 = polynomials[boundary_id2]
            
            # 取两个 boundary_id 车道线的 x 值的交集
            x_values = np.linspace(min(data['x'].min(), data['x'].max()), max(data['x'].min(), data['x'].max()), 100)
            
            distances = self.calculate_distances(poly1, poly2, x_values)
            average_distance = np.mean(distances)
            average_distances.append(average_distance)
        return np.mean(average_distances)
    
    def cal_offset_with_detected_line(self):
        pass

    def count_cluster_ids(self):
        """计算 self.trajectories 中 cluster_id 标签的数量"""
        return self.trajectories['cluster_id'].nunique()
    
    def calculate_curvature(self, coeffs, x):
        """计算多项式曲线在给定 x 处的曲率"""
        p = np.poly1d(coeffs)
        dp = np.polyder(p, 1)
        ddp = np.polyder(p, 2)
        dx = dp(x)
        ddx = ddp(x)
        curvature = np.abs(ddx) / (1 + dx**2)**1.5
        return curvature

    def transform_coordinates(self, x, y, x1, y1, x2, y2):
        """将坐标转换到以两点连线为 x 轴的坐标系"""
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        cos_theta = dx / length
        sin_theta = dy / length
        x_new = (x - x1) * cos_theta + (y - y1) * sin_theta
        y_new = -(x - x1) * sin_theta + (y - y1) * cos_theta
        return x_new, y_new

    def inverse_transform_coordinates(self, x_new, y_new, x1, y1, x2, y2):
        """将坐标从以两点连线为 x 轴的坐标系转换回原坐标系"""
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        cos_theta = dx / length
        sin_theta = dy / length
        x = x_new * cos_theta - y_new * sin_theta + x1
        y = x_new * sin_theta + y_new * cos_theta + y1
        return x, y

    def fit_lane_centrelines(self):
        """根据车辆轨迹点拟合车道中心线，并对拟合结果进行再次拟合"""
        for lane_id, lane_data in self.trajectories.groupby('cluster_id'):
            x = lane_data['x'].values
            y = lane_data['y'].values
            # 选取 x 最小和 y 最大的两个点
            x1, y1 = x[np.argmin(x)], y[np.argmin(x)]
            x2, y2 = x[np.argmax(y)], y[np.argmax(y)]
            # 坐标转换
            x_new, y_new = self.transform_coordinates(x, y, x1, y1, x2, y2)
            # 第一次多项式拟合
            initial_coeffs = np.polyfit(x_new, y_new, self.degree)
            y_fit_initial = np.polyval(initial_coeffs, x_new)
            # 计算曲率
            curvature = self.calculate_curvature(initial_coeffs, x_new)
            self.lane_centrelines[lane_id] = initial_coeffs
            smodeg = self.smooth_degree
            while np.max(curvature) > self.max_curvature:
                # 对拟合结果进行再次多项式拟合
                smooth_coeffs = np.polyfit(x_new, y_new, smodeg)
                self.lane_centrelines[lane_id] = smooth_coeffs
                curvature = self.calculate_curvature(smooth_coeffs, x_new)
                smodeg -= 1
                if smodeg <= 1:
                    break
            
            # 判断是直线还是曲线
            p = np.poly1d(self.lane_centrelines[lane_id])
            dp = np.polyder(p, 1)
            dx = dp(x_new)
            max_diff= np.max(dx) - np.min(dx)
            if max_diff < 0.5:  # 直线
                smooth_coeffs = np.polyfit(x_new, y_new, 1)
                self.lane_centrelines[lane_id] = smooth_coeffs
                label = 'straight'
            else: # 曲线
                label = 'curve'

            # 保存转换信息
            self.lane_centrelines[lane_id] = (self.lane_centrelines[lane_id], x1, y1, x2, y2, label)

    
        # print(self.lane_centrelines)

        # 解析拟合中心线坐标点并保存
        rows = []
        for lane_id in range(self.lane_num):
            
            coeffs, x1, y1, x2, y2, label = self.lane_centrelines[lane_id]
            x_fit_new = np.linspace(0, np.sqrt((x2 - x1)**2 + (y2 - y1)**2), 100)

            y_fit_new = np.polyval(coeffs, x_fit_new)
            x_fit, y_fit = self.inverse_transform_coordinates(x_fit_new, y_fit_new, x1, y1, x2, y2)
            # plt.plot(x_fit, y_fit, 'k', label=f'Lane {lane_id} Fit')
            for x, y in zip(x_fit, y_fit):
                rows.append({'cluster_id': lane_id, 'x': x, 'y': y, 'label': label})
        self.lane_center_xy = pd.DataFrame(rows)
        # print(self.lane_center_xy.head())
    
    def calculate_center_line(self, lines):  # todo: 传入多条线
        """计算多条线的中心线"""
        x_all = np.array([line['x'].values for line in lines])
        y_all = np.array([line['y'].values for line in lines])
        x_center = np.mean(x_all, axis=0)
        y_center = np.mean(y_all, axis=0)
        return x_center, y_center
    
    def cluster_trajectories_and_refit(self, n_clusters=2):
        """对轨迹进行聚类，对同类别内的轨迹求中心线，得到标准线"""
        # 提取每条轨迹的特征
        Lane_info = []
        for lane_id, lane_data in self.lane_center_xy.groupby('cluster_id'):
            Lane_info.append(lane_data[['x', 'y']].values.flatten())
        
        # 聚类
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(Lane_info)
        
        # 将聚类标签添加到数据中
        self.lane_center_xy['cluster'] = self.lane_center_xy['cluster_id'].map(lambda cid: labels[np.where(self.lane_center_xy['cluster_id'].unique() == cid)[0][0]])
        
        
        center_lines_rows = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) >= 2:
                # 取该类别中线段的中心线
                lines = [self.lane_center_xy[self.lane_center_xy['cluster_id'] == self.lane_center_xy['cluster_id'].unique()[idx]] for idx in cluster_indices]
                x_center, y_center = self.calculate_center_line(lines)
                
                for x, y in zip(x_center, y_center):
                    center_lines_rows.append({'recluster_id': cluster_id, 'x': x, 'y': y, 'label': lines[0]['label'][0], 'offset': self.lane_width, 'lane_num': len(cluster_indices)})
            else:
                for idx in cluster_indices:
                    lane_data = self.lane_center_xy[self.lane_center_xy['cluster_id'] == self.lane_center_xy['cluster_id'].unique()[idx]]
                    for x, y in zip(lane_data['x'], lane_data['y']):
                        center_lines_rows.append({'recluster_id': cluster_id, 'x': x, 'y': y, 'label': lane_data['label'].values[0], 'offset': self.lane_width/2, 'lane_num': 1})
        self.lane_std_line = pd.DataFrame(center_lines_rows)
        # self.lane_std_line.to_csv('lane_std_line.csv', index=False)

    def calculate_offset_lines(self, x_center, y_center, offset=3.5):
        """计算沿标准线垂线方向向两侧偏移的两条线"""
        dx = np.gradient(x_center)
        dy = np.gradient(y_center)
        length = np.sqrt(dx**2 + dy**2)
        nx = -dy / length
        ny = dx / length
        x_offset1 = x_center + offset * nx
        y_offset1 = y_center + offset * ny
        x_offset2 = x_center - offset * nx
        y_offset2 = y_center - offset * ny
        return (x_offset1, y_offset1), (x_offset2, y_offset2)

    def get_lane_boundary(self):
        """计算车道边界坐标点"""
        boundary_rows = []
        idx = 0
        for cluster_id, cluster_data in self.lane_std_line.groupby('recluster_id'):
            x_center = cluster_data['x'].values
            y_center = cluster_data['y'].values

            lane_num = cluster_data['lane_num'].values[0]

            # 判断车道线的奇偶性
            if lane_num % 2 == 0:
                for i in range(1,int(lane_num/2)+1):
                    offset_line1, offset_line2 = self.calculate_offset_lines(x_center, y_center,i * self.lane_width)
                    for x, y in zip(offset_line1[0], offset_line1[1]):
                        boundary_rows.append({'boundary_id': idx, 'x': x, 'y': y, 'stdline_id': cluster_id})
                    idx += 1
                    for x, y in zip(offset_line2[0], offset_line2[1]):
                        boundary_rows.append({'boundary_id': idx, 'x': x, 'y': y, 'stdline_id': cluster_id})
                    idx += 1
                for x, y in zip(x_center, y_center):
                    boundary_rows.append({'boundary_id': idx, 'x': x, 'y': y, 'stdline_id': cluster_id})
                idx += 1
            else:
                for i in range(int((lane_num-1)/2)+1):
                    offset_line1, offset_line2 = self.calculate_offset_lines(x_center, y_center,(i+0.5) * self.lane_width)
                    for x, y in zip(offset_line1[0], offset_line1[1]):
                        boundary_rows.append({'boundary_id': idx, 'x': x, 'y': y, 'stdline_id': cluster_id})
                    idx += 1
                    for x, y in zip(offset_line2[0], offset_line2[1]):
                        boundary_rows.append({'boundary_id': idx, 'x': x, 'y': y, 'stdline_id': cluster_id})
                    idx += 1
        self.lane_boundary = pd.DataFrame(boundary_rows)
        self.lane_boundary.to_csv('lane_boundaries111.csv', index=False)

    def find_max_y_line(self, data, x_min, x_max):
        """找到所有线中 y 值平均值最大的线"""
        max_avg_y = -np.inf
        max_y_line = None
        max_x_values = None
        
        for lane_id, lane_data in data.groupby('boundary_id'):
            x = lane_data[(lane_data['x'] >= x_min) & (lane_data['x'] <= x_max)]['x'].values
            y = lane_data[(lane_data['x'] >= x_min) & (lane_data['x'] <= x_max)]['y'].values
            
            avg_y = np.mean(y)
            if avg_y > max_avg_y:
                max_avg_y = avg_y
                max_y_line = y
                max_x_values = x
        
        return max_x_values, max_y_line
    
    def calculate_offset(self, c1_x, c1_y, c2_x, c2_y, degree=3):
        """计算两条线的偏移量"""
        # 拟合两条中心线
        coeffs1 = np.polyfit(c1_x, c1_y, degree)
        poly1 = np.poly1d(coeffs1)
        coeff2 = np.polyfit(c2_x, c2_y, degree)
        poly2 = np.poly1d(coeff2)
        
        # 取两条中心线的 x 值的交集
        x_min = max(c1_x.min(), c2_x.min())
        x_max = min(c1_x.max(), c2_x.max())
        x_values = np.linspace(x_min, x_max, 100)
        
        # 计算对应的 y 值
        y1 = poly1(x_values)
        y2 = poly2(x_values)
        
        # 计算偏移量
        offset = np.mean(y2 - y1)
        return offset

    def cal_offset_with_detected_line_and_translate(self):    
        '''计算虚拟车道线与感知车道线的偏移量并进行平移'''       
        # 读取车道线数据
        data1 = pd.read_csv(self.dected_lane_path)
        data2 = self.lane_boundary

        x_min = max(data1['x'].min(), data2['x'].min())
        x_max = min(data1['x'].max(), data2['x'].max())

        
        # 计算两个文件中的车道线的中心线
        c1_x, c1_y = self.find_max_y_line(data1, x_min, x_max)
        c2_x, c2_y = self.find_max_y_line(data2, x_min, x_max)

        # 计算偏移量
        offset = self.calculate_offset(c1_x, c1_y, c2_x, c2_y)
        print(offset)
    
        # 平移 lane_boundaries111.csv 中的点
        self.lane_boundary['y'] -= offset

    def plot_virtual_center_lines(self):
        """绘制拟合的车道中心线"""
        plt.figure(figsize=(10, 6))
        rows = []
        for lane_id, lane_data in self.trajectories.groupby('cluster_id'):
            x = lane_data['x'].values
            y = lane_data['y'].values
            plt.scatter(x, y, label=f'Lane {lane_id} Scatter')
            # 绘制拟合曲线
            coeffs, x1, y1, x2, y2, label = self.lane_centrelines[lane_id]
            x_fit_new = np.linspace(0, np.sqrt((x2 - x1)**2 + (y2 - y1)**2), 100)
            y_fit_new = np.polyval(coeffs, x_fit_new)
            x_fit, y_fit = self.inverse_transform_coordinates(x_fit_new, y_fit_new, x1, y1, x2, y2)
            plt.plot(x_fit, y_fit, 'k', label=f'Lane {lane_id} Fit')
            for x, y in zip(x_fit, y_fit):
                rows.append({'cluster_id': lane_id, 'x': x, 'y': y})
        # df = pd.DataFrame(rows)
        # df.to_csv("lane_center.csv", index=False)

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.axis('equal')
        plt.title('Virtual Lane Centerlines')
        plt.show()

    def plot_lane_boundaries(self):
        """绘制计算得到的车道边界"""
        plt.figure(figsize=(10, 6))

        for lane_id, lane_data in self.trajectories.groupby('cluster_id'):
            x = lane_data['x'].values
            y = lane_data['y'].values
            plt.scatter(x, y, label=f'Lane {lane_id} Scatter')

        # for cluster_id, cluster_data in self.lane_std_line.groupby('recluster_id'):
        #     plt.plot(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster_id} Standard Line', linestyle='--')
        # 绘制车道边界
        for boundary_id, boundary_data in self.lane_boundary.groupby('boundary_id'):
            plt.plot(boundary_data['x'], boundary_data['y'], label='id:{}'.format(boundary_id))

        detec_lane = pd.read_csv(self.dected_lane_path)  # 检测到的车道边线
        for lane_id, lane_data in detec_lane.groupby('boundary_id'):
            x = lane_data['x'].values
            y = lane_data['y'].values
            plt.plot(x, y, 'k')
        
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.axis('equal')
        plt.title('Virtual Lane Boundaries')
        plt.show()

    def run(self):
        # 拟合车道中心线
        self.fit_lane_centrelines()
        # 对轨迹进行聚类，并对同类别内的轨迹求标准线
        self.cluster_trajectories_and_refit()
        # 计算车道边界
        self.get_lane_boundary()
        # 计算虚拟车道线与感知车道线的偏移量并进行平移
        self.cal_offset_with_detected_line_and_translate()
        # 绘制拟合的车道中心线
        self.plot_virtual_center_lines()
        # 绘制车道边界
        self.plot_lane_boundaries()

    

if __name__ == "__main__":
    config_file_path = './cfg/config.json'  # JSON 配置文件路径
    # 创建 TrajectoryFitting 对象
    virtual_lane = VirtualLane(config_file_path)
    virtual_lane.run()