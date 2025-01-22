import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

class IDIdentifier:
    def __init__(self, file_path, line_file_path, before_change_index, after_change_index):
        self.df = pd.read_csv(file_path)
        self.line_df = pd.read_csv(line_file_path)
        self.before_change_index = before_change_index
        self.after_change_index = after_change_index
        self.ego_yaw = None
        self.ego_pos = None
        self.baseline = None
        self.lane_counts = None
        self.lane_boundaries = None
        self.get_vehicle_direction()


    def get_vehicle_direction(self):

        self.ego_yaw = self.line_df.iloc[0]['ego_yaw']
        self.ego_pos = self.line_df.iloc[0][['ego_x', 'ego_y']].values

    def tangent_at_point(self, p1, p2):
        # 计算垂线的斜率和截距
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0:
            return None  # 垂直线
        slope = - dx / dy
        intercept = p1[1] - slope * p1[0]
        return slope, intercept

    def has_intersection(self, tangent, segment):
        if tangent is None:
            return segment[0][0] <= self.baseline_point[0] <= segment[1][0]
        slope, intercept = tangent
        x1, y1 = segment[0]
        x2, y2 = segment[1]
        y1_tangent = slope * x1 + intercept
        y2_tangent = slope * x2 + intercept
        return (y1 - y1_tangent) * (y2 - y2_tangent) <= 0

    def find_baseline_and_lane_count(self):
        grouped = self.df.groupby('cluster')
        self.baseline = max(grouped, key=lambda x: len(x[1]))[1]
        print('基准线为：Cluster {}'.format(self.baseline['cluster'].iloc[0]))

        self.lane_counts = np.zeros(len(self.baseline))

        for i, (x0, y0) in enumerate(zip(self.baseline['x'], self.baseline['y'])):
            if i == 0:
                tangent = self.tangent_at_point((x0, y0), (self.baseline['x'].iloc[i+1], self.baseline['y'].iloc[i+1]))
            else:
                tangent = self.tangent_at_point((self.baseline['x'].iloc[i-1], self.baseline['y'].iloc[i-1]), (x0, y0))

            for cluster, group in grouped:
                if cluster == self.baseline['cluster'].iloc[0]:
                    continue

                points = group[['x', 'y']].values
                for j in range(len(points) - 1):
                    segment = [points[j], points[j + 1]]
                    if self.has_intersection(tangent, segment):
                        self.lane_counts[i] += 1
                        break

    def reverse_lines_if_needed(self):
        # 选取self.df中任意一条线
        sample_line = self.df[self.df['cluster'] == self.df['cluster'].unique()[0]]

        # 选择与ego_pos最近的相邻的两点
        DIS = sample_line[['x', 'y']].values - self.ego_pos
        data = np.array(DIS, dtype=np.float32)
        distances = np.linalg.norm(data, axis=1)
        closest_indice = np.argsort(distances)[:1]
        closest_points_1 = sample_line.iloc[closest_indice][['x', 'y']].values
        closest_points_2 = sample_line.iloc[closest_indice+1][['x', 'y']].values

        # 计算与ego_yaw的夹角
        vector = closest_points_2 - closest_points_1
        angle = np.arctan2(vector[0][1], vector[0][0]) - self.ego_yaw
        angle = np.degrees(angle)

        # 若夹角大于90度，则将self.df中每条线的点倒序
        if abs(angle) > 90:
            print('Reverse lines')
            self.df = self.df.sort_values(by=['cluster', 'x'], ascending=[True, False])

    def update_road_id(self):
        self.find_baseline_and_lane_count()
        print(self.lane_counts)
        change_index = next((i for i in range(1, len(self.lane_counts)) if (self.lane_counts[i-1] == self.before_change_index and self.lane_counts[i] == self.after_change_index)), None)
        self.df['road_id'] = 1
        if change_index == None:
            print('没有找到由{}变{}的索引'.format(self.before_change_index, self.after_change_index))
            return

        print('由{}变{}的索引：{}'.format(self.before_change_index, self.after_change_index, change_index))

        baseline_point = (self.baseline['x'].iloc[change_index], self.baseline['y'].iloc[change_index])

        x0, y0 = baseline_point
        if change_index == 0:
            tangent = self.tangent_at_point((x0, y0), (self.baseline['x'].iloc[change_index+1], self.baseline['y'].iloc[change_index+1]))
        else:
            tangent = self.tangent_at_point((self.baseline['x'].iloc[change_index-1], self.baseline['y'].iloc[change_index-1]), (x0, y0))

        for cluster, group in self.df.groupby('cluster'):
            if cluster == self.baseline['cluster'].iloc[0]:
                self.df.loc[group.index[change_index+1]:group.index[-1], 'road_id'] = 2
                continue

            points = group[['x', 'y']].values
            for j in range(len(points) - 1):
                segment = [points[j], points[j + 1]]
                if self.has_intersection(tangent, segment):
                    intersection_index = j
                    break
                else:
                    intersection_index = -1
            if intersection_index != -1:
                if intersection_index == 0:
                    self.df.loc[group.index[intersection_index]:group.index[-1], 'road_id'] = 2
                else:
                    self.df.loc[group.index[intersection_index-1]:group.index[-1], 'road_id'] = 2

    def sort_lanes_by_direction(self):

        def angle_from_direction(point, direction, ego_pos):
            return direction - np.arctan2(point[1]-ego_pos[1], point[0]-ego_pos[0])

        sorted_df = pd.DataFrame()
        self.df['lane_order'] = -1
        for road_id, road_group in self.df.groupby('road_id'):
            clusters = road_group['cluster'].unique()
            lane_end = {cluster: road_group[road_group['cluster'] == cluster][['x', 'y']].values[-1] for cluster in clusters}
            sorted_clusters = sorted(clusters, key=lambda cluster: angle_from_direction(lane_end[cluster], self.ego_yaw, self.ego_pos))
            print('Road_id {} 的车道排序结果为 {}'.format(road_id, sorted_clusters))
            for order, cluster in enumerate(sorted_clusters, start=0):
                # print( order)
                road_group.loc[road_group['cluster'] == cluster, 'lane_order'] = order
            sorted_df = pd.concat([sorted_df, road_group])
        self.df = sorted_df

    def generate_lane_boundaries(self):
        lane_boundaries = []

        max_road_id = self.df['road_id'].max()

        for road_id, road_group in self.df.groupby('road_id'):
            max_lane_order = road_group['lane_order'].max()

            append_flag = False
            if road_id < max_road_id:
                append_flag = True

            # print('append_flag:', append_flag)
            for lane_order in range(max_lane_order):
                cluster_left = road_group[road_group['lane_order'] == lane_order]['cluster'].values[0]
                cluster_right = road_group[road_group['lane_order'] == lane_order + 1]['cluster'].values[0]
                # print(cluster_left, cluster_right)

                if append_flag:

                    left_boundary_x_addition = self.df[(self.df['cluster']==cluster_left)&(self.df['road_id']==road_id+1)]['x'].values[0]
                    left_boundary_y_addition = self.df[(self.df['cluster']==cluster_left)&(self.df['road_id']==road_id+1)]['y'].values[0]
                    right_boundary_x_addition = self.df[(self.df['cluster']==cluster_right)&(self.df['road_id']==road_id+1)]['x'].values[0]
                    right_boundary_y_addition = self.df[(self.df['cluster']==cluster_right)&(self.df['road_id']==road_id+1)]['y'].values[0]

                left_boundary_x = road_group[road_group['lane_order'] == lane_order]['x']
                left_boundary_y = road_group[road_group['lane_order'] == lane_order]['y']
                right_boundary_x = road_group[road_group['lane_order'] == lane_order + 1]['x']
                right_boundary_y = road_group[road_group['lane_order'] == lane_order + 1]['y']

                if not left_boundary_x.empty and not right_boundary_x.empty:
                    for x_l, y_l in zip(left_boundary_x, left_boundary_y):
                        lane_boundaries.append({
                            'road_id': road_id,
                            'lane_id': lane_order,
                            'line_pos': 'left',
                            'boundary_x': x_l,
                            'boundary_y': y_l
                        })
                    if append_flag:
                        lane_boundaries.append({
                            'road_id': road_id,
                            'lane_id': lane_order,
                            'line_pos': 'left',
                            'boundary_x': left_boundary_x_addition,
                            'boundary_y': left_boundary_y_addition
                        })
                    for x_r, y_r in zip(right_boundary_x, right_boundary_y):
                        lane_boundaries.append({
                            'road_id': road_id,
                            'lane_id': lane_order,
                            'line_pos': 'right',
                            'boundary_x': x_r,
                            'boundary_y': y_r
                        })
                    if append_flag:
                        lane_boundaries.append({
                            'road_id': road_id,
                            'lane_id': lane_order,
                            'line_pos': 'right',
                            'boundary_x': right_boundary_x_addition,
                            'boundary_y': right_boundary_y_addition
                        })

        self.lane_boundaries = pd.DataFrame(lane_boundaries)

    def load_lane_boundaries(self):
        lane_boundaries = []

        for road_id, road_group in self.lane_boundaries.groupby('road_id'):
            for lane_id, lane_group in road_group.groupby('lane_id'):
                left_boundary = lane_group[lane_group['line_pos'] == 'left'][['boundary_x', 'boundary_y']].values
                right_boundary = lane_group[lane_group['line_pos'] == 'right'][['boundary_x', 'boundary_y']].values

                if len(left_boundary) > 0 and len(right_boundary) > 0:
                    polygon_points = list(left_boundary) + list(right_boundary[::-1])
                    polygon = Polygon(polygon_points)
                    lane_boundaries.append({
                        'road_id': road_id,
                        'lane_id': lane_id,
                        'polygon': polygon,
                        'left_boundary': left_boundary,
                        'right_boundary': right_boundary
                    })

        return lane_boundaries


    def find_lane_for_point(self, lane_boundaries, x, y):
        point = Point(x, y)
        for boundary in lane_boundaries:
            if boundary['polygon'].contains(point):
                return boundary['road_id'], boundary['lane_id']
        return -1, -1

    def plot_vehicle_trajectory_and_lane_boundaries(self, lane_boundaries, line_df):
        plt.figure(figsize=(10, 10))

        # 绘制车道线
        for boundary in lane_boundaries:
            left_boundary = boundary['left_boundary']
            right_boundary = boundary['right_boundary']
            plt.plot(left_boundary[:, 0], left_boundary[:, 1], 'b-', label='Left Boundary' if boundary['lane_id'] == 0 else "")
            plt.plot(right_boundary[:, 0], right_boundary[:, 1], 'r-', label='Right Boundary' if boundary['lane_id'] == 0 else "")

        # 绘制车辆轨迹
        plt.plot(line_df['x'], line_df['y'], 'g-', label='Vehicle Trajectory')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Vehicle Trajectory and Lane Boundaries')
        plt.axis('equal')
        plt.legend()
        plt.show()

    def add_info(self, file_folder, file_name):
        lane_boundaries = self.load_lane_boundaries()

        line_file_path = file_folder + file_name + '.csv'
        line_df = pd.read_csv(line_file_path)

        line_df['road_id'] = None
        line_df['lane_id'] = None

        for idx, row in line_df.iterrows():
            x, y = row['x'], row['y']
            road_id, lane_id = self.find_lane_for_point(lane_boundaries, x, y)
            line_df.at[idx, 'road_id'] = road_id
            line_df.at[idx, 'lane_id'] = lane_id

        line_df.to_csv(file_folder + file_name + '_with_lane_info.csv', index=False)
        if file_name == 'ego':
            self.plot_vehicle_trajectory_and_lane_boundaries(lane_boundaries, line_df)

    def run(self, case):
        # 按需对识别的车道线点进行倒序排序
        self.reverse_lines_if_needed()

        # 识别车道线的road_id
        self.update_road_id()
        # self.df.to_csv('../data/'+ case +'/line_processed_with_road_id.csv', index=False)

        # 识别车道线的lane_id
        self.sort_lanes_by_direction()
        # self.df.to_csv('../data/'+ case +'/line_processed_with_lane_order.csv', index=False)

        # 得到道路边界
        self.generate_lane_boundaries()
        # self.lane_boundaries.to_csv('../data/'+ case +'/lane_boundaries.csv', index=False)

        # 为车辆轨迹数据添加车道信息
        self.add_info('../data/'+ case +'/', 'ego')
        self.add_info('../data/'+ case +'/', 'obj')


if __name__ == '__main__':
    # case = 'Xianyousuanfa'
    case = 'Ganzhide'
    # file_path = '../data/'+ case +'/line_processed.csv'
    file_path = './results/line_processed.csv'
    # line_file_path = '../data/' + case + '/line.csv'
    line_file_path = './line.csv'
    road_identifier = IDIdentifier(file_path, line_file_path, before_change_index=2, after_change_index=4)
    road_identifier.run(case)