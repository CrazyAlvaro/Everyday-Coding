import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from lib.tracks_to_virtual_lane import (
    TrajectoryProcessor,
    TrajectoryClustering
)

from lib.line_processor import (
    VirtualLane,
    LaneProcessor,
    IDIdentifier
)

from lib.atomic_labeling import (
    raw_tracks_generator,
    path_handler
)

def integrate_line_and_virtual(df_line_processed, df_virtual_line):
    """
    integrate virtual line and original sensed line together, and then cluster to lines
    """

    # _file_line_processed = case_folder + case_name + '/line_processed.csv'
    # _file_virtual_line = case_folder + case_name + '/lane_boundaries_tracks.csv'

    # read virtual_line, line_processed
    # df_line_processed = pd.read_csv(_file_line_processed)
    # df_virtual_line = pd.read_csv(_file_virtual_line)
    # df_line_processed = pd.read_csv(_file_line_processed)
    # df_virtual_line = pd.read_csv(_file_virtual_line)

    # concat two dataframes
    common_columns = ['x', 'y']
    df_lp = df_line_processed[common_columns]
    df_vl = df_virtual_line[common_columns]

    combined_df = pd.concat([df_lp, df_vl], ignore_index=True)

    # cluster use DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    combined_df["cluster"] = dbscan.fit_predict(combined_df)

    return combined_df

def line_id_identifier(data_folder, result_folder, case_name):
    """
    labeling integrated line_combined, then enrich original ego.csv and obj.csv
    """
    _original_line_path  = data_folder + case_name + '/line.csv'
    _line_processed_path = result_folder + case_name + '/line_combined.csv'

    # TODO how they are determined?
    _before_index = 2
    _after_index = 4

    _line_identifier = IDIdentifier(_line_processed_path, _original_line_path, _before_index, _after_index)
    _line_identifier.run(case_name)

# TODO next function, atomic_labeling
def atomic_labeling():
    pass

def tracks_to_virtual(config_file_path, case_name):
    #----轨迹过滤----#
    processor = TrajectoryProcessor(config_file_path, case_name)
    processor.process_data()
    print("轨迹过滤完成！")

    #----轨迹聚类----#
    clustering = TrajectoryClustering(config_file_path, case_name)
    clustering.run()
    print("轨迹聚类完成！")

    #----虚拟车道----#
    # 创建 TrajectoryFitting 对象
    virtual_lane = VirtualLane(config_file_path, case_name)
    # 拟合车道中心线
    virtual_lane.fit_lane_centrelines()
    # 对轨迹进行聚类，并对同类别内的轨迹求标准线
    virtual_lane.cluster_trajectories_and_refit()
    # 计算车道边界
    virtual_lane.get_lane_boundary("./results/" + case_name + "/lane_boundaries_tracks.csv") # 保存车道边界在指定文件夹下
    # # # 计算虚拟车道线与感知车道线的偏移量并进行平移
    # # virtual_lane.cal_offset_with_detected_line_and_translate()

    # 绘制拟合的车道中心线
    # virtual_lane.plot_virtual_center_lines()

    # 绘制车道边界
    # virtual_lane.plot_lane_boundaries()
    print("虚拟车道生成完成！")

def line_processing(line_file, _filter, lastfit_degree, read_interval, max_class_num):
    # line_file_path = 'data/line.csv'
    processor = LaneProcessor(line_file, _filter, lastfit_degree, read_interval, max_class_num, plot=False)
    processor.read_csv()
    processor.cluster_and_fit()
    # processor.coordinate_transform_and_concatenate()
    processor.dbscan_on_absolute_coordinates()
    # processor.line.to_csv('../data/'+ case +'/line_processed.csv', index=False)
    return processor.line

def calculate_perpendicular_points(row, distance=1):
    # Extract the velocity vector
    vx, vy = row['vx'], row['vy']

    # Find the perpendicular vector (-vy, vx)
    perp_vector = np.array([-vy, vx])

    # Normalize the perpendicular vector
    _divident = np.linalg.norm(perp_vector)
    if _divident == 0:
        return pd.DataFrame({})
    perp_unit_vector = perp_vector / _divident

    # Scale the vector by the desired distance
    offset = perp_unit_vector * distance

    # Calculate the new points
    left_point = (row['x'] + offset[0], row['y'] + offset[1])
    right_point = (row['x'] - offset[0], row['y'] - offset[1])

    return pd.DataFrame({
        'x': [left_point[0], right_point[0]],
        'y': [left_point[1], right_point[1]],
        'type': ['left', 'right']
    })

def tracks_to_virtual_line_by_width(df_tracks, lane_width=3.75):
    """
    For each track data point, create two virtual line points,
    perpendicular to the direction of the speed, width = 3.75m
    """
    # Create the new dataframe df2
    return pd.concat([calculate_perpendicular_points(row, lane_width/2) for _, row in df_tracks.iterrows()], ignore_index=True)

def plot_df(df, _case_name, _plot_name):
    # Plot the (x, y) points
    plt.figure(figsize=(8, 8))
    plt.scatter(df['x'], df['y'], color='blue', label='Points')

    # Add grid, labels, and title
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Scatter Plot of (x, y)', fontsize=14)
    plt.legend(fontsize=10)

    # Save the plot as a file
    plt.savefig(f'results/{_case_name}/{_plot_name}.png', dpi=300, bbox_inches='tight')  # Save as PNG with high resolution

    # Show the plot
    # plt.show()

if __name__ == "__main__":
    config_file_path = './config/config.json'  # JSON 配置文件路径
    case_folder = 'data/'

    # generate tracks and line_processed from data directory, get directory file info
    files_info = path_handler(case_folder)

    for _ego_file, _obj_file, _ego_config_file, _case_name in files_info:
        # case_name = "2c989271-6833-484d-b4db-3db05ed81df3"

        # log case
        print('Case {} \nego {} \nobj {} \nego_config {} \n'.format(
            _ego_file, _obj_file, _ego_config_file, _case_name))

        df_tracks = raw_tracks_generator(_ego_file, _obj_file, _ego_config_file, _case_name)
        df_tracks.to_csv(f'results/{_case_name}/tracks_result.csv', index=False)

        # process line.csv generate line_processed.csv

        _line_file = f'data/{_case_name}/line.csv'
        # 确定超参数
        _filter = 1        # LD_end - LD_start := filter
        _lastfit_degree=3        # 最后拟合车道边界的时候使用的多项式次数
        _read_interval = 1       # 读取数据的间隔
        _max_class_num = 50     # 选择的类别中最少包含的点数
        df_line_processed = line_processing(_line_file, _filter, _lastfit_degree, _read_interval, _max_class_num)
        df_line_processed.to_csv(f'results/{_case_name}/line_processed.csv', index=False)

        # for each case, generate virtual line from tracks
        # generate: lane_boundaries_tracks.csv
        # tracks_to_virtual(config_file_path, _case_name)
        df_virtual_line = tracks_to_virtual_line_by_width(df_tracks)
        df_virtual_line.to_csv(f'results/{_case_name}/line_virtual_from_tracks.csv', index=False)

        # integrate line_processed.csv and lane_boundaries_tracks.csv, cluster them
        df_combined_line = integrate_line_and_virtual(df_line_processed, df_virtual_line)
        df_combined_line.to_csv(f'results/{_case_name}/line_combined.csv', index=False)
        plot_df(df_combined_line, _case_name, 'combined_line.png')
        # TODO: Post processing cluster data

        # TODO: id lane
        # label id, enrich ego/obj to ego_with_lane_info.csv, obj_with_lane_info.csv
        # line_id_identifier(case_folder, 'results/', _case_name)

        # TODO integrate atomic_labeling
        # input ego_with_lane, obj_with_lane,
        # atomic_labeling()