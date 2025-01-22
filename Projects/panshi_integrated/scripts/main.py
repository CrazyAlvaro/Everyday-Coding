
from lib.tracks_to_virtual_lane import (
    TrajectoryProcessor,
    TrajectoryClustering
)
from lib.line_processor import (
    VirtualLane,
    LaneProcessor
)
from lib.atomic_labeling import raw_tracks_generator

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

def line_processing(case_name):
    line_case_path = 'data/'+ case_name +'/line.csv'
    # line_file_path = 'data/line.csv'
    # 确定超参数
    _filter = 1        # LD_end - LD_start := filter
    lastfit_degree=3        # 最后拟合车道边界的时候使用的多项式次数
    read_interval = 1       # 读取数据的间隔
    # max_class_num = 200     # 选择的类别中最少包含的点数
    max_class_num = 10
    processor = LaneProcessor(line_case_path, _filter, lastfit_degree, read_interval, max_class_num, plot=False)
    processor.read_csv()
    processor.cluster_and_fit()
    # processor.coordinate_transform_and_concatenate()
    processor.dbscan_on_absolute_coordinates()
    # processor.line.to_csv('../data/'+ case +'/line_processed.csv', index=False)
    processor.line.to_csv('./results/' + case_name + '/line_processed.csv', index=False)

if __name__ == "__main__":
    config_file_path = './config/config.json'  # JSON 配置文件路径
    case_folder = 'data/'

    # generate tracks and line_processed from data directory, get directory file info
    files_info = raw_tracks_generator(case_folder)

    for _ego_file, _obj_file, _ego_config_file, _case_name in files_info:
        # case_name = "2c989271-6833-484d-b4db-3db05ed81df3"

        # process line.csv generate line_processed.csv
        line_processing(_case_name)

        # for each case directory, generate virtual line from tracks
        # generate: lane_boundaries_tracks.csv
        tracks_to_virtual(config_file_path, _case_name)