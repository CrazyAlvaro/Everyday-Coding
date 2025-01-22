from lib.line_processor import (
    LaneProcessor
)

if __name__ == '__main__':
    # 使用示例
    '''
    Ganzhide
    Duochulai
    Xianyousuanfa
    '''
    # case = "Newcase2"
    # case = "Ganzhide"
    # line_case_path = '../data/'+ case +'/line.csv'
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
