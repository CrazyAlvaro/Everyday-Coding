# 代码说明

## 代码结构

- `data/` : 原始车端提取的数据
  - `data/Duochulai/` : 对应车端提取数据“多出来的车道线”
  - `data/Ganzhide/` : 对应车端提取数据“感知的路口分段”
  - `data/Xianyousuanfa/` : 对应车端提取数据“现有算法车道线解析不准”
- `src/` : 
  - `src/LineProcessor.py` : 根据多帧数据融合得到车道线
  - `src/IDIdentifier.py` : 根据得到的融合车道线信息，识别RoadID和LaneID，并给ego.csv和obj.csv赋值

## 使用说明

1. 在`src/LineProcessor.py`中，选择需要进行车道线融合提取的case，并确定超参数

```python
if __name__ == '__main__':
    # 使用示例
    '''
    Ganzhide
    Duochulai
    Xianyousuanfa
    '''
    case = "Ganzhide"
    # 确定超参数
    filer = 60
    lastfit_degree=3        # 最后拟合车道边界的时候使用的多项式次数
    read_interval = 2       # 读取数据的间隔
    max_class_num = 200     # 选择的类别中最少包含的点数

    processor = LaneProcessor('../data/'+ case +'/line.csv', 60, plot=True)
    processor.read_csv()
    processor.cluster_and_fit()
    processor.dbscan_on_absolute_coordinates()、
    processor.line.to_csv('../data/'+ case +'/line_processed.csv', index=False)
```

2. 运行`src/LineProcessor.py`

```bash
python LineProcessor.py
```

3. 运行结束后，弹出可视化结果并保存至`extracted_line.png`中，同时保存提取的车道线文件至`line_processed.csv`中

4. 在`src/IDIdentifier.py`中，选择需要进行道路ID和车道ID识别的case

```python
if __name__ == '__main__':
    # 使用示例
    '''
    Ganzhide
    Duochulai
    Xianyousuanfa
    '''
    case = 'Xianyousuanfa'
    file_path = '../data/'+ case +'/line_processed.csv'
    line_file_path = '../data/' + case + '/line.csv'
    road_identifier = IDIdentifier(file_path, line_file_path, before_change_index=2, after_change_index=4)
    road_identifier.run(case)
```

5. 可在`run()`函数中，选择需要保存的csv数据

```python
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
```

6.  运行`src/IDIdentifier.py`

```bash
python IDIdentifier.py
```

7. 运行结束后，弹出可视化结果（为包含自车运行轨迹的车道线），并为`ego.csv`和`obj.csv`分别打上`RoadID`和`LaneID`标签，打标签之后的结果保存在`ego_with_lane_info.csv`和`obj_with_lane_info.csv`中

## 代码超参数及调参建议

1. `self.lastfit_degree`

   - 含义：最后拟合车道边线的时候，拟合多项式的系数
   - 说明：一般推荐值为3，当道路曲率较大时，三次多项式不能很好拟合，此时可以将该值调大，一般建议不超过10，可以先试试6左右

2. `self.read_interval`

     - 含义：读取车端信息时，间隔读取，该值决定间隔的行数
     - 说明：一般推荐值为2，当道路结构比较简单时（如平行的大直路），可将该值调大，从而可以筛选掉一些干扰点

3. `self.max_class_num`

    - 含义：最后聚类结束进行车道边界拟合之前，筛选掉包含少量点的类别的阈值
    - 说明：可与`self.max_class_num`配合使用，增大该值，一些小的线段将被筛选

## 代码运行效果

### 1. Case：多出来的车道线

<img src="./data/Duochulai/extracted_line.png" width = 80% > 

### 2. Case：感知的车道分段

<img src="./data/Ganzhide/extracted_line.png" width = 80% > 

### 3. Case：现有算法车道线解析不准

<img src="./data/Xianyousuanfa/extracted_line.png" width = 80% > 