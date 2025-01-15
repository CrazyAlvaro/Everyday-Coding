# 代码说明

## 代码结构
- `cfg` 配置文件
- `data` 输入的原始数据
- `result` 输出的数据处理结果
- `src` 源码
  - `src/TrajectoryExtractor.py` 轨迹提取，包括：提取相关场景车辆轨迹数据，包括ego、RU，输出object.csv/ego.csv；提取车道中心线的数据，输出lane_centerline.csv；提取车道线的数据，输出lane_boundaries.csv
  - `src/trajectory_filtering.py` 轨迹数据筛选，输出valid_trajectories.csv
  - `src/trajectory_clustering.py` 轨迹聚类并打上分类标签，输出vehicle_with_cluster_id.csv
  - `src/trajectory_clustering.py` 轨迹拟合生成车道中心线并画出虚拟车道，可视化结果并与感知结果做验证
  
## 代码使用说明
1. 新建终端
2. 运行`test.py`文件
```bash
    python test.py
```