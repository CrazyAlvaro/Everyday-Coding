import json
import pandas as pd

def read_json(file_path):
    """读取 JSON 文件并返回数据"""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_lane_boundaries(data):
    """提取 road_network 下 vertex_descriptor 中的 lane_boundary"""
    lane_boundaries = []
    for boundary in data['road_network'][0]['vertex_descriptor']['lane_boundary']:
        for point in boundary['boundary_point']:
            lane_boundaries.append({
                'boundary_id': boundary['id'],
                'x': point[0],
                'y': point[1]
            })
    return lane_boundaries

def save_to_csv(lane_boundaries, output_file):
    """将提取的 lane_boundary 保存为 CSV 文件"""
    df = pd.DataFrame(lane_boundaries)
    df.to_csv(output_file, index=False, encoding='utf-8')

def main():
    # 读取 JSON 文件
    data = read_json('路口.json')
    # 提取 lane_boundary
    lane_boundaries = extract_lane_boundaries(data)
    # 保存为 CSV 文件
    save_to_csv(lane_boundaries, 'lane_boundaries.csv')

if __name__ == "__main__":
    main()