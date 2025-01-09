import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
# import matplotlib.colors as colors


def load_data(file):
    # 读取CSV文件
    df = pd.read_csv(file)
    return df

def plot_data(data, frame):
    """
    根据给定的数据和时间戳绘制散点图和速度向量图。

    Args:
        data: 包含轨迹数据的DataFrame。
        timestamp: 要绘制的特定时间戳。

    Returns:
        matplotlib.figure.Figure: 生成的图形对象。
    """

    # 筛选出指定时间戳的数据

    # print(data['timestamp'][1])
    # print(type(data['timestamp'][1]))
    # print(type(ts))
    # print(ts)

    frame_id= np.int64(frame)
    filtered_data = data[data['frame'] == frame_id]

    # 获取唯一的 trackId 列表
    # unique_track_ids = filtered_data['trackId'].unique()

    # 创建一个颜色列表
    # cmap = plt.cm.get_cmap('jet')
    # colors = [cmap(i) for i in np.linspace(0, 1, len(unique_track_ids))]

    print("Current frame: {} with data length {}".format(frame_id, len(filtered_data)))

    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 8))

    filtered_data['trackId'] = filtered_data['trackId'].astype(int)
    filtered_data['vehicle_id'] = filtered_data['vehicle_id'].astype(int)

    # 绘制散点图，每个点代表一个trackID
    scatter = ax.scatter(filtered_data['xCenter'], filtered_data['yCenter'])

    # 为每个点添加速度向量
    for _index, row in filtered_data.iterrows():
        x, y = row['xCenter'], row['yCenter']
        u, v = row['xVelocity'], row['yVelocity']
        ax.quiver(x, y, u, v, angles='uv', scale_units='width', color='blue')

    # set x, y axis in the same unit length 
    plt.axis('equal')

    # 添加 trackId 和 vehicle_id  
    for i, (x, y) in enumerate(zip(filtered_data['xCenter'], filtered_data['yCenter'])):
        # print(type(filtered_data['trackId']))
        curr_track_id  = filtered_data['trackId'].iloc[i]
        plt.text(x, y, f"tkId: {curr_track_id}")
    
    correspondence_table = filtered_data[['trackId', 'vehicle_id']]

    table = plt.table(cellText=correspondence_table.values,
                  colLabels=correspondence_table.columns,
                  loc='bottom', cellLoc='center',
                  bbox=[0, -0.5, 1, 0.4])
    table.set_fontsize(12)  # 设置字体大小
    table.scale(2, 1.5)  # 调整表格大小

    # 添加标题和坐标轴标签
    ax.set_title(f'Track Positions and Velocities at frame: {frame_id}')
    ax.set_xlabel('xCenter')
    ax.set_ylabel('yCenter')

    # 添加颜色条
    # plt.colorbar(scatter, label='trackId')

    return fig

def main():
    file = '../results/tracks_result.csv'

    st.title('交互式轨迹可视化')

    df = load_data(file)

    # print(len(df))

    frame= st.text_input('请输入要查询的frame:')

    if st.button('绘制'):
        fig = plot_data(df, frame)
        st.pyplot(fig)

if __name__ == '__main__':
    main()