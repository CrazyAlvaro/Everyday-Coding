import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
# import matplotlib.colors as colors

def convert_df_to_int64(df):
    """
    Converts all numeric columns in a DataFrame to np.int64.

    Args:
      df: The pandas DataFrame to convert.

    Returns:
      A new DataFrame with all numeric columns converted to np.int64.
    """

    df_int64 = df.copy()  # Create a copy to avoid modifying the original DataFrame

    for col in df_int64.columns:
      if pd.api.types.is_numeric_dtype(df_int64[col]):
        df_int64[col] = df_int64[col].astype(np.int64)

    return df_int64

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
    fig, ax = plt.subplots(figsize=(12, 10))

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

    correspondence_table = filtered_data[['trackId', 'vehicle_id', 'precedingId',
                                          'followingId', 'leftPrecedingId', 'leftAlongsideId', 'leftFollowingId',
                                          'rightPrecedingId', 'rightAlongsideId', 'rightFollowingId',
                                          'laneId']]
    
    table_df = convert_df_to_int64(correspondence_table)

    table = plt.table(cellText=table_df.values,
                  colLabels=table_df.columns,
                  loc='bottom', cellLoc='center',
                  bbox=[-0.5, -0.5, 1.8, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # 设置字体大小
    table.scale(4, 3.5)  # 调整表格大小

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