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

def display_dataframe_in_multiple_tables(df, columns_per_table=10):

    """
    Displays a DataFrame with more than 40 columns in multiple tables
    using plt.table, with specified columns per table and
    including index and id columns in each table.

    Args:
        df: The input DataFrame.
        columns_per_table: The number of columns to display in each table.
        index_col: The name of the index column.
        id_col: The name of the id column.

    Returns:
        fig
    """

    num_cols = len(df.columns)
    num_tables = np.ceil(num_cols / columns_per_table).astype(int)

    fig, axes = plt.subplots(nrows=num_tables, figsize=(12, num_tables*3))

    for i in range(num_tables):
        start_col = i * columns_per_table
        end_col = min((i + 1) * columns_per_table, num_cols)
        table_cols = ['vehicle_id'] + list(df.columns[start_col:end_col])
        table_data = df[table_cols].head(20).values  # Display first 20 rows

        axes[i].axis('off')
        axes[i].table(cellText=table_data,
                    #  rowLabels=df.index[:20],
                     colLabels=table_cols,
                     cellLoc='center',
                     loc='center')
        # axes[i].set_title(f"Table {i+1}")

    plt.tight_layout()
    plt.show()
    return fig

def processing_data(data, frame):
    # 筛选出指定时间戳的数据

    frame_id= np.int64(frame)
    filtered_data = data[data['frame'] == frame_id]
    print("Current frame: {} with data length {}".format(frame_id, len(filtered_data)))

    correspondence_table = filtered_data[['trackId', 'timestamp','frame','vehicle_id','laneId','class_str',
              'ru1', 'ru2', 'ru3', 'ru4', 'ru5', 'ru6', 'ru7', 'ru8', 'ru9', 'ru10',
              'ru11', 'ru12', 'ru13', 'ru14', 'ru15', 'ru16', 'ru17', 'ru18', 'ru19', 'ru20',
              'ru21', 'ru22', 'ru23', 'ru24', 'ru25', 'ru26', 'ru27', 'ru28', 'ru29', 'ru30',
              'ru31', 'ru32', 'ru33', 'ru34', 'ru35']]
#
    df = convert_df_to_int64(correspondence_table)

    return df

def plot_data(data, frame):
    """
    根据给定的数据和时间戳绘制散点图和速度向量图。

    Args:
        data: 包含轨迹数据的DataFrame。
        frame: 要绘制的特定时间戳。

    Returns:
        matplotlib.figure.Figure: 生成的图形对象。
    """

    # 筛选出指定时间戳的数据

    frame_id= np.int64(frame)
    filtered_data = data[data['frame'] == frame_id]

    # print("Current frame: {} with data length {}".format(frame_id, len(filtered_data)))

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

    for i, (x, y) in enumerate(zip(filtered_data['xCenter'], filtered_data['yCenter'])):
        # print(type(filtered_data['trackId']))
        curr_track_id  = filtered_data['trackId'].iloc[i]
        plt.text(x, y, f"tk: {curr_track_id}")

    # 添加标题和坐标轴标签
    ax.set_title(f'Track Positions and Velocities at frame: {frame_id}')
    ax.set_xlabel('xCenter')
    ax.set_ylabel('yCenter')

    return fig

def main():
    file = 'results/tracks_result.csv'

    st.title('交互式轨迹可视化')

    df = load_data(file)

    # print(len(df))

    frame= st.text_input('请输入要查询的frame:')

    if st.button('绘制'):
        fig = plot_data(df, frame) # multiple tables
        st.pyplot(fig)

        # fig = plot_data(df, frame) # original: 1 table
        df_processed = processing_data(df, frame)
        fig_1 = display_dataframe_in_multiple_tables(df_processed, 6)
        st.pyplot(fig_1)


if __name__ == '__main__':
    main()


# def plot
    # correspondence_table = filtered_data[['trackId', 'vehicle_id','laneId','class_str',
            #   'ru1', 'ru2', 'ru3', 'ru4', 'ru5', 'ru6', 'ru7', 'ru8', 'ru9', 'ru10']]
            # 'timestamp','frame',
            #   'ru11', 'ru12', 'ru13', 'ru14', 'ru15', 'ru16', 'ru17', 'ru18', 'ru19', 'ru20',
            #   'ru21', 'ru22', 'ru23', 'ru24', 'ru25', 'ru26', 'ru27', 'ru28', 'ru29', 'ru30',
            #   'ru31', 'ru32', 'ru33', 'ru34', 'ru35']]

    # table_df = convert_df_to_int64(correspondence_table)
    # table_df = correspondence_table

    # table = plt.table(cellText=table_df.values,
                #   colLabels=table_df.columns,
                #   loc='bottom', cellLoc='center',
                #   bbox=[-0.5, -0.5, 2.5, 0.4])
    # table.auto_set_font_size(False)
    # table.set_fontsize(10)  # 设置字体大小
    # table.scale(5, 4.5)  # 调整表格大小

    # 添加标题和坐标轴标签
    # ax.set_title(f'Track Positions and Velocities at frame: {frame_id}')
    # ax.set_xlabel('xCenter')
    # ax.set_ylabel('yCenter')