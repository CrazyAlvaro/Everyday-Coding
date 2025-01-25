import pandas as pd
import sys
import os
import json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add the parent directory of the script to the Python path
# This is needed to import the package if you're running the script directly
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .process import (
    preprocessor,
    ego_interpolate,
    obj_augment,
    calculate_ttc,
    reference_matching,
    track_data_generator
)

from .metadata import(
    time_interpolate_columns,
    shift_interpolate_columns,
    columns_tracks,
    columns_recording_meta,
    obj_time_cols,
    obj_shift_cols,
    columns_tracks_meta
)

_log_level = "debug"
_log_level = "run"

# Read the names of all folders under the given path
def get_all_file_name(path):
    case_name = []
    if os.path.exists(path):
        for file in os.listdir(path):
            if os.path.isdir(path + file):
                case_name.append(file)
    else:
        print("path not exist")
    return case_name


def create_directory(path):
    try:
        os.makedirs(path, exist_ok=True)  # exist_ok=True prevents FileExistsError
        print(f"Directory '{path}' created (or already existed).")
    except OSError as error:
        print(f"Error creating directory '{path}': {error}")

def read_input(ego_file = "ego.csv", obj_file = "obj.csv"):
    try:
        df_ego = pd.read_csv(ego_file)
        df_obj = pd.read_csv(obj_file)
    except ValueError as e:
        print(f"Reading input files failed: {e}")
        exit(1)

    if _log_level == "debug":
        print("ego len: {}, obj len: {}".format(len(df_ego), len(df_obj)))

    return df_ego, df_obj

def write_output(pd_tracks, case_name):
    create_directory(f"results/{case_name}")

    try:
        # 创建新的DataFrame仅保留所需的列并重命名
        pd_tracks = pd.DataFrame({
            'timestamp': pd_tracks['timestamp'],
            'vehicle_id': pd_tracks['vehicle_id'].astype(int),
            'length': pd_tracks['length'],
            'width': pd_tracks['width'],
            'height': pd_tracks['height'],
            'x': pd_tracks['xCenter'],
            'y': pd_tracks['yCenter'],
            'yaw': pd_tracks['yaw'],
            'yaw_rate': pd_tracks['yaw_rate'],
            'vx': pd_tracks['xVelocity'],
            'vy': pd_tracks['yVelocity'],
            'ax': pd_tracks['xAcceleration'],
            'ay': pd_tracks['yAcceleration'],

        })
        # 保存到csv文件
        pd_tracks.to_csv(f'results/{case_name}/tracks_result.csv', index=False)
        print("Output Tracks File in Results/")
    except OSError as error:
        print("Error writing tracks output")

def _data_processing(df_ego, df_obj, ego_obj_id, ego_config):
    _verbose = True if _log_level == "debug" else False
    df_ego.sort_values(by='ts', inplace=True)
    _ego_timestamps = df_ego['ts']

    _frame_rate = int(10**9 / (_ego_timestamps[1] - _ego_timestamps[0]))
    # print(_frame_rate)

    ###########################################
    # augment ego data with timestamp in obj
    ###########################################
    _df_ego_augment, _ts_ego= preprocessor(df_ego, df_obj, ego_obj_id, shift_interpolate_columns, _verbose)

    ###########################################
    # interpolate ego columns
    ###########################################
    _df_ego_interpolated = ego_interpolate(_df_ego_augment, time_interpolate_columns,shift_interpolate_columns, ego_config, _verbose)

    ###########################################
    # concat df_obj with _df_ego_interpolated
    # for each obj_id, augment timestamp with
    # range include all timestamps using ego frame rate
    ###########################################
    _df_obj_augment = obj_augment(df_obj, _df_ego_interpolated, _ts_ego, obj_time_cols, obj_shift_cols, _verbose)

    if _verbose:
        print("=== main_1 === {}".format(len(_df_obj_augment)))

    ###########################################
    # Reference re-match for obj values based on ego reference frame
    ###########################################
    _df_obj_ref = reference_matching(_df_obj_augment, ego_obj_id, columns_tracks, _verbose)

    ###########################################
    # Given obj and ego state value matched on the same reference frame
    # check each vehicle's surrounding objects at each timestamp
    ###########################################
    # _df_obj_srd = check_surrounding_objects(_df_obj_ref, ego_config)

    ###########################################
    # calculate ttc, thw
    ###########################################
    # _df_obj_cal = calculate_ttc(_df_obj_srd)

    # df_obj_augment.to_csv("nine_box_tracks.csv")

    # return track_data_generator(_df_obj_cal, _ego_timestamps, columns_tracks, columns_recording_meta, _frame_rate)
    return track_data_generator(_df_obj_ref, _ego_timestamps, columns_tracks, columns_recording_meta, _frame_rate)


def ego_config_handler(ego_config_file):
    with open(ego_config_file, 'r') as file:
        _ego_data = json.load(file)

    required_keys = {"height", "width", "length"}

    # validate
    if required_keys.issubset(_ego_data.keys()):
        print("Ego config all keys are present.")
    else:
        missing_keys = required_keys - _ego_data.keys()
        print(f"Missing ego keys: {missing_keys}")

    return _ego_data

def path_handler(data_path):
    # data_path="data/"
    case_names = get_all_file_name(data_path)

    if not case_names:
        print("No case names found.")
        exit(1)

    # Store all file paths
    files_info = []

    for case_name in case_names:
        ego_file = f"{data_path}/{case_name}/ego.csv"
        obj_file = f"{data_path}/{case_name}/obj.csv"
        ego_config_file = "config/ego_config.json"

        if len(sys.argv) > 1:
            print("Arguments passed:", sys.argv[1:])  # Print all arguments except the script name

            # Accessing specific arguments
            if len(sys.argv) > 2:
                ego_file = sys.argv[1]
                obj_file = sys.argv[2]
                ego_config_file = sys.argv[3]
                print(f"ego file: {ego_file}, obj file: {obj_file}, ego_config file: {ego_config_file}")
            else:
                print("Please provide ego, obj and ego_config 3 files' path.")
        else:
            print("No ego and obj file path provided.")

        print(f"file path for case '{case_name}': \n eg file: {ego_file}\n obj file: {obj_file}\n ego_config file: {ego_config_file}")

        # Store file paths in files_info
        files_info.append((ego_file, obj_file, ego_config_file, case_name))

    return files_info

# def raw_tracks_generator(data_folder):
def raw_tracks_generator(ego_file, obj_file, ego_config_file, case_name):

    # for ego_file, obj_file, ego_config_file, case_name in files_info:

    ego_data = ego_config_handler(ego_config_file)
    ego_obj_id = 1

    # File Input
    df_ego, df_obj = read_input(ego_file, obj_file)

    # Data Prcess and Tracks Generate
    pd_tracks, _, _= _data_processing(df_ego, df_obj, ego_obj_id, ego_data)

    # File Output
    write_output(pd_tracks, case_name)