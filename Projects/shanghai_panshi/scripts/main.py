import pandas as pd
import sys
import os
import json

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Add the parent directory of the script to the Python path
# This is needed to import the package if you're running the script directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processor import (
    preprocessor, 
    ego_interpolate, 
    obj_augment, 
    check_surrounding_objects,
    calculate_ttc,
    reference_matching,
    track_data_generator
)

from metadata import(
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

def write_output(pd_tracks, pd_tracks_meta, pd_recording_meta):
    create_directory("results")

    try:
        pd_tracks.to_csv('results/tracks_result.csv', index=False)
        pd_tracks_meta.to_csv('results/tracks_meta_result.csv', index=False)
        pd_recording_meta.to_csv('results/recording_result.csv', index=False)

        print("Output Files in Results/")
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
    _df_obj_srd = check_surrounding_objects(_df_obj_ref, ego_config)

    ###########################################
    # calculate ttc, thw 
    ###########################################
    _df_obj_cal = calculate_ttc(_df_obj_srd)

    # df_obj_augment.to_csv("nine_box_tracks.csv")

    return track_data_generator(_df_obj_cal, _ego_timestamps, columns_tracks, columns_recording_meta, _frame_rate)
    
def args_handler():
    ego_file = "ego.csv"
    obj_file = "obj.csv"
    ego_config_file = "ego_config.json"

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
    
    print(f"file path: \n ego file: {ego_file}\n obj file: {obj_file}\n ego_config file: {ego_config_file}")
    return ego_file, obj_file, ego_config_file

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


def tracks_generator():
    """
    From csv files to generate tracks
    """
    ego_file, obj_file, ego_config_file = args_handler()
    ego_data = ego_config_handler(ego_config_file)

    print("==================================================================================\n")
    print(" 5 Steps: Obj_Augment, Reference_Matching, Checking_Surroundings, TTC_THW, Tracks_Stats\n")
    print("==================================================================================\n")

    ego_obj_id = 1

    # File Input
    df_ego, df_obj = read_input(ego_file, obj_file)

    # Data Prcess and Tracks Generate
    pd_tracks, pd_tracks_meta, pd_recording = _data_processing(df_ego, df_obj, ego_obj_id, ego_data)

    # File Output
    write_output(pd_tracks, pd_tracks_meta, pd_recording)

def tracks_to_labels():
    """
    Generate [1-35] NIO atomic labeling system from tracks data
    """
    df_tracks = pd.read_csv('results/tracks_result.csv')
    pass

if __name__ == "__main__":

    # Step 1: 
    #   Input:      ego.csv, obj.csv
    #   Output:     tracks.csv tracks_meta.csv recording.csv 

    # tracks_generator()

    tracks_to_labels()