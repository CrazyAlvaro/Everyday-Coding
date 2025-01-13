import math
import pandas as pd
import numpy as np
from bisect import bisect_left, bisect_right
from tqdm import tqdm

def preprocessor(df_ego, df_obj, ego_obj_id, shift_intrplt_cols, _verbose=False):
    # df_ego 加入 obj_id 字段 取 1
    df_ego['obj_id'] = ego_obj_id 

    # NOTE: Assume we have laneId 
    # add column 'road_id' and 'laneId' to both ego and obj if needed
    for _col in shift_intrplt_cols:
        if _col not in df_ego.columns:
            df_ego[_col] = 0

        if _col not in df_obj.columns:
            df_obj[_col] = 0

    ###########################################
    # Preprocessing
    ###########################################

    ts_obj = df_obj['ts'].unique()
    ts_ego = df_ego['ts'].unique()
    ts_obj_not_ego = [ts for ts in ts_obj if ts not in ts_ego]
    ts_obj_and_ego = [ts for ts in ts_obj if ts in ts_ego]

    if _verbose:
        print(f"unique ts in obj: {len(ts_obj)}")
        print(f"unique ts in ego: {len(ts_ego)}")
        print(f"unique ts in obj_not_ego: {len(ts_obj_not_ego)}")
        print(f"unique ts in obj_and_ego: {len(ts_obj_and_ego)}")

    # augment ego with ts in obj
    df_obj_ts = pd.DataFrame(np.array(ts_obj_not_ego), columns=['ts'])

    # concat with df_ego
    df_ego_augment = pd.concat([df_ego, df_obj_ts], axis=0, ignore_index=True)

    # sort by ts and reindex
    df_ego_augment.sort_values(by='ts', inplace=True)
    df_ego_augment.reset_index(drop=True, inplace=True)

    # update obj_id of new rows
    df_ego_augment['obj_id'] = ego_obj_id

    ###########################################
    # Interpolation for x, y values
    ###########################################

    # 1 create new index with ts
    df_ego_augment['timestamp'] = pd.to_datetime(df_ego_augment['ts'])
    df_ego_augment.set_index('timestamp', inplace=True)

    if _verbose:
        print("========= preprocessor ===========")
        df_ego_augment.to_csv('./debug/ego_augment.csv', index=False)

    return df_ego_augment, ts_ego 

def ego_interpolate(df_ego_interpolated, time_intrplt_cols, shift_intrplt_cols, ego_config, _verbose=False):
    for col in time_intrplt_cols:
        df_ego_interpolated[col] = df_ego_interpolated[col].interpolate(method='time')
    
    for col in shift_intrplt_cols:
        df_ego_interpolated[col] = df_ego_interpolated[col].ffill()
    
    # set ego length, width to 0, since we don't know
    df_ego_interpolated['length'] = ego_config['length'] 
    df_ego_interpolated['width']  = ego_config['width'] 
    df_ego_interpolated['height']  = ego_config['height'] 
    
    # assume ego is car
    df_ego_interpolated['class_str']  = 'car' 

    if _verbose:
        print("========= ego_interpolate ===========")
        df_ego_interpolated.to_csv('./debug/ego_augment_interpolated.csv', index=False)
    
    return df_ego_interpolated


def _get_timestamp_range(_timestamps, target_timestamps, _key='ts'):
    """
    return the timestamp range from df_timestamps for target_timestamps
    Example:
        min, max = target_timestamps.min(), target_timestamps.max()
        return [lowerbound(df_timestamps, min), upperbound(df_timestamps, max)]
    """
    _ts_min, _ts_max = target_timestamps.min(), target_timestamps.max()

    sorted_arr = np.sort(_timestamps)

    _lower_bound = max(0, bisect_left(sorted_arr, _ts_min))
    _upper_bound = min(len(sorted_arr), bisect_right(sorted_arr, _ts_max))

    return sorted_arr[_lower_bound:_upper_bound] 

def obj_augment(df_obj, df_ego_interpolated, ts_ego, obj_time_cols, obj_shift_cols, _verbose=False):

    # construct output dataframe
    df_output = df_ego_interpolated.copy()

    # Filter row only with original ts
    df_output = df_output[df_output['ts'].isin(ts_ego)] 

    if _verbose:
        print("========= obj_augment ===========")
        print("df_obj {} df_ego{}".format(len(df_obj), len(df_output)))

    ## START
    for _obj_id, _df_group in tqdm(df_obj.groupby('obj_id'), desc="Augment Obj timestamps"):
        if _obj_id == 1:
            continue 

        _df_group.sort_values(by='ts', inplace=True)

        # get corresponding timestamp range from df_ego
        _group_ts = _df_group['ts']
        _group_ts_ego_range = _get_timestamp_range(ts_ego, _group_ts)

        # print(_group_ts_ego_range)
        _group_new_ts = [ts for ts in _group_ts_ego_range if ts not in _group_ts]

        # augment current obj_id within all timestamp
        _df_group_ego_ts = pd.DataFrame(np.array(_group_new_ts), columns=['ts'])
        _df_group_augment = pd.concat([_df_group, _df_group_ego_ts], axis=0, ignore_index=True)

        # sort by ts and reindex
        _df_group_augment['timestamp'] = pd.to_datetime(_df_group_augment['ts'])
        _df_group_augment.set_index('timestamp', inplace=True)

        # interpolate by time
        for col in obj_time_cols:
            _df_group_augment[col] = _df_group_augment[col].interpolate(method='time')

        # interpolate by shift
        for col in  obj_shift_cols:
            _df_group_augment[col] = _df_group_augment[col].ffill()
        
        # contain timestamp that is only in original ts_ego 
        _df_group_augment = _df_group_augment[_df_group_augment['ts'].isin(ts_ego)] 

        # drop duplicate ts
        _df_group_augment = _df_group_augment.drop_duplicates(subset=['ts'], keep='first')

        # LOOP_END add to output dataframe
        df_output = pd.concat([df_output, _df_group_augment], axis=0, ignore_index=True)
    # END

    if _verbose:
        df_output.to_csv('./debug/obj_augment.csv', index=True)

    return df_output 

def _ensure_columns_exist_robust(df, column_names, verbose=False):
    """
    Robust version of ensure_columns_exist, handling potential errors.

    Args:
        df: The Pandas DataFrame.
        column_names: A list of strings representing column names.

    Returns:
        The DataFrame, potentially with new columns added, or None if an error occurs.
        Prints messages indicating actions taken or errors.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input 'df' must be a Pandas DataFrame.")
        return None

    if not column_names:  # Handle empty list of column names
        print("No column names provided. Nothing to do.")
        return df

    if not all(isinstance(col, str) for col in column_names):
        print("Error: All elements in 'column_names' must be strings.")
        return None
    
    try:
        for col_name in column_names:
            if col_name not in df.columns:
                df[col_name] = pd.NA

                if verbose:
                    print(f"Column '{col_name}' created.")
            else:
                if verbose:
                    print(f"Column '{col_name}' already exists.")
        return df
    except Exception as e: # Catch potential exceptions during column creation
        print(f"An error occurred: {e}")
        return None
    

def _rotate_vector(x1, y1, angle_radians):
    """
    Rotates a vector from one coordinate system (RF1) to another (RF2).

    Args:
        x1: The x-component of the vector in RF1.
        y1: The y-component of the vector in RF1.
        angle_radians: The angle of rotation from RF1 to RF2, measured counter-clockwise.

    Returns:
        A tuple containing the x and y components of the rotated vector in RF2 (x2, y2).
        Returns None if input is invalid.
    """
    if not isinstance(x1, (int, float)) or not isinstance(y1, (int, float)) or not isinstance(angle_radians, (int, float)):
      print("x1, y1 and angle must be numbers")
      return None

    x2 = x1 * math.cos(angle_radians) - y1 * math.sin(angle_radians)
    y2 = x1 * math.sin(angle_radians) + y1 * math.cos(angle_radians)
    return (x2, y2)

def _calculate_track_statistics(df_tracks): 
    """
    Calculates track statistics for each vehicle ID in the DataFrame.

    Args:
      df_tracks: DataFrame containing track information with columns:
                 'frame', 'trackId', 'x', 'y', 'XVelocity', 'dhw', 'thw', 'ttc', 'lane_id'

    Returns:
      A new DataFrame containing track statistics for each vehicle ID.
    """

    track_stats = []
    for track_id, group in tqdm(df_tracks.groupby('trackId'), desc="Tracks Statistics: "):
        first_row = group.iloc[0] 

        # Calculate the differences between consecutive points
        group['x_diff'] = group['xCenter'].diff()
        group['y_diff'] = group['yCenter'].diff()
        group['x_diff'] = group['x_diff'].fillna(0)
        group['y_diff'] = group['y_diff'].fillna(0)
        group['distance'] = np.sqrt(group['x_diff']**2 + group['y_diff']**2)
        total_distance = group['distance'].sum()

        # Calculate velocity statistics
        min_x_velocity = group['xVelocity'].min()
        max_x_velocity = group['xVelocity'].max()
        mean_x_velocity = group['xVelocity'].mean()

        # Calculate minimum values for DHW, THW, and TTC
        min_dhw = group['dhw'].min()
        min_thw = group['thw'].min()
        min_ttc = group['ttc'].min()

        # Count lane changes within same 'road_id' (simple approach)
        num_lane_changes = 0
        for _, road_group in group.groupby('roadId'):
          curr_lane_diff = road_group['laneId'].diff().dropna()
          num_lane_changes += (curr_lane_diff != 0).sum()

        # Count lane changes (simple approach)
        # lane_diff = group['laneId'].diff().dropna()
        # num_lane_changes = (lane_diff != 0).sum()

        # Create a dictionary for track statistics
        track_stats.append({
            'trackId': track_id,
            'obj_id': first_row['obj_id'],
            'length': first_row['length'],
            'width': first_row['width'],
            'height': first_row['height'],
            'initialFrame': group['frame'].min(),
            'finalFrame': group['frame'].max(),
            'numFrames': len(group),
            'class': first_row['class_str'],
            'drivingDirection': 2 if first_row['laneId'] >= 0 else 1,
            'traveledDistance': total_distance,
            'minXVelocity': min_x_velocity,
            'maxXVelocity': max_x_velocity,
            'meanXVelocity': mean_x_velocity,
            'minDHW': min_dhw,
            'minTHW': min_thw,
            'minTTC': min_ttc,
            'numLaneChanges': num_lane_changes
        })

    return pd.DataFrame(track_stats)

def calculate_ttc(df):
    """
    Calculates Time-to-Collision (TTC) for each row in the DataFrame.

    Args:
      df: pandas DataFrame with columns: 'precedingId', 'xVelocity', 'precedingXVelocity', 'frontSightDistance'

    Returns:
      A new DataFrame with an additional column 'TTC'.
    """
    df['ttc'] = 0 # Initialize TTC to 0 
    df['ttc'] = df['ttc'].astype(float)

    for index, row in tqdm(df.iterrows(), desc="TTC, THW Computing: "):
        if row['precedingId'] != 0: 
            # If a preceding vehicle exists
            relative_velocity = row['xVelocity'] - row['precedingXVelocity'] 
            if relative_velocity > 0:  # Ego vehicle is faster than preceding vehicle
                df.loc[index, 'ttc'] = row['dhw'] / relative_velocity
            elif row['xVelocity'] != 0:
                df.loc[index, 'thw'] = row['dhw'] / row['xVelocity']
            else:
                # current xVelocity == 0
                df.loc[index, 'thw'] = np.inf 
        else:
            # no preceding vehicle exist
            df.loc[index, 'thw'] = 0
    return df

def _create_tracks(df, _frame_dict):
    """
    Assigns unique track IDs and frame numbers to objects in a DataFrame.

    Args:
        df: DataFrame with 'objectID' and 'timestamp' columns.

    Returns:
        DataFrame with added 'track_ID' and 'frame' columns, or None if input is invalid.
    """
    _objectID = "obj_id"
    _timestamp = "ts"
    _track_ID = "trackId"
    _frame = "frame"

    if not isinstance(df, pd.DataFrame):
        print("Input must be a Pandas DataFrame.")
        return None

    if not {_objectID, _timestamp}.issubset(df.columns):
        print("DataFrame must contain '{}' and '{}' columns.".format(_objectID, _timestamp))
        return None

    df = df.sort_values(by=[_objectID, _timestamp])  # Important: Sort by objectID then timestamp

    track_id_map = {}  # Dictionary to store objectID to track_ID mapping
    current_track_id = 1 # object starts from 1, track record 0 stand for no object later
    df[_track_ID] = -1  # Initialize track_ID column
    df[_frame] = -1       # Initialize frame column

    for index, row in df.iterrows():
        object_id = row[_objectID]

        if object_id not in track_id_map:
            track_id_map[object_id] = current_track_id
            current_track_id += 1

        df.loc[index, _track_ID] = track_id_map[object_id]

        # position frame within each track from timestamp
        _np_int_ts = np.int64(row[_timestamp]) 
        if _np_int_ts not in _frame_dict.keys():
            print("NOTE: skip timestamp not in ego timestamps")
            continue
        df.loc[index, _frame] = _frame_dict[_np_int_ts]

    return df


def reference_matching(df_obj_augment, ego_obj_id, columns_tracks, _verbose=False):
    """
    Figure ego and obj rows, compute column attributes on the ground reference frame
    """
    # sort by obj_id, make sure ego always comes first, then sort by ts stable
    df_obj_augment.sort_values(by='obj_id', inplace=True)
    df_obj_augment.sort_values(by='ts', inplace=True, kind='stable')
    df_obj_augment.reset_index(drop=True, inplace=True)

    # set ego vel_lgt_mps, vel_lat_mps
    df_obj_augment.loc[df_obj_augment['obj_id'] == ego_obj_id, 'vel_lgt_mps'] = df_obj_augment['spd_mps']
    df_obj_augment.loc[df_obj_augment['obj_id'] == ego_obj_id, 'vel_lat_mps'] = 0

    # First traverse to calculate the x, y, z, and speed, acceleration respectively.
    updated_col = 'updated'
    df_obj_augment[updated_col] = False 

    curr_obj_idx, curr_ego_idx = None, None 

    # print(df_obj_augment.head)
    if _verbose:
        df_obj_augment.to_csv('./debug/obj_before_first_trav.csv')

    # create all columns_tracks in df_obj_augment if needed
    df_obj_augment = _ensure_columns_exist_robust(df_obj_augment, columns_tracks)

    obj_id_col = 'obj_id'
    for idx in tqdm(range(len(df_obj_augment)), desc="Reference Matching: "): 
        # print("current idx: {}".format(idx))

        # obj
        if int(df_obj_augment.loc[idx, obj_id_col]) != ego_obj_id:
            curr_obj_idx = idx

            # check curr_obj_idx and curr_ego_idx timestamp matched
            assert df_obj_augment.loc[curr_obj_idx, 'ts'] == df_obj_augment.loc[curr_ego_idx, 'ts'], "object {} and ego {} timestamp not match".format(curr_obj_idx, curr_ego_idx)

            # NOTE update according to curr_ego_h
            # NOTE compute all x, y, xVelocity, yVelocity, xAcceleration, yAcceleration from ground-reference-frame
            _x, _y = _rotate_vector(df_obj_augment.loc[idx, 'lgt'], df_obj_augment.loc[idx, 'lat'], curr_ego_h)
            df_obj_augment.loc[curr_obj_idx, 'x'] = df_obj_augment.loc[curr_ego_idx, 'x'] + _x
            df_obj_augment.loc[curr_obj_idx, 'y'] = df_obj_augment.loc[curr_ego_idx, 'y'] + _y
    
            _xVel, _yVel = _rotate_vector(df_obj_augment.loc[idx, 'vel_lgt_mps'], df_obj_augment.loc[idx, 'vel_lat_mps'], curr_ego_h)
            df_obj_augment.loc[curr_obj_idx, 'xVelocity'] =  _xVel
            df_obj_augment.loc[curr_obj_idx, 'yVelocity'] =  _yVel
        
            _xAcc, _yAcc = _rotate_vector(df_obj_augment.loc[idx, 'acc_lgt_mpss'], df_obj_augment.loc[idx, 'acc_lat_mpss'], curr_ego_h)
            df_obj_augment.loc[curr_obj_idx, 'xAcceleration'] =  _xAcc
            df_obj_augment.loc[curr_obj_idx, 'yAcceleration'] =  _yAcc
            
            # flag column updated
            df_obj_augment.loc[curr_obj_idx, updated_col] = True 
    
        # ego
        else: 
            # print("update curr_ego_idx: {}".format(idx))
            curr_ego_idx = idx
            curr_ego_h = df_obj_augment.loc[curr_ego_idx, 'h']

            # update x, y from rear axis middle point to center of the ego vehicle 
            _x, _y = _rotate_vector(df_obj_augment.loc[idx, 'length']/2, 0, curr_ego_h)
            df_obj_augment.loc[curr_ego_idx, 'x'] = df_obj_augment.loc[curr_ego_idx, 'x'] + _x
            df_obj_augment.loc[curr_ego_idx, 'y'] = df_obj_augment.loc[curr_ego_idx, 'y'] + _y
    
            # update ego frame vel_lgt_mps, vel_lat_mps for later computation
            _xVel, _yVel = _rotate_vector(df_obj_augment.loc[curr_ego_idx, 'spd_mps'], 0, curr_ego_h)
            df_obj_augment.loc[curr_ego_idx, 'xVelocity'] = _xVel
            df_obj_augment.loc[curr_ego_idx, 'yVelocity'] = _yVel 
    
            _xAcc, _yAcc = _rotate_vector(df_obj_augment.loc[idx, 'acc_lgt_mpss'], df_obj_augment.loc[idx, 'acc_lat_mpss'], curr_ego_h)
            df_obj_augment.loc[curr_ego_idx, 'xAcceleration'] = _xAcc
            df_obj_augment.loc[curr_ego_idx, 'yAcceleration'] = _yAcc

        # NOTE xCenter, yCenter
        df_obj_augment.loc[idx, 'xCenter'] = df_obj_augment.loc[idx, 'x'] 
        df_obj_augment.loc[idx, 'yCenter'] = df_obj_augment.loc[idx, 'y'] 

        # angle: cannot calculate w/out lane info
        df_obj_augment.loc[idx, 'angle'] = 0

        # orientation
        df_obj_augment.loc[idx, 'orientation'] = df_obj_augment.loc[idx, 'h'] 

        # yaw_rate ~= yAcc / xAcc
        if df_obj_augment.loc[idx, 'xAcceleration'] != 0:
            df_obj_augment.loc[idx, 'yaw_rate'] = df_obj_augment.loc[idx, 'yAcceleration'] / df_obj_augment.loc[idx, 'xAcceleration']
        else:
            df_obj_augment.loc[idx, 'yaw_rate'] = 0

        # ego_offset: cannot calculate w/out lane width info, set to 0
        df_obj_augment.loc[idx, 'ego_offset'] = 0

    if _verbose: 
        df_obj_augment.to_csv('./debug/obj_first_trav.csv')

    return df_obj_augment

def construct_recording(frame_rate):
    _recordings = []
    _recordings.append({
        "recordingId": 0,
        "frameRate": frame_rate,
        "locationId": 0,
        "speedLimit": 0,
        "month": 0,
        "weekDay": 0,
        "startTime":0,
        "duration": 0,
        "totalDrivenDistance": 0,
        "totalDrivenTime": 0,
        "numVehicles": 0,
        "numCars": 0,
        "numTrucks": 0,
        "numBusus": 0,
        "laneMarkings": 0,
        "scale": 0
    })

    return pd.DataFrame(_recordings)
    
def track_data_generator(df_obj_augment, ego_timestamps, cols_tracks, cols_recording_meta, frame_rate, _verbose=False):
    _frame_dict = {timestamp: index for index, timestamp in enumerate(np.sort(ego_timestamps))}
    df_tracks = _create_tracks(df_obj_augment, _frame_dict)

    # Construct 3 output dataframes
    # pd_recording_meta = pd.DataFrame(columns=cols_recording_meta)
    # pd_tracks_meta = pd.DataFrame(columns=cols_tracks_meta)
    pd_tracks = pd.DataFrame(columns=cols_tracks)

    # rename some columns
    df_tracks['laneId'] = df_tracks['lane_id']
    df_tracks['roadId'] = df_tracks['road_id']
    df_tracks['timestamp'] = df_tracks['ts']
    df_tracks['vehicle_id'] = df_tracks['obj_id']
    df_tracks['yaw'] = df_tracks['h']

    # set for 0, not in use
    df_tracks['frontSightDistance'] = 0
    df_tracks['backSightDistance'] = 0

    if _verbose:
        df_tracks.to_csv('my_tracks.csv')

    pd_tracks = df_tracks.loc[:, cols_tracks].copy()
    pd_tracks['class_str'] = df_tracks['class_str'].copy()

    pd_tracks_meta = _calculate_track_statistics(pd_tracks)

    # drop class_str for meta computation purpose
    pd_tracks.drop('class_str', axis=1, inplace=True)
    # drop obj_id which is contained in pd_reacks_meta
    pd_tracks.drop('obj_id', axis=1, inplace=True)

    pd_recording_meta = construct_recording(frame_rate)

    return pd_tracks, pd_tracks_meta, pd_recording_meta