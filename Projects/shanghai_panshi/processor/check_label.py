from tqdm import tqdm
import numpy as np
import pandas as pd


############################################################
## MACRO DEFINITION
############################################################
_VEHICLE_COLUMN = 'obj_id'
_LABEL_RU = ["Car", "SUV", "Truck", "Special_truck"]
_LABEL_VRU = ["tricycle", "bicycle", "motorbike"]


############################################################
## module methods
############################################################
def _is_RU(row):
    return row['class_str'] in set(_LABEL_RU)

def _is_VRU(row):
    return row['class_str'] in set(_LABEL_VRU)

def _is_same_direction(row, vector, error_degree=15):
    """
    Checks if the row's (xVelocity, yVelocity) vector is in the same direction
    as the given ego_vel (xv, yv) vector with an error range.

    Args:
        row: A pandas Series representing a single row of the DataFrame.
        vector: a tuple contain (x,y)
        error_degree: Allowed error in degrees (default: 15 degrees).

    Returns:
        True if the vectors are in the same direction within the error range,
        False otherwise.
    """

    row_vector = np.array([row['xVelocity'], row['yVelocity']])

    # define stop ru on any direction
    if np.all(row_vector == 0):
        return True

    reference_vector = np.array([vector[0], vector[1]])

    # Normalize vectors
    row_vector_normalized = row_vector / np.linalg.norm(row_vector)
    reference_vector_normalized = reference_vector / np.linalg.norm(reference_vector)

    # Calculate dot product
    dot_product = np.dot(row_vector_normalized, reference_vector_normalized)

    # Calculate error_range from error_degree
    error_range = 1 - np.cos(np.radians(error_degree))

    # Check if dot product is within the error range
    return dot_product >= (1 - error_range)

def _is_vector_perpendicular(row, vector, error_degrees=15):
    """
    Checks if the row's (xVelocity, yVelocity) vector is perpendicular
    to the given vector with an error range.

    Args:
        row: A pandas Series representing a single row of the DataFrame.
        vectort: A tuple (xv, yv) representing the reference vector.
        error_degrees: Allowed error in degrees (default: 15 degrees).

    Returns:
        True if the vectors are perpendicular within the error range,
        False otherwise.
    """

    row_vector = np.array([row['xVelocity'], row['yVelocity']])
    reference_vector = np.array(vector)  # Convert tuple to NumPy array

    # Normalize vectors
    row_vector_normalized = row_vector / np.linalg.norm(row_vector)
    reference_vector_normalized = reference_vector / np.linalg.norm(reference_vector)

    # Calculate dot product
    dot_product = np.dot(row_vector_normalized, reference_vector_normalized)

    # Calculate allowed error in dot product for perpendicular vectors
    allowed_error = np.sin(np.radians(error_degrees))  # Perpendicular vectors have a dot product near 0

    # Check if dot product is within the allowed error
    return abs(dot_product) <= allowed_error

def rotate_vector(vector, degrees):
  """
  Rotates a 2D vector by the specified degrees.

  Args:
    vector: A tuple representing the 2D vector (x, y).
    degrees: The angle in degrees to rotate the vector.
             Positive values indicate counter-clockwise rotation,
             negative values indicate clockwise rotation.

  Returns:
    A tuple representing the rotated vector.
  """
  radians = np.radians(degrees)
  rotation_matrix = np.array([[np.cos(radians), -np.sin(radians)],
                             [np.sin(radians),  np.cos(radians)]])
  rotated_vector = np.dot(rotation_matrix, vector)
  return tuple(rotated_vector)

############################################################
## label sorting methods
############################################################
def project_onto_axis_df(df, origin, direction, vehicle_id_col=_VEHICLE_COLUMN):
    """
    Projects data points in a DataFrame onto a one-dimensional axis
    and returns a DataFrame with projection values and vehicle IDs.

    Args:
        df: pandas DataFrame with columns 'x', 'y', and 'vehicle_id'.
        origin: Tuple (x, y) representing the origin of the axis.
        direction: Tuple (xVel, yVel) representing the direction vector of the axis.
        vehicle_id_col: Name of the column containing vehicle IDs.

    Returns:
        pandas DataFrame with columns 'vehicle_id' and 'projection',
        sorted by 'projection' values.
    """

    # Calculate unit vector of the axis direction
    unit_vector = direction / np.linalg.norm(direction)

    # Calculate vector from origin to each data point
    vectors = df[['x', 'y']].values - origin

    # Calculate dot product to get projected distances
    projections = np.dot(vectors, unit_vector)

    # Create DataFrame with vehicle IDs and projections
    projection_df = pd.DataFrame({
        'x': df['x'],
        'y': df['y'],
        vehicle_id_col: df[vehicle_id_col],
        'projection': projections
    })

    # Sort by projection values
    projection_df = projection_df.sort_values(by='projection')

    return projection_df

def find_pos_neg_neighbors(projected_distances):
    """
    Returns:
        Tuple: (pos_index, neg_inex)

    """
    # Find indices of positive and negative projections
    positive_indices = projected_distances[projected_distances > 0].index
    negative_indices = projected_distances[projected_distances < 0].index

    return positive_indices, negative_indices

############################################################
## NIO atomic labeling system checking [1-35]
############################################################
def _check_RU_same_direction(df_timeframe, ego_pos, ego_vel, ego_lane):
    """
    return dict{} represents RU_hor mapping
    """
    _df_ru = df_timeframe[df_timeframe.apply(_is_RU, axis=1)]
    _df_ru_same_dir = _df_ru[_df_ru.apply(lambda row: _is_same_direction(row, ego_vel), axis=1)]

    _RU_hor = {}

    # Left 3rd Lane,
    #   26: forward 1st RU
    _df_ru_left_3 = _df_ru_same_dir[_df_ru_same_dir['lane_id'] == (ego_lane - 3)]
    _df_ru_left_3_sorted = project_onto_axis_df(_df_ru_left_3, ego_pos, ego_vel)
    _df_ru_left_3_fwd = _df_ru_left_3_sorted[_df_ru_left_3_sorted['projection'] >= 0]
    if len(_df_ru_left_3_fwd) > 0:
        _RU_hor[26] = _df_ru_left_3_fwd.iloc[0][_VEHICLE_COLUMN]

    # Left 2nd Lane,
    #   25: forwawrd 1st RU
    _df_ru_left_2 = _df_ru_same_dir[_df_ru_same_dir['lane_id'] == (ego_lane - 2)]
    _df_ru_left_2_sorted = project_onto_axis_df(_df_ru_left_2, ego_pos, ego_vel)
    _df_ru_left_2_fwd = _df_ru_left_2_sorted[_df_ru_left_2_sorted['projection'] >= 0]
    if len(_df_ru_left_2_fwd) > 0:
        _RU_hor[25] = _df_ru_left_2_fwd.iloc[0][_VEHICLE_COLUMN]

    # Left 1st Lane,
    #   3: forward 1st RU
    #   4: forward 2nd RU
    #   11: backward 1st RU
    #   23: backward 2nd RU
    _df_ru_left_1 = _df_ru_same_dir[_df_ru_same_dir['lane_id'] == (ego_lane - 1)]
    _df_ru_left_1_sorted = project_onto_axis_df(_df_ru_left_1, ego_pos, ego_vel)
    _df_ru_left_1_fwd = _df_ru_left_1_sorted[_df_ru_left_1_sorted['projection'] >= 0]
    if len(_df_ru_left_1_fwd) > 0:
        _RU_hor[3] = _df_ru_left_1_fwd.iloc[0][_VEHICLE_COLUMN]
    if len(_df_ru_left_1_fwd) > 1:
        _RU_hor[4] = _df_ru_left_1_fwd.iloc[1][_VEHICLE_COLUMN]

    _df_ru_left_1_bwd = _df_ru_left_1_sorted[_df_ru_left_1_sorted['projection'] < 0]
    if len(_df_ru_left_1_bwd) > 0:
        _RU_hor[11] = _df_ru_left_1_bwd.iloc[-1][_VEHICLE_COLUMN]
    if len(_df_ru_left_1_bwd) > 1:
        _RU_hor[23] = _df_ru_left_1_bwd.iloc[-2][_VEHICLE_COLUMN]

    # Same Lane
    #   1: forward 1st RU
    #   2: forward 2nd RU
    #   9: backward 1st RU
    #   10: backward 2nd RU
    _df_ru_same = _df_ru_same_dir[_df_ru_same_dir['lane_id'] == ego_lane]
    _df_ru_same_sorted = project_onto_axis_df(_df_ru_same, ego_pos, ego_vel)
    _df_ru_same_fwd = _df_ru_same_sorted[_df_ru_same_sorted['projection'] >= 0]
    if len(_df_ru_same_fwd) > 0:
        _RU_hor[1] = _df_ru_same_fwd.iloc[0][_VEHICLE_COLUMN]
    if len(_df_ru_same_fwd) > 1:
        _RU_hor[2] = _df_ru_same_fwd.iloc[1][_VEHICLE_COLUMN]

    _df_ru_same_bwd = _df_ru_same_sorted[_df_ru_same_sorted['projection'] < 0]
    if len(_df_ru_same_bwd) > 0:
        _RU_hor[9] = _df_ru_same_bwd.iloc[-1][_VEHICLE_COLUMN]
    if len(_df_ru_same_bwd) > 1:
        _RU_hor[10] = _df_ru_same_bwd.iloc[-2][_VEHICLE_COLUMN]

    # Right 1st Lane
    #   5: forward 1st RU
    #   6: forward 2nd RU
    #   12: backward 1st RU
    #   24: backward 2nd RU
    _df_ru_right_1 = _df_ru_same_dir[_df_ru_same_dir['lane_id'] == (ego_lane + 1)]
    _df_ru_right_1_sorted = project_onto_axis_df(_df_ru_right_1, ego_pos, ego_vel)
    _df_ru_right_1_fwd = _df_ru_right_1_sorted[_df_ru_right_1_sorted['projection'] >= 0]
    if len(_df_ru_right_1_fwd) > 0:
        _RU_hor[5] = _df_ru_right_1_fwd.iloc[0][_VEHICLE_COLUMN]
    if len(_df_ru_right_1_fwd) > 1:
        _RU_hor[6] = _df_ru_right_1_fwd.iloc[1][_VEHICLE_COLUMN]

    _df_ru_right_1_bwd = _df_ru_right_1_sorted[_df_ru_right_1_sorted['projection'] < 0]
    if len(_df_ru_right_1_bwd) > 0:
        _RU_hor[12] = _df_ru_right_1_bwd.iloc[-1][_VEHICLE_COLUMN]
    if len(_df_ru_right_1_bwd) > 1:
        _RU_hor[24] = _df_ru_right_1_bwd.iloc[-2][_VEHICLE_COLUMN]

    # Right 2nd Lane
    #   26: forwawrd 1st RU
    _df_ru_right_2 = _df_ru_same_dir[_df_ru_same_dir['lane_id'] == (ego_lane + 2)]
    _df_ru_right_2_sorted = project_onto_axis_df(_df_ru_right_2, ego_pos, ego_pos)
    _df_ru_right_2_fwd = _df_ru_right_2_sorted[_df_ru_right_2_sorted['projection'] >= 0]
    if len(_df_ru_right_2_fwd ) > 0:
        _RU_hor[26] = _df_ru_right_2_fwd.iloc[0][_VEHICLE_COLUMN]

    return _RU_hor

def _check_RU_vertical(df_timeframe, ego_pos, ego_vel, ego_lane):
    """
    return dict{} represents RU_ver mapping
    """
    _df_ru = df_timeframe[df_timeframe.apply(_is_RU, axis=1)]

    _RU_ver = {}
    # Left Side Vertical, towards ego
    #   29: forward 1st RU
    #   31: backward 1st RU

    # rotate ego direction to right
    _ego_perpendicular_right = rotate_vector(ego_vel, -90)
    # on the same direction as previous vector

    _df_ru_ppd_right_same = _df_ru[_df_ru.apply(lambda row: _is_same_direction(row, _ego_perpendicular_right), axis=1)]

    # to get ru on the left
    _df_ru_ppd_right_same_sorted = project_onto_axis_df(_df_ru_ppd_right_same, ego_pos, _ego_perpendicular_right)
    _df_ru_ppd_right_same_left = _df_ru_ppd_right_same_sorted[_df_ru_ppd_right_same_sorted['projection'] < 0]

    # to further differentiate between front and back
    _df_ru_ppd_right_same_left_sorted = project_onto_axis_df(_df_ru_ppd_right_same_left, ego_pos, ego_vel)

    _df_ru_ppd_right_same_left_front = _df_ru_ppd_right_same_left_sorted[_df_ru_ppd_right_same_left_sorted['projection'] >= 0]
    if len(_df_ru_ppd_right_same_left_front) > 0:
        _RU_ver[29] = _df_ru_ppd_right_same_left_front.iloc[0][_VEHICLE_COLUMN]

    _df_ru_ppd_right_same_left_back = _df_ru_ppd_right_same_left_sorted[_df_ru_ppd_right_same_left_sorted['projection'] < 0]
    if len(_df_ru_ppd_right_same_left_back) > 0:
        _RU_ver[31] = _df_ru_ppd_right_same_left_back.iloc[-1][_VEHICLE_COLUMN]

    # Right Side Vertical, towards ego
    #   30: forward 1st RU
    #   32: backward 1st RU

    # rotate ego vel vector to left
    _ego_perpendicular_left = rotate_vector(ego_vel, 90)

    # on the same direction
    _df_ru_ppd_left_same = _df_ru[_df_ru.apply(lambda row: _is_same_direction(row, _ego_perpendicular_left), axis=1)]

    # to get ru on the right
    _df_ru_ppd_left_same_sorted = project_onto_axis_df(_df_ru_ppd_left_same, ego_pos, _ego_perpendicular_left)
    _df_ru_ppd_left_same_right = _df_ru_ppd_left_same_sorted[_df_ru_ppd_left_same_sorted['projection'] < 0]

    # further differentiate between front and back
    _df_ru_ppd_left_same_right_sorted = project_onto_axis_df(_df_ru_ppd_left_same_right, ego_pos, ego_vel)

    _df_ru_ppd_left_same_right_front = _df_ru_ppd_left_same_right_sorted[_df_ru_ppd_left_same_right_sorted['projection'] >= 0]
    if len(_df_ru_ppd_left_same_right_front) > 0:
        _RU_ver[30] = _df_ru_ppd_left_same_right_front.iloc[0][_VEHICLE_COLUMN]

    _df_ru_ppd_left_same_right_back = _df_ru_ppd_left_same_right_sorted[_df_ru_ppd_left_same_right_sorted['projection'] < 0]
    if len(_df_ru_ppd_left_same_right_back ) > 0:
        _RU_ver[32] = _df_ru_ppd_left_same_right_back.iloc[-1][_VEHICLE_COLUMN]

    return _RU_ver

def _check_VRU(df_timeframe, ego_pos, ego_vel, ego_lane):
    """
    return dict{} represents VRU mapping
    """
    _df_vru = df_timeframe[df_timeframe.apply(_is_VRU, axis=1)]
    _VRU = {}

    # Left 1st Lane,
    #   34: forward 1st VRU
    #   7: backward 1st VRU
    _df_left_vru = _df_vru[_df_vru['lane_id'] < ego_lane]    # filter left lane
    _df_left_vru_sorted = project_onto_axis_df(_df_left_vru, ego_pos, ego_vel)

    _df_left_vru_fwd = _df_left_vru_sorted[_df_left_vru_sorted['projection'] > 0]
    if len(_df_left_vru_fwd) > 0:
        _VRU[34] = _df_left_vru_fwd.iloc[0][_VEHICLE_COLUMN]

    _df_left_vru_bwd = _df_left_vru_sorted[_df_left_vru_sorted['projection'] <= 0]
    if len(_df_left_vru_bwd) > 0:
        _VRU[7] = _df_left_vru_bwd.iloc[-1][_VEHICLE_COLUMN]

    # Same Lane
    #   33: forward 1st VRU
    _df_same_vru = _df_vru[_df_vru['lane_id'] == ego_lane]    # filter same lane
    _df_same_vru_sorted = project_onto_axis_df(_df_same_vru, ego_pos, ego_vel)

    _df_same_vru_fwd = _df_same_vru_sorted[_df_same_vru_sorted['projection'] >= 0]
    if len(_df_same_vru_fwd) > 0:
        _VRU[33] = _df_same_vru_fwd.iloc[0][_VEHICLE_COLUMN]

    # Right 1st Lane
    #   35: forward 1st VRU
    #   8: backward 1st VRU
    _df_right_vru = _df_vru[_df_vru['lane_id'] > ego_lane]    # filter right lane
    _df_right_vru_sorted = project_onto_axis_df(_df_right_vru, ego_pos, ego_vel)

    _df_right_vru_fwd = _df_right_vru_sorted[_df_right_vru_sorted['projection'] >= 0]
    if len(_df_right_vru_fwd) > 0:
        _VRU[35] = _df_right_vru_fwd.loc[0, _VEHICLE_COLUMN]

    _df_right_vru_bwd = _df_right_vru_sorted[_df_right_vru_sorted['projection'] < 0]
    if len(_df_right_vru_bwd) > 0:
        _VRU[8] = _df_right_vru_bwd.loc[-1, _VEHICLE_COLUMN]

    return _VRU

def _construct_label(df_frame, _index, _RU_label):
    for _key, _val in _RU_label.items():
        assert int(_key) >= 1 and int(_key) <= 35, "[check_label construct_label] Atomic labeling system index range [1-35]"
        df_frame.loc[_index, 'ru'+str(_key)] = _val

def _timeframe_processing(df_ru, _index):
    _curr_row = df_ru.loc[_index]
    _ts_group = df_ru[df_ru['ts'] == _curr_row['ts']]
    _ts_group = _ts_group[_ts_group.index != _index]

    # No other data to check except for RU ego
    if len(_ts_group) == 0:
        return {}

    #######################################################################
    ## NOTE: Assumptions:
    ##      lane_id: identifier sorting from left to right
    #######################################################################
    _ego_pos = (_curr_row['x'], _curr_row['y'])
    _ego_lane = _curr_row['lane_id']

    # set _ego_vel to pos dir if 0 vector
    if _curr_row['xVelocity'] != 0 or _curr_row['yVelocity'] != 0:
        _ego_vel = (_curr_row['xVelocity'], _curr_row['yVelocity'])
    else:
        _ego_vel = (1,0)

    _RU_hor = _check_RU_same_direction(_ts_group, _ego_pos, _ego_vel, _ego_lane)
    _RU_ver = _check_RU_vertical(_ts_group, _ego_pos, _ego_vel, _ego_lane)
    _VRU = _check_VRU(_ts_group, _ego_pos, _ego_vel, _ego_lane)

    return {**_RU_hor, **_RU_ver, **_VRU}

############################################################
## module entry function:
##      processing tracks dataframe to have atomic label system
############################################################
def check_surrounding_label(df_ru):
    """
    return dict{} represents VRU mapping
    """
    # Create new `pandas` methods which use `tqdm` progress
    # tqdm.pandas()
    # df_checked = df_ru.groupby('ts').progress_apply()

    # print('before out check {}'.format(df_ru.loc[0, 'ru1']))
    for _index in tqdm(df_ru.index, desc="Checking Atomic Labels"):
        _labels_dict = _timeframe_processing(df_ru, _index)

        # if len(_labels_dict) > 0:
            # print('Dict not 0: {}'.format(_labels_dict))

        _construct_label(df_ru, _index, _labels_dict)
    # print('after out check {}'.format(df_ru.loc[0, 'ru1']))
    return df_ru