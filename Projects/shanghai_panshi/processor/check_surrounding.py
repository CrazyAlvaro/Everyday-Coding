import numpy as np
from tqdm import tqdm

def _angle_between_vectors(v1, v2):
    """
    Calculates the angle between two vectors in degrees.

    Args:
      v1: The first vector as a NumPy array.
      v2: The second vector as a NumPy array.

    Returns:
      The angle between the vectors in degrees.
    """
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def _check_vehicle_sides(df, current_index, lookahead_factor=3.0):
    """
    Checks for vehicles in 8 directions (front, back, left_following, 
    left_preceding, left_alongside, right_preceding, right_following, right_alongside) 
    based on ts, x, y, orientation, xVelocity, and yVelocity data.
   
    Args:
      df: pandas DataFrame containing 'ts', 'x', 'y', 'orientation', 'obj_id', 'xVelocity', 'yVelocity' columns.
      current_index: Index of the current vehicle in the DataFrame.
      lookahead_factor: Factor to adjust lookahead distance based on speed.
   
    Returns:
      A dictionary with keys for each direction, 
      each containing the obj_id of the nearest vehicle in that direction, 
      or 0 if no vehicle is found.
    """
    current_row = df.iloc[current_index]
   
    curr_obj_id       = current_row['obj_id']
    x_velocity        = current_row['xVelocity']
    y_velocity        = current_row['yVelocity']
    current_x         = current_row['x']
    current_y         = current_row['y']
    current_timestamp = current_row['ts']
   
    # Calculate lookahead distance based on speed
    current_speed     = np.sqrt(x_velocity**2 + y_velocity**2)
    lookahead_distance = current_speed * lookahead_factor 
   
    # Calculate unit vectors for current vehicle's direction
    heading_x = np.cos(np.radians(current_row['orientation']))
    heading_y = np.sin(np.radians(current_row['orientation']))
   
    # Define unit vectors for all directions
    directions = {
        'front': np.array([heading_x, heading_y]),
        'back': -np.array([heading_x, heading_y]),
        'left_alongside': np.array([-heading_y, heading_x]), 
        'left_preceding': np.array([heading_x-heading_y, heading_x+heading_y]),   # 'front' + 'left_alongside'
        'left_following': np.array([-heading_x-heading_y, heading_x-heading_y]),  # 'back'  + 'left_alongside'
        'right_alongside': np.array([heading_y, -heading_x]),
        'right_preceding': np.array([heading_x+heading_y, -heading_x+heading_y]), # 'front' + 'right_alongside'
        'right_following': np.array([-heading_x+heading_y, -heading_x-heading_y]) # 'back'  + 'right_alongside'
    }
   
    # Initialize results
    vehicle_ids = {
        'front':            (0, float('inf'), 0),  # need precedingXVelocity
        'back':             (0, float('inf')),
        'left_preceding':   (0, float('inf')),
        'left_alongside':   (0, float('inf')),
        'left_following':   (0, float('inf')),
        'right_preceding':  (0, float('inf')),
        'right_alongside':  (0, float('inf')),
        'right_following':  (0, float('inf'))
    }
   
    # Filter DataFrame to consider only vehicles within a reasonable time window
    # (e.g., vehicles within the last few seconds)
    # time_window = 0.1**6  # total time_window would be 0.1*2 = 0.2 second, original unit: microsecond
    # filtered_df = df[(df['ts'] >= current_timestamp - time_window) & (df['ts'] <= current_timestamp + time_window)] 

    # Filter DataFrame to consider matching timestamp
    filtered_df = df[df['ts'] == current_timestamp] 
   
    for index, row in filtered_df.iterrows():
        if index == current_index:
          continue  # Skip the current vehicle
    
        if row['obj_id'] == curr_obj_id:
          continue  # Skip check with itself on other timestamp
    
        # Calculate distance and direction vectors
        distance_vector = np.array([row['x'] - current_x, row['y'] - current_y])
        distance = np.linalg.norm(distance_vector)
   
        # Check if within lookahead distance
        if distance > lookahead_distance:
          continue
    
        # Normalize direction vector
        direction_vector = distance_vector / distance
   
        for direction, direction_unit_vector in directions.items():
            # loop through all directions
   
            # dot_product = np.dot(direction_vector, direction_unit_vector)
   
            if abs(_angle_between_vectors(direction_vector, direction_unit_vector)) > 22.5:
                # not on the same direction
                continue

            if not vehicle_ids[direction][0] or distance < vehicle_ids[direction][1]:
                # not exist or closer than current obj_id, then udpate
   
                # front contains xVelocity
                if direction == 'front':
                    vehicle_ids[direction] = (row['obj_id'], distance, row['xVelocity'])
                else:
                    vehicle_ids[direction] = (row['obj_id'], distance)
   
    return vehicle_ids

def check_surrounding_objects(df):
    """
    Assuming 'df' is your DataFrame with 
    'ts', 'x', 'y', 'orientation', 'obj_id', 'xVelocity', 'yVelocity' columns
    """
    for _index in tqdm(df.index, desc="Checking Surroundings: "):
        surrounding_results = _check_vehicle_sides(df, _index) 

        # update precedingId, dhw, precedingXVelocity
        df.loc[_index, 'precedingId'] = surrounding_results['front'][0]

        if df.loc[_index, 'precedingId'] != 0:
            df.loc[_index, 'dhw'] = surrounding_results['front'][1]
        else:
            # no precedingId exist
            df.loc[_index, 'dhw'] = 0
        df.loc[_index, 'precedingXVelocity'] = surrounding_results['front'][2]

        # update surrounding values
        df.loc[_index, 'followingId']      =  surrounding_results['back'][0]
        df.loc[_index, 'leftPrecedingId']  =  surrounding_results['left_preceding'][0]
        df.loc[_index, 'leftAlongsideId']  =  surrounding_results['left_alongside'][0]
        df.loc[_index, 'leftFollowingId']  =  surrounding_results['left_following'][0]
        df.loc[_index, 'rightPrecedingId'] =  surrounding_results['right_preceding'][0]
        df.loc[_index, 'rightAlongsideId'] =  surrounding_results['right_alongside'][0]
        df.loc[_index, 'rightFollowingId'] =  surrounding_results['right_following'][0]
    
    return df
