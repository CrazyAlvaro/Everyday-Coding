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

def _normalize_vector(heading_x, heading_y):
    """
    Normalizes a 2D vector.

    Args:
      heading_x: The x-component of the vector.
      heading_y: The y-component of the vector.

    Returns:
      A tuple containing the normalized x and y components of the vector.
    """
    magnitude = np.sqrt(heading_x**2 + heading_y**2)
    # print("heading_x {} heading_y {}".format(heading_x, heading_y))
    if magnitude == 0:
      return 0, 0  # Handle the case where the vector is zero
    else:
      normalized_heading_x = heading_x / magnitude
      normalized_heading_y = heading_y / magnitude
      return normalized_heading_x, normalized_heading_y

def _get_all_directions(heading_x, heading_y):
    """
    Calculates 8 directional vectors given an initial heading vector.

    Args:
      heading_x: The x-component of the initial heading vector.
      heading_y: The y-component of the initial heading vector.

    Returns:
      A list of 8 tuples, each representing a normalized direction vector 
      (x, y) in the order: 
      front, back, left, right, left_front, left_back, right_front, right_back.
    """

    # Normalize the initial heading vector
    heading_x, heading_y = _normalize_vector(heading_x, heading_y)

    # Calculate angles for each direction (in radians)
    angles = np.arange(0, 2*np.pi, np.pi/4)  # 8 directions, 45 degrees each

    # Rotate the initial vector to get all directions
    all_directions = []
    for angle in angles:
      cos_theta = np.cos(angle)
      sin_theta = np.sin(angle)
      new_x = heading_x * cos_theta - heading_y * sin_theta
      new_y = heading_x * sin_theta + heading_y * cos_theta
      all_directions.append((new_x, new_y))

    return all_directions

def _check_vehicle_sides(df, current_index, lookahead_factor=3.0, lookahead_min=15):
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
    lookahead_distance = max(lookahead_distance, lookahead_min)
   
    # Calculate unit vectors for current vehicle's direction

    _directions = _get_all_directions(x_velocity, y_velocity)
   
    # Define unit vectors for all directions
    directions = {
        'front': _directions[0],
        'left_preceding': _directions[1],
        'left_alongside': _directions[2],
        'left_following': _directions[3],
        'back': _directions[4],
        'right_following': _directions[5],
        'right_alongside': _directions[6],
        'right_preceding': _directions[7] 
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

def check_surrounding_objects(df, ego_config):
    """
    Assuming 'df' is your DataFrame with 
    'ts', 'x', 'y', 'orientation', 'obj_id', 'xVelocity', 'yVelocity' columns
    """

    lookahead_factor = ego_config['lookahead']
    lookahead_min = ego_config['lookahead_min']

    for _index in tqdm(df.index, desc="Checking Surroundings: "):
        surrounding_results = _check_vehicle_sides(df, _index, lookahead_factor, lookahead_min) 

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
