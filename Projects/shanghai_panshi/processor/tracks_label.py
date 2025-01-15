


def process_vehicles_by_frame(df):
    """
    Processes vehicles in each frame of the DataFrame.

    Args:
        df: pandas DataFrame with columns:
            - 'frame'
            - 'trackId'
            - 'timestamp'
            - 'vehicle_id'
            - 'class_str'
            - 'xCenter'
            - 'yCenter'
            - 'length'
            - 'width'
            - 'height'
            - 'xVelocity'
            - 'yVelocity'
            - 'roadId'
            - 'laneId'
            - 'angle'

    Returns:
        DataFrame with potentially added columns or modifications based on 
        the 'run_function1' and 'run_function2' results.
    """


    # TODO: 
    # TODO instantiate an object, that contains the result, frame[timestamp] - vehicle[RU] - labels[1-35]


    for frame, frame_df in df.groupby('frame'):
        unique_vehicle_ids = frame_df['vehicle_id'].unique()
        for vehicle_id in unique_vehicle_ids:
            vehicle_data = frame_df[frame_df['vehicle_id'] == vehicle_id]

            # check for RU
            result1 = check_RU_same_direction(frame_df, vehicle_id)  
            result2 = check_RU_verticle(frame_df, vehicle_id)  

            # check for VRU 
            result3 = check_VRU(frame_df, vehicle_id)  

            # for current vehicle_id
            # create a new [label_class] object to store [1-35] labels

            # TODO: finally, an object that contains all timestamp frame, '
            #       for each frame, contains a list of vehicle_id,
            #       for each vehicle_id, contains an object contains [1-35] labels
            

    return df  # Return the potentially modified DataFrame

# Example usage (assuming you have defined run_function1 and run_function2)
processed_df = process_vehicles_by_frame(your_dataframe)