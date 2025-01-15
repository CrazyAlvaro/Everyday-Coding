import json

class Label:
    """
    Represent an atomic label
    """

    def __init__(self, vehicle_id, type_str):
        self._veihicle_id = vehicle_id
        self._type = type_str

    def to_dict(self):
        return {
            'vehicle_id': self._veihicle_id,
            'type': self._type
        }


class VehicleFrame:
    """
    Represent a specific time frame for an Vehicle/RU
    """
    def __init__(self, vehicle_id, timestamp, frame):
        self._vehicle_id = vehicle_id
        self._timestamp = timestamp
        self._frame = frame
        self._labels = {}

    def update_label(self, id, label):
        assert isinstance(id, int), "label id must be an integer"
        assert 1 <= id <= 35, "label id must be between 1 and 35"
        assert isinstance(label, Label), "label must be a Label type"

        self._labels[id] = label

    def to_dict(self):
        return {
            'vehicle_id': self._vehicle_id,
            'timestamp': self._timestamp,
            'frame': self._frame,
            'labels': self._labels
        }

class VehicleTrack:
    """
    Represent a vehcle track
    """
    def __init__(self, vehicle_id, frames):
        self._vehicle_id = vehicle_id
        self._frames = frames

    def to_dict(self):
        return {
            'vehicle_id': self._vehicle_id,
            'frames': [_frame.to_dict() for _frame in self._frames]
        }

class Tracks:
    def __init__(self):
        self._tracks = []

    def add_track(self, track):
        assert isinstance(track, VehicleTrack), "track must be a type of VehicleTrack"
        self._tracks.append(track)

    def to_dict(self):
        return {
            'tracks': [_track.to_dict() for _track in self._tracks]
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)









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