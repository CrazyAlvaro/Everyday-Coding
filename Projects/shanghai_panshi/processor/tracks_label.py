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
    
class LabelSystem:
    """
    Contain [1-35] labels
    """

    def __init__(self, labels):
        self._label_system = labels

    def to_dict(self):
        return self._label_system



class VehicleFrameLabel:
    """
    Represent a specific time frame for an Vehicle/RU
    """
    def __init__(self, vehicle_id, timestamp, label_system):
        self._vehicle_id = vehicle_id
        self._timestamp = timestamp
        # self._frame = frame
        self._labels = label_system 

    def get_vehicle_id(self):
        return self._vehicle_id

    # def update_label(self, id, label):
        # assert isinstance(id, int), "[VehicleFrame] label id must be an integer"
        assert 1 <= id <= 35, "[VehicleFrame] label id must be between 1 and 35"
        # assert isinstance(label, Label), "[VehicleFrame] label must be a Label type"
        # self._labels[id] = label

    def to_dict(self):
        return {
            'vehicle_id': self._vehicle_id,
            'timestamp': self._timestamp,
            'label_system': self._labels
        }

class VehicleTrack:
    """
    Represent a vehcle track
    """
    def __init__(self, vehicle_id, frames):
        self._vehicle_id = vehicle_id
        self._frames = frames # [VehicleFrameLabel]

    def to_dict(self):
        return {
            'vehicle_id': self._vehicle_id,
            'frames': [_frame.to_dict() for _frame in self._frames]
        }

class TimeframeVehicles:
    def __init__(self, timestamp):
        self._timestamp = timestamp
        self._vehicles = {} 

    def add_vehicle(self, vehicle_id, vehicle_frame_label):
        assert isinstance(vehicle_frame_label, VehicleFrameLabel), "[TimeframeVehicles] vehicle_frame_label must be a VehicleFrameLabel"
        # assert int(vehicle_id) == vehicle_frame_label.get_vehicle_id(), "[TimeframeVehicles] vehicle_id and vehicle_frame_label must for same vehicle"

        self._vehicles[vehicle_id] = vehicle_frame_label

    def get_timestamp(self):
        return self._timestamp

    def to_dict(self):
        return {
            'timestamp': self._timestamp,
            'vehicles': self._vehicles
        }

class FullTimeframeVehicles:
    def __init__(self, start_ts, end_ts, ts_rate):
        self._start_timestamp = start_ts
        self._end_timestamp = end_ts
        self._timestamp_rate = ts_rate
        self._tf_vehicles = []

    def add_timeframe(self, timeframe_vehicles):
        assert isinstance(timeframe_vehicles, TimeframeVehicles), "[FullTimeframeVehicles] timeframe_vehicles must be TimeframeVehicles type"
        curr_timestamp = timeframe_vehicles.get_timestamp()
        # assert self._start_timestamp <= curr_timestamp & self._end_timestamp >= curr_timestamp, "[FullTimeframeVehicles] timestamp not in between"
    
        self._tf_vehicles.append(timeframe_vehicles)

    def to_dict(self):
        return {
            'start_timestamp': self._start_timestamp,
            'end_timestamp': self._end_timestamp,
            'timestamp_rate': self._timestamp_rate,
            'timestamp_vehicles': [_ts_vehicle.to_dict() for _ts_vehicle in self._tf_vehicles]
        }
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
        
        
class TracksByVehicle:
    def __init__(self):
        self._tracks = []

    def add_track(self, track):
        assert isinstance(track, VehicleTrack), "[Tracks] track must be a type of VehicleTrack"
        self._tracks.append(track)

    def to_dict(self):
        return {
            'tracks': [_track.to_dict() for _track in self._tracks]
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)