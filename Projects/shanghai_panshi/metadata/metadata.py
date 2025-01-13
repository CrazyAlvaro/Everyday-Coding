###########################################
# Configuation
###########################################

# Assume laneId provided in preceeding data processor

columns_tracks = ['frame', 'trackId', 'timestamp', 'vehicle_id', 'obj_id', 'xCenter', 'yCenter', 'length', 'width',
                'height', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration',
                'frontSightDistance', 'backSightDistance', 'dhw', 'thw', 'ttc',
                'precedingXVelocity', 'precedingId', 'followingId', 'leftPrecedingId',
                'leftAlongsideId', 'leftFollowingId', 'rightPrecedingId',
                'rightAlongsideId', 'rightFollowingId', 'roadId', 'laneId', 'angle',
                'orientation', 'yaw', 'yaw_rate', 'ego_offset']

columns_tracks_meta = ['trackId', 'length', 'width', 'height', 'initialFrame', 'finalFrame', 'numFrames',
       'class', 'drivingDirection', 'traveledDistance', 'minXVelocity',
       'maxXVelocity', 'meanXVelocity', 'minDHW', 'minTHW', 'minTTC',
       'numLaneChanges'] 

columns_recording_meta = ['recordingId', 'frameRate', 'locationId', 'speedLimit', 'month',
       'weekDay', 'startTime', 'duration', 'totalDrivenDistance',
       'totalDrivenTime', 'numVehicles', 'numCars', 'numTrucks', 'numBuses',
       'laneMarkings', 'scale']

time_interpolate_columns = ['x','y','z','h','spd_mps', 'spd_kph', 'acc_lgt_mpss', 'acc_lat_mpss']

shift_interpolate_columns = ['road_id', 'lane_id']

obj_time_cols = ['lgt', 'lat', 'x', 'y', 'z', 'h', 'vel_lgt_mps', 'vel_lat_mps', 
                             'speed_direction','acc_lgt_mpss', 'acc_lat_mpss', 'acc_direction']

obj_shift_cols = ['obj_id', 'class', 'class_str', 'length', 'width', 'height', 'road_id', 'lane_id']