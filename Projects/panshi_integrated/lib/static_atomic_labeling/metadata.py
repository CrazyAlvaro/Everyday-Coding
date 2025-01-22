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

columns_tracks_label = ['frame', 'trackId', 'timestamp', 'vehicle_id', 'obj_id', 'xCenter', 'yCenter', 'length', 'width',
              'height', 'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration',
              'frontSightDistance', 'backSightDistance',
              'ru1', 'ru2', 'ru3', 'ru4', 'ru5', 'ru6', 'ru7', 'ru8', 'ru9', 'ru10',
              'ru11', 'ru12', 'ru13', 'ru14', 'ru15', 'ru16', 'ru17', 'ru18', 'ru19', 'ru20',
              'ru21', 'ru22', 'ru23', 'ru24', 'ru25', 'ru26', 'ru27', 'ru28', 'ru29', 'ru30',
              'ru31', 'ru32', 'ru33', 'ru34', 'ru35',
              'roadId', 'laneId', 'angle', 'orientation', 'yaw', 'yaw_rate', 'ego_offset']

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