###########################################
# Configuation
###########################################

# Assume laneId provided in preceeding data processor
lane_id_col = 'lane_id'

columns_tracks = ['frame', 'trackId', 'xCenter', 'yCenter', 'length', 'width',
                'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration',
                'frontSightDistance', 'backSightDistance', 'dhw', 'thw', 'ttc',
                'precedingXVelocity', 'precedingId', 'followingId', 'leftPrecedingId',
                'leftAlongsideId', 'leftFollowingId', 'rightPrecedingId',
                'rightAlongsideId', 'rightFollowingId', 'laneId', 'angle',
                'orientation', 'yaw_rate', 'ego_offset']

columns_tracks_meta = ['trackId', 'length', 'width', 'initialFrame', 'finalFrame', 'numFrames',
       'class', 'drivingDirection', 'traveledDistance', 'minXVelocity',
       'maxXVelocity', 'meanXVelocity', 'minDHW', 'minTHW', 'minTTC',
       'numLaneChanges'] 

columns_recording_meta = ['recordingId', 'frameRate', 'locationId', 'speedLimit', 'month',
       'weekDay', 'startTime', 'duration', 'totalDrivenDistance',
       'totalDrivenTime', 'numVehicles', 'numCars', 'numTrucks', 'numBuses',
       'laneMarkings', 'scale']

time_interpolate_columns = ['x','y','z','h','spd_mps', 'spd_kph', 'acc_lgt_mpss', 'acc_lat_mpss']

shift_interpolate_columns = ['lane_id']