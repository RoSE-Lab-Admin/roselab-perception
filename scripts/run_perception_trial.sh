#!/bin/bash

# This script should be used for single trial execution when using the mast cam payload on RoSEy
#   Capabilities:
#       - Easily launch and bag all relevant topics for perception testing
#       - Toggle realsense IMU, depth alignment, and color/depth resolution and fps
#       - Generate custom tag for bag, which is automatically suffixed with timestamp
#
#   TODO
#       - LEXI       :  Add your scanning routine to beginning and end of bagging! We want a before and after of each trial
#       - RYAN + CAM :  Make sure all relevant transforms and topics are captured! I think I have the big ones, but we want any and all rover controls and output data
#       - ALL        :  Experiment with the best configurations for data throughput. I'd prefer 15 fps if it means we can get the largest res imagery. IMU data is probably not realistic to collect but thats fine. Lastly, I dont know if alignment is needed since thatll slow data pipeline down...
#
#   Use (default plz):
#
#       run_perception_trial.sh --res=1240x720x15 --no-imu --tag dryrun_featureless_

set -e

ENABLE_IMU=false
RESOLUTION="1280x720x15"   # Default: max resolution @ 30 fps
TAG=""                     # Optional prefix for bag name
USE_ALIGNED_DEPTH=false    # Default: raw depth

# Parse flags
for arg in "$@"; do
    case $arg in
        --use-imu)
            ENABLE_IMU=true
            shift
            ;;
        --res=*)
            RESOLUTION="${arg#*=}"
            shift
            ;;
        --tag=*)
            TAG="${arg#*=}_"
            shift
            ;;
        --align)
            USE_ALIGNED_DEPTH=true
            shift
            ;;
        *)
            # THIS COLLECTS ALL EXTRA TOPICS SPECIFIED AFTER FLAGS
            ;;
    esac
done

# Bag output directory (ROSE_DATA should already exist, or will be created)
mkdir -p ~/ROSE_DATA

BAG_DIR=~/ROSE_DATA/${TAG}$(date +%Y%m%d_%H%M%S)

# Start realsense2_camera with desired settings
echo "Starting Realsense D456 node..."
ros2 launch realsense2_camera rs_launch.py \
    rgb_camera.profile:=$RESOLUTION \
    depth_module.profile:=$RESOLUTION \
    enable_gyro:=$ENABLE_IMU \
    enable_accel:=$ENABLE_IMU \
    align_depth.enable:=$USE_ALIGNED_DEPTH &
REALSENSE_PID=$!

# wait a bit for init
sleep 5

# Set up D456 topics
CAM_TOPICS="/MastCam/Front/color/image_raw
            /MastCam/Front/color/camera_info
            /MastCam/Front/extrinsics/depth_to_color" # Check the format on this extrinsics topic plz

if [ "$USE_ALIGNED_DEPTH" = true ]; then
    CAM_TOPICS="$CAM_TOPICS
                /MastCam/Front/aligned_depth_to_color/image_raw
                /MastCam/Front/aligned_depth_to_color/camera_info"
else
    CAM_TOPICS="$CAM_TOPICS
                /MastCam/Front/depth/image_rect_raw
                /MastCam/Front/depth/camera_info"
fi

# Add D456 IMU topics if enabled
if [ "$ENABLE_IMU" = true ]; then
    CAM_TOPICS="$CAM_TOPICS
                /MastCam/Front/gyro/sample
                /MastCam/Front/accel/sample"
fi

# Find all mocap topics ending in /pose
POSE_TOPICS=$(ros2 topic list | grep '/pose$' | tr '\n' ' ')


# RYAN + CAM : PUT ALL ROVER CONTROL AND TELEMETRY TOPICS HERE
ROVER_TOPICS="/odom
              /tf
              /tf_static"

# Merge topics
ALL_TOPICS="$CAM_TOPICS $POSE_TOPICS $ROVER_TOPICS"

echo "Recording rosbag to $BAG_DIR ..."
echo "Topics: $ALL_TOPICS"

ros2 bag record -o $BAG_DIR $ALL_TOPICS &
BAG_PID=$!

# RYAN + CAM : PUBLISH STATIC TFs FOR WHATEVER WE NEED TO SHOW POINT CLOUDS IN WORLD (world -> rover_body -> mast_cam -> data)
# be sure to publish after recording starts?

# Trap CTRL+C and cleanup
trap "echo 'Stopping...'; kill $REALSENSE_PID $BAG_PID; wait" SIGINT

# Keep script alive until stopped
wait
