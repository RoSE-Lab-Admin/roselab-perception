#!/bin/bash

# This script is for use with Ansible or users to launch the D456 Realsense MastCam
# CLI for ease of use in adjusting runtime data exposed via ROS2

set -e

sudo sysctl -w net.core.wmem_max=1073741824 #changes write UDP buffer to 512mb
sudo sysctl -w net.core.rmem_max=1073741824 #changes read UDP buffer to 512mb

ENABLE_IMU=false
RESOLUTION=1280x720x30   # Default: max resolution @ 30 fps
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
        --align)
            USE_ALIGNED_DEPTH=true
            shift
            ;;
        *)
            # THIS COLLECTS ALL EXTRA TOPICS SPECIFIED AFTER FLAGS
            ;;
    esac
done

# Start realsense2_camera with desired settings
echo "Starting Realsense D456 node..."
source /home/dev/roselab-perception/ros/install/setup.sh
ros2 launch realsense2_camera rs_launch.py rgb_camera.color_profile:=$RESOLUTION depth_module.depth_profile:=$RESOLUTION color_qos:=DEFAULT depth_qos:=DEFAULT enable_sync:=false enable_gyro:=$ENABLE_IMU enable_accel:=$ENABLE_IMU align_depth.enable:=$USE_ALIGNED_DEPTH camera_name:=Front camera_namespace:=/MastCam/ & REALSENSE_PID=$!

# wait a bit for init
sleep 5

# Trap CTRL+C and cleanup
trap "echo 'Stopping...'; kill $REALSENSE_PID; wait" SIGINT

# Keep script alive until stopped
wait
