#!/usr/bin/env bash
# configurable delay after LiDAR start

LIDAR_SECS=60

echo "Perception Ground Control Started!"

while true; do
    # 1. start LiDAR
    read -r -p "Press Enter to START LiDAR (/start_lidar)..." _
    echo "Starting LiDAR..."
    ros2 topic pub --once /start_lidar std_msgs/msgs/Bool "{data: true}"

    # 2. spinup / fixed wait
    echo "Waiting ${LIDAR_SECS}s for LiDAR to finish..."
    sleep "${LIDAR_SECS}"

    # 3. start MastCam + Rosey bag
    read -r -p "Press Enter to START MastCam (/start_mastcam) AND RoSEy bag (/start_rosey_bag)..." _
    echo "Starting MastCam..."
    ros2 topic pub --once /start_mastcam std_msgs/msgs/Bool "{data: true}"
    echo "Starting Rosey bag..."
    ros2 topic pub --once /start_rosey_bag std_msgs/msgs/Bool "{data: true}"

    # 4. stop both
    read -r -p "Press Enter to STOP MastCam (/stop_mastcam) AND Rosey bag (/stop_rosey_bag)..." _
    echo "Stopping MastCam..."
    ros2 topic pub --once /stop_mastcam std_msgs/msgs/Bool "{data: true}"
    echo "Stopping Rosey bag..."
    ros2 topic pub --once /stop_rosey_bag std_msgs/msgs/Bool "{data: true}"

    echo
    echo "Cycle complete. Ctrl+C to exit, or it will start over."
    echo
done
