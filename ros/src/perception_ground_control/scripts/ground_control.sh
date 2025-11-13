#!/usr/bin/env bash
# configurable delay after LiDAR start

LIDAR_SECS=30 # Change depending on trial scan length

echo "Perception Ground Control Started!"

# 1. start LiDAR
read -r -p "Press Enter to START LiDAR (/start_lidar)..."
echo "Starting LiDAR..."
ros2 topic pub --once /start_lidar std_msgs/msg/Bool "{data: true}"

# 2. spinup / fixed wait
echo "Waiting ${LIDAR_SECS}s for LiDAR to finish..."
sleep "${LIDAR_SECS}"

while true; do
    # 3. start MastCam + Rosey bag
    read -r -p "Press Enter to START MastCam (/start_mastcam) AND RoSEy bag (/start_rosey_bag)..."
    echo "Starting MastCam..."
    ros2 topic pub --once /start_mastcam std_msgs/msg/Bool "{data: true}"

    # 4. stop both
    read -r -p "Press Enter to STOP MastCam (/stop_mastcam) AND Rosey bag (/stop_rosey_bag)..."
    echo "Stopping MastCam..."
    ros2 topic pub --once /stop_mastcam std_msgs/msg/Bool "{data: true}"

    # 5. start LiDAR
    read -r -p "Press Enter to START LiDAR (/start_lidar)..."
    echo "Starting LiDAR..."
    ros2 topic pub --once /start_lidar std_msgs/msg/Bool "{data: true}"

    # 6. spinup / fixed wait
    echo "Waiting ${LIDAR_SECS}s for LiDAR to finish..."
    sleep "${LIDAR_SECS}"

    echo
    echo "Cycle complete. Ctrl+C to exit, or it will start over. Waiting 10 seconds for input..."
    sleep 10
    echo

done