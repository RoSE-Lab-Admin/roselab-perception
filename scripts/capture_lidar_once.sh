#!/usr/bin/env bash
# configurable delay after LiDAR start

# LIDAR_SECS={$1:-30} # Change depending on trial scan length

echo "Perception Ground Control Started!"

# 1. start LiDAR
read -r -p "TURN OFF MOCAP CAMERAS Press Enter to START LiDAR (/start_lidar)..."
echo "Starting LiDAR..."
ros2 topic pub --once /start_lidar std_msgs/msg/Bool "{data: true}"

