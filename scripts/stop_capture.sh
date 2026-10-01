#!/usr/bin/env bash
# configurable delay after LiDAR start

# E-stop recording
echo "Stopping MastCam and Rosey Bagging..."
ros2 topic pub --once /stop_mastcam std_msgs/msg/Bool "{data: true}"

