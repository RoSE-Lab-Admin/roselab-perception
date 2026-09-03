#!/usr/bin/env bash
# configurable delay after LiDAR start

# 3. start MastCam + Rosey bag
read -r -p "TURN ON MOCAP CAMERAS Press Enter to START MastCam (/start_mastcam) AND RoSEy bag (/start_rosey_bag)..."
echo "Starting MastCam..."
ros2 topic pub --once /start_mastcam std_msgs/msg/Bool "{data: true}"

# 4. stop both
read -r -p "Press Enter to STOP MastCam (/stop_mastcam) AND Rosey bag (/stop_rosey_bag)..."
echo "Stopping MastCam..."
ros2 topic pub --once /stop_mastcam std_msgs/msg/Bool "{data: true}"

