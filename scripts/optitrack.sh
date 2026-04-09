source /opt/ros/jazzy/setup.bash
source natnet_ws/install/setup.bash
ros2 daemon start
ros2 launch natnet_ros2 natnet_ros2.launch.py
