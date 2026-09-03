sudo sysctl -w net.core.rmem_max=134217728 #changes read UDP buffer to 128mb 
source /opt/ros/jazzy/setup.bash
ros2 daemon start
sleep 3s
ros2 param set /Rover/camera FrameDurationLimits [16666,16666]
ros2 param set /Rover/camera LensPosition 7.0
