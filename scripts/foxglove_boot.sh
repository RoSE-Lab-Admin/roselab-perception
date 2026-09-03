#sudo sysctl -w net.core.rmem_max=1073741824 #changes read UDP buffer to 128mb
source /opt/ros/jazzy/setup.bash
source ~/CubeRover/install/setup.bash
source gantry_lidars/install/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765
