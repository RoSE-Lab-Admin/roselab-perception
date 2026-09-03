from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    #launch arguments
    slade_root_arg = DeclareLaunchArgument(
        "slade_root",
        default_value="/mnt/d/perception-data",
        description="Where bagged data is stored on slade"
    )

    panda_file_arg = DeclareLaunchArgument(
        "panda_file",
        default_value="lidar_bags", #adds date and time later
        description="Where bags are stored on the LattePanda"
    )

    pi_file_arg = DeclareLaunchArgument(
        "pi_file",
        default_value="mastcam_bags", 
        description="Where bags are stored on the pi"
    )

    duration_arg = DeclareLaunchArgument(
        "duration",
        default_value='10.0', 
        description="how long to run the lidar scan"
    )


    #launch node
    node = Node(
        package="perception_ground_control",
        executable="ground_control",
        name="ground_control",
        parameters=[{
            "slade_root": LaunchConfiguration("slade_root"),
            "pi_file": LaunchConfiguration("pi_file"),
            "panda_file": LaunchConfiguration("panda_file"),
            "duration":LaunchConfiguration("duration")
        }]
    )

    return LaunchDescription([
        slade_root_arg,
        pi_file_arg,
        duration_arg,
        panda_file_arg,
        node
    ])