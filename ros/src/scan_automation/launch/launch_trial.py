from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

import os

def generate_launch_description():
    #config file path
    config_file = os.path.join(
        get_package_share_directory("scan_automation"),
        "config",
        "lidar_param.yaml"
    )

    #launch arguments
    data_file_arg = DeclareLaunchArgument(
        "data_file",
        default_value="D:/perception_data",
        description="Where bagged data is stored on slade"
    )

    trajectory_file_arg = DeclareLaunchArgument(
        "trajectory_file",
        default_value="src/scan_automation/scan_automation/path_files/RH_test.yaml",
        description="Trajectory file to be used for the gantry"
    )

    panda_file_arg = DeclareLaunchArgument(
        "panda_file",
        default_value="lidar_bags/", #adds date and time later
        description="Where bags are stored on the LattePanda"
    )

    #launch node
    node = Node(
        package="scan_automation",
        executable="gantry_command",
        name="gantry_command",
        parameters=[config_file, {
            "data_file": LaunchConfiguration("data_file"),
            "trajectory_file": LaunchConfiguration("trajectory_file"),
            "panda_file": LaunchConfiguration("panda_file"),
        }]
    )

    return LaunchDescription([
        data_file_arg,
        trajectory_file_arg,
        panda_file_arg,
        node
    ])
