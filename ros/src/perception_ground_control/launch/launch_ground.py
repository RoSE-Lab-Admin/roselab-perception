"""
Launch perception ground control (the client of all remote_data_collectors).

    ros2 launch perception_ground_control launch_ground.py
    ros2 launch perception_ground_control launch_ground.py duration:=60.0 \
        slade_root:=/mnt/d/perception-data
    ros2 launch perception_ground_control launch_ground.py config:=/path/to/my_fleet.yaml

The legacy arguments are kept so existing commands (and the ansible playbook) still work:
  slade_root -> data_root, duration -> remotes.gantry.duration (lidar scan seconds),
  pi_file -> remotes.mastcam_front.bag_name, panda_file -> remotes.gantry.bag_name.
Empty arguments leave the value from the config file untouched.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch(context):
    config = LaunchConfiguration('config').perform(context) or os.path.join(
        get_package_share_directory('perception_ground_control'), 'config', 'ground_control.yaml')
    overrides = {}
    mapping = {
        'slade_root': ('data_root', str),
        'duration': ('remotes.gantry.duration', float),
        'pi_file': ('remotes.mastcam_front.bag_name', str),
        'panda_file': ('remotes.gantry.bag_name', str),
        'default_group': ('default_group', str),
    }
    for arg, (param, cast) in mapping.items():
        value = LaunchConfiguration(arg).perform(context)
        if value:
            overrides[param] = cast(value)
    return [Node(
        package='perception_ground_control',
        executable='ground_control',
        name='ground_control',
        parameters=[os.path.expanduser(config)] + ([overrides] if overrides else []),
        output='screen',
        emulate_tty=True,
    )]


def generate_launch_description():
    args = [
        DeclareLaunchArgument('config', default_value='',
                              description='Fleet config (default: share/.../ground_control.yaml)'),
        DeclareLaunchArgument('slade_root', default_value='',
                              description='Local data root (overrides data_root)'),
        DeclareLaunchArgument('duration', default_value='',
                              description='Lidar scan seconds (remotes.gantry.duration)'),
        DeclareLaunchArgument('pi_file', default_value='',
                              description='Mastcam bag prefix (remotes.mastcam_front.bag_name)'),
        DeclareLaunchArgument('panda_file', default_value='',
                              description='Lidar bag prefix (remotes.gantry.bag_name)'),
        DeclareLaunchArgument('default_group', default_value='',
                              description='Targets used when start is called without any'),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_launch)])
