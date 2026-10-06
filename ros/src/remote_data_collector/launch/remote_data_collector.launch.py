"""
Launch one remote_data_collector.

    ros2 launch remote_data_collector remote_data_collector.launch.py name:=mastcam_front
    ros2 launch remote_data_collector remote_data_collector.launch.py \
        name:=my_collector config:=/path/to/my_collector.yaml

`config` is either a path to a parameter file or the name of a file shipped in this
package's config/ directory (without .yaml). It defaults to `name`, so a collector whose name
matches a shipped config (mastcam_front, gantry, ground_station, rosey_avionics, wheelcam) just
needs `name:=...`. The node name determines the endpoints: /<name>/record, /<name>/status, ...
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resolve_config(value: str) -> str:
    if os.path.sep in value or value.endswith('.yaml'):
        path = os.path.expanduser(value)
    else:
        path = os.path.join(get_package_share_directory('remote_data_collector'), 'config',
                            f'{value}.yaml')
    if not os.path.isfile(path):
        raise RuntimeError(f'remote_data_collector: config file not found: {path}')
    return path


def _launch(context):
    name = LaunchConfiguration('name').perform(context)
    config = LaunchConfiguration('config').perform(context) or name
    params = [_resolve_config(config)]
    overrides = {}
    arg_to_param = {'data_dir': ('data_dir', str), 'http_port': ('http.port', int),
                    'advertise_host': ('http.advertise_host', str)}
    for arg, (param, cast) in arg_to_param.items():
        value = LaunchConfiguration(arg).perform(context)
        if value:
            overrides[param] = cast(value)
    if overrides:
        params.append(overrides)
    return [Node(
        package='remote_data_collector',
        executable='remote_data_collector',
        name=name,
        namespace=LaunchConfiguration('namespace').perform(context),
        parameters=params,
        output='screen',
        emulate_tty=True,
        # The node finalizes recordings on SIGINT; give it time before escalating.
        sigterm_timeout='420',
        sigkill_timeout='60',
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('name', default_value='remote_data_collector',
                              description='Node name; endpoints are /<name>/record etc.'),
        DeclareLaunchArgument('config', default_value='',
                              description='Parameter file path or shipped config name '
                                          '(default: same as name)'),
        DeclareLaunchArgument('namespace', default_value='', description='Optional namespace'),
        DeclareLaunchArgument('data_dir', default_value='', description='Override data_dir'),
        DeclareLaunchArgument('http_port', default_value='', description='Override http.port'),
        DeclareLaunchArgument('advertise_host', default_value='',
                              description='Override http.advertise_host (IP in download URLs)'),
        OpaqueFunction(function=_launch),
    ])
