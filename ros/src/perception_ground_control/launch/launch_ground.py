from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

from perception_ground_control.rosey_bag_defaults import (
    DEFAULT_ROSEY_TOPICS,
    DEFAULT_ROSEY_NAMESPACES,
)

def generate_launch_description():

    #launch arguments
    slade_root_arg = DeclareLaunchArgument(
        "slade_root",
        default_value="/mnt/e/perception-data",
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

    # roseybag topic selection. Each accepts a comma/space-separated string,
    # e.g. rosey_extra_topics:="/foo,/bar", or a YAML list, e.g. "['/foo', '/bar']"
    rosey_topics_arg = DeclareLaunchArgument(
        "rosey_topics",
        default_value=",".join(DEFAULT_ROSEY_TOPICS),
        description="Master list of explicit topics to record in the roseybag (replaces the defaults)"
    )

    rosey_namespaces_arg = DeclareLaunchArgument(
        "rosey_namespaces",
        default_value=",".join(DEFAULT_ROSEY_NAMESPACES),
        description="Namespaces whose topics are auto-discovered and added to the roseybag (replaces the defaults)"
    )

    rosey_extra_topics_arg = DeclareLaunchArgument(
        "rosey_extra_topics",
        default_value="",
        description="Additional explicit topics appended to rosey_topics"
    )

    rosey_extra_namespaces_arg = DeclareLaunchArgument(
        "rosey_extra_namespaces",
        default_value="",
        description="Additional namespaces appended to rosey_namespaces for topic auto-discovery"
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
            "duration":LaunchConfiguration("duration"),
            "rosey_topics": LaunchConfiguration("rosey_topics"),
            "rosey_namespaces": LaunchConfiguration("rosey_namespaces"),
            "rosey_extra_topics": LaunchConfiguration("rosey_extra_topics"),
            "rosey_extra_namespaces": LaunchConfiguration("rosey_extra_namespaces"),
        }]
    )

    return LaunchDescription([
        slade_root_arg,
        pi_file_arg,
        duration_arg,
        panda_file_arg,
        rosey_topics_arg,
        rosey_namespaces_arg,
        rosey_extra_topics_arg,
        rosey_extra_namespaces_arg,
        node
    ])
