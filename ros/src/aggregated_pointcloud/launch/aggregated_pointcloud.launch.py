from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="aggregated_pointcloud",
            executable="aggregated_pointcloud_node",
            name="aggregated_pointcloud",
            output="screen",
            parameters=[{
                "cloud_topic": "/MastCam/Front/points",
                "pose_topic": "/CubeRoverV1/pose",
                "voxel_size": 0.01,
            }]
        )
    ])
