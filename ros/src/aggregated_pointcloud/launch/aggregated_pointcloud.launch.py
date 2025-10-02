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
                "pose_topic": "/CubeRover_V1/pose",
                "voxel_size": 0.01,
            }]
        ),

        Node(
            package="depth_image_proc",
            executable="register_node",
            name="depth_register",
            remappings=[
                ("rgb/camera_info", "/MastCam/Front/color/camera_info"),
                ("depth/camera_info", "/MastCam/Front/depth/camera_info"),
                ("depth/image_rect", "/MastCam/Front/depth/image_rect_raw"),
                ("depth_registered/image_rect", "/MastCam/Front/depth/image_rect_registered"),
            ],
            parameters=[{
                "approx_sync": True,
                "queue_size": 10,
            }]
        ),

        Node(
            package="depth_image_proc",
            executable="point_cloud_xyzrgb_node",
            name="point_cloud_xyzrgb",
            remappings=[
                ("rgb/image_rect_color", "/MastCam/Front/color/image_raw"),
                ("rgb/camera_info", "/MastCam/Front/color/camera_info"),
                ("depth_registered/image_rect", "/MastCam/Front/depth/image_rect_registered"),
                ("points", "/MastCam/Front/points"),
            ],
            parameters=[{
                "approx_sync": True,
                "queue_size": 10,
            }]
        ),
    ])
