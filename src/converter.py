import sys
import struct
import numpy as np
import open3d as o3d

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, StorageFilter, ConverterOptions
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


def extract_rgb_float(rgb_float):
    """Unpack float32 RGB to normalized r, g, b"""
    packed = struct.pack('f', rgb_float)
    i = struct.unpack('I', packed)[0]
    r = (i >> 16) & 0xFF
    g = (i >> 8) & 0xFF
    b = i & 0xFF
    return [r / 255.0, g / 255.0, b / 255.0]


def read_pointclouds_from_bag(bag_path, topic_name):
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    reader.set_filter(StorageFilter(**{'topics': [topic_name]}))

    points = []
    colors = []

    print(f"Reading point clouds from: {bag_path}, topic: {topic_name}")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg = deserialize_message(data, PointCloud2)

        for pt in pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb = pt
            points.append([x, y, z])
            colors.append(extract_rgb_float(rgb))

    return np.array(points), np.array(colors)

def read_rgbd_from_bag(bag_path, depth_topic, color_topic):
    # TODO - Need to support reading in Image msgs from bag for both depth and color topics, sync time, and generate point clouds using Open3D
    return depth_images, color_images

def convert_rgbd_to_pointclouds(depth, color):
    return pcds

def fuse_dynamic_pointclouds(pcds, camera_trajectory):
    return pcd

def save_to_pcd_or_ply(points_np, colors_np, output_file):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    if colors_np.size > 0:
        pcd.colors = o3d.utility.Vector3dVector(colors_np)

    if output_file.endswith(".pcd"):
        o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    elif output_file.endswith(".ply"):
        o3d.io.write_point_cloud(output_file, pcd)
    else:
        raise ValueError("Output file must end in .pcd or .ply")


def main():
    if len(sys.argv) != 4:
        print("Usage: python ros2_bag_to_pcd.py <bag_path> <pointcloud_topic> <output_file.pcd|.ply>")
        sys.exit(1)

    bag_path = sys.argv[1]
    topic = sys.argv[2]
    output_file = sys.argv[3]

    points, colors = read_pointclouds_from_bag(bag_path, topic)
    print(f"Total points extracted: {len(points)}")
    save_to_pcd_or_ply(points, colors, output_file)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
