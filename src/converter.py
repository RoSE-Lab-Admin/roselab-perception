import sys
import struct
import numpy as np
import open3d as o3d

import rclpy
from rclpy.serialization import deserialize_message
from rclpy.node import Node
from rosbag2_py import SequentialReader, StorageOptions, StorageFilter, ConverterOptions
import cv2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
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

# Assumes already aligned, so don't need to use camera intrinsics
def read_rgbd_from_bag(bag_path, depth_topic, color_topic):
    # TODO - Need to support reading in Image msgs from bag for both depth and color topics, sync time, and generate point clouds using Open3D
    storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = ConverterOptions('', '')

    return rgbd_images

def convert_rgbd_to_pointclouds(rgbd_images):
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

# RGBD to PointCloud Node
class RGBDPointCloud(Node):
    def __init__(self):
        super().__init__('rgbd_pointcloud_node')
        self.bridge = CvBridge()

        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')
        self.info_sub = Subscriber(self, CameraInfo, '/camera/depth/camera_info')

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.pub = self.create_publisher(PointCloud2, '/fused_point_cloud', 10)
        self.get_logger().info('RGBD Point Cloud Node Initialized')

    def callback(self, rgb_msg, depth_msg, info_msg, agg=np.mean):
        color = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        fx = info_msg.k[0]
        fy = info_msg.k[4]
        cx = info_msg.k[2]
        cy = info_msg.k[5]

        height, width = depth.shape
        points = []

        # Conversion factor for L515 is 1 bit = 0.00025 m
        convert_to_m_factor = 0.00025

        for v in range(height):
            for u in range(width):
                z = depth[v, u] * convert_to_m_factor
                if z == 0 or np.isnan(z): continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                b, g, r = color[v, u]
                rgb = int(r << 16 | g << 8 | b)
                points.append([x, y, z, rgb])

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]

        header = rgb_msg.header
        pc = pc2.create_cloud(header, fields, points)
        self.pub.publish(pc)

#def main(args=None):
#    rclpy.init(args=args)
#    node = RGBDPointCloud()
#    rclpy.spin(node)
#    node.destroy_node()
#    rclpy.shutdown()

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
