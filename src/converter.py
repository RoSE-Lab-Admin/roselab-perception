import sys
import struct
import numpy as np
import open3d as o3d
from pathlib import Path

import rclpy
from rclpy.serialization import deserialize_message
from rclpy.node import Node
from rosbag2_py import SequentialReader, StorageOptions, StorageFilter, ConverterOptions
import cv2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sensor_msgs_py.point_cloud2 as pc2
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage


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
    # Make list of color and depth images
    color_images = []
    depth_images = []
    with AnyReader([Path(bag_path)]) as reader:
        # Make list of color and depth images
        for connection, timestamp, rawdata in reader.messages():
            # Color
            if connection.topic == color_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                img = message_to_cvimage(msg, 'rgb8')
                color_images.append(img)
            # Depth
            if connection.topic == depth_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                img = message_to_cvimage(msg, '16UC1')
                if(img.shape == (360, 640)): depth_images.append(img) # Take only the aligned ones
    #storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
    #converter_options = ConverterOptions('', '')

    # Make stacked and median images
    stacked_color = np.stack(color_images, axis=0)
    median_color_img = np.median(stacked_color, axis=0).astype(np.uint8)
    stacked_depth = np.stack(depth_images, axis=0)
    median_depth_img = np.median(stacked_depth, axis=0).astype(np.uint16)

    # Convert to Open3D Image types
    o3d_color = o3d.geometry.Image(median_color_img.astype(np.uint8))
    o3d_depth = o3d.geometry.Image(median_depth_img.astype(np.uint16))
    #o3d_color = o3d.io.read_image(median_color_img)
    #o3d_depth = o3d.io.read_image(median_depth_img)

    # Create an Open3D RGBDImage
    rgbd_image  = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color,
        o3d_depth,
        depth_scale=1000,      # if depth is already in meters
        depth_trunc=4.0,      # max depth to keep (meters)
        convert_rgb_to_intensity=False
    )

    return rgbd_image

def convert_rgbd_to_pointclouds(rgbd_image):
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(640, 360, 450.6466369628906, 450.8058776855469, 327.085693359375, 177.85765075683594)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    return pcd

def fuse_dynamic_pointclouds(pcds, camera_trajectory):
    return None #pcd

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

def save_pcd_as_file(pcd, output_file):
    if output_file.endswith(".pcd"):
        o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
    elif output_file.endswith(".ply"):
        o3d.io.write_point_cloud(output_file, pcd)
    else:
        raise ValueError("Output file must end in .pcd or .ply")
    
def transform_lidar_to_world(pcd):
    # Transform
    # Lidar
    lidar_pose = np.eye(4)
    lidar_pose[0:3, 0] = np.array([0.80778813 , 0.01618329, 0.58925074 ])
    lidar_pose[0:3, 1] = np.array([-0.58921483, -0.00741073, 0.80794243 ])
    lidar_pose[0:3, 2] = np.array([ 0.01744194, -0.99984158 , 0.00354913  ])
    lidar_pose[0:3, 3] = np.array([0.32536598, 2.34371088, -0.48987012])
    pcd = pcd.transform(lidar_pose)

    return pcd

# RGBD to PointCloud Node
class RGBDPointCloud(Node):
    def __init__(self):
        super().__init__('rgbd_pointcloud_node')
        self.bridge = CvBridge()

        self.rgb_sub = Subscriber(self, Image, '/l515_center/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/l515_center/aligned_depth_to_color/image_raw')
        self.info_sub = Subscriber(self, CameraInfo, '/l515_center/depth/camera_info')

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
    if len(sys.argv) != 5:
        #print("Usage: python ros2_bag_to_pcd.py <bag_path> <pointcloud_topic> <output_file.pcd|.ply>")
        print("Usage: python ros2_bag_to_pcd.py <bag_path> <color_topic> <aligned_depth_topic> <output_file.pcd|.ply>")
        sys.exit(1)

    bag_path = sys.argv[1]
    color_topic = sys.argv[2]
    depth_topic = sys.argv[3]
    output_file = bag_path + sys.argv[4]

    #points, colors = read_pointclouds_from_bag(bag_path, topic)
    rgbd_image = read_rgbd_from_bag(bag_path, depth_topic, color_topic)
    pcd = convert_rgbd_to_pointclouds(rgbd_image)

    pcd = transform_lidar_to_world(pcd)

    print(f"Total points extracted: {len(pcd.points)}")
    #save_to_pcd_or_ply(points, colors, output_file)
    save_pcd_as_file(pcd, output_file)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
