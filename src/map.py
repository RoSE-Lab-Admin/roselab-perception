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

def read_sync_align_rgbds_from_bag(bag_path, depth_topic, color_topic, camera_info_topic):
    # Make list of color and depth images
    rgbd_images = []

    # Open reader, sync topics, pack into RGBD images
    with AnyReader([Path(bag_path)]) as reader:
        # Make list of color and depth images
        for connection, timestamp, rawdata in reader.messages():
            # Color
            if connection.topic == color_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                img = message_to_cvimage(msg, 'rgb8')
            # Depth
            if connection.topic == depth_topic:
                msg = reader.deserialize(rawdata, connection.msgtype)
                img = message_to_cvimage(msg, '16UC1')

            # Convert to Open3D Image types
            o3d_color = o3d.geometry.Image(median_color_img.astype(np.uint8))
            o3d_depth = o3d.geometry.Image(median_depth_img.astype(np.uint16))

            # Create an Open3D RGBDImage
            rgbd_image  = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color,
                o3d_depth,
                depth_scale=1000,      # if depth is already in meters
                depth_trunc=5.0,      # max depth to keep (meters), making this long for d456
                convert_rgb_to_intensity=False
            )

        # Return a list of RGBD images, or yield?
        return rgbd_images

# This should be generalized and moved into an imaging class or something
def convert_rgbd_to_pointclouds(rgbd_image):
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    # THIS MUST BE CHANGED TO USE CAMERA_INFO TOPIC FOR INTRINSICS!!!!!!!
    intrinsics.set_intrinsics(640, 360, 450.6466369628906, 450.8058776855469, 327.085693359375, 177.85765075683594)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    return pcd

def fuse_dynamic_pointclouds(pcds, camera_trajectory):
    return None #pcd

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

if __name__=="__main__":
    # Load Optitrack poses from bag

    from sync import BagSynchronizer
    import sys

    bag_file = sys.argv[1]
    color_topic = sys.argv[2]
    depth_topic = sys.argv[3]
    cam_info_topic = sys.argv[4]
    extrinsics_topic = sys.argv[5]
    pose_topic = sys.argv[6] # This should be either 

    synchronizer = BagSynchronizer(bag_file, color_topic, depth_topic, cam_info_topic)
    synchronizer.synchronize()

    # 
