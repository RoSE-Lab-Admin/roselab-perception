#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import pyrealsense2 as rs

class MastCamPointCloudNode(Node):
    def __init__(self):
        super().__init__('mastcam_pointcloud')

        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 30)

        w = self.get_parameter('width').value
        h = self.get_parameter('height').value
        fps = self.get_parameter('fps').value

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.pipeline.start(cfg)

        self.align_to_color = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()

        self.publisher = self.create_publisher(PointCloud2, 'MastCam/pointcloud', 10)
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

        self.get_logger().info(f"Started MastCam PointCloud node at {w}x{h}@{fps}Hz")

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align_to_color.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)

        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex_coords = np.asanyarray(points.get_texture_coordinates()).reshape(-1, 2)

        color_image = np.asanyarray(color_frame.get_data())
        u = (tex_coords[:, 0] * color_image.shape[1]).astype(np.int32)
        v = (tex_coords[:, 1] * color_image.shape[0]).astype(np.int32)

        valid = (u >= 0) & (u < color_image.shape[1]) & (v >= 0) & (v < color_image.shape[0])
        u[~valid] = 0
        v[~valid] = 0

        colors = color_image[v, u, :]
        r = colors[:, 2].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 0].astype(np.uint32)
        rgb = (r << 16) | (g << 8) | b
        rgb = rgb.view(np.float32)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera_color_optical_frame"

        pc2_msg = PointCloud2(
            header=header,
            height=1,
            width=verts.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=[
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ],
            point_step=16,
            row_step=16 * verts.shape[0],
            data=np.column_stack((verts, rgb)).astype(np.float32).tobytes()
        )
        self.publisher.publish(pc2_msg)

    def destroy_node(self):
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = D456PointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
