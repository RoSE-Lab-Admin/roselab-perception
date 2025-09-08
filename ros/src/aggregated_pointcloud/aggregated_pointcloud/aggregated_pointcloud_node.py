#! $VIRTUAL_ENV/bin/python

import rclpy
from rclpy.node import Node

import numpy as np
import open3d as o3d

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import sensor_msgs_py.point_cloud2 as pc2

from scipy.spatial.transform import Rotation as R


def pose_to_matrix(pose: PoseStamped) -> np.ndarray:
    """Convert a PoseStamped into a 4x4 homogeneous transform."""
    q = pose.pose.orientation
    t = pose.pose.position
    quat = [q.x, q.y, q.z, q.w]
    trans = [t.x, t.y, t.z]

    R_mat = R.from_quat(quat).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = trans
    return T


class AggregatedPointCloud(Node):
    def __init__(self):
        super().__init__("aggregated_pointcloud_node")

        # Parameters
        self.declare_parameter("cloud_topic", "/camera/points")
        self.declare_parameter("pose_topic", "/camera/pose")
        self.declare_parameter("voxel_size", 0.05)  # 5cm

        cloud_topic = self.get_parameter("cloud_topic").value
        pose_topic = self.get_parameter("pose_topic").value

        self.latest_pose = None
        self.aggregated_cloud = o3d.geometry.PointCloud()

        # Subscribers
        self.create_subscription(PointCloud2, cloud_topic, self.cloud_callback, 10)
        self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 10)

        # Timer for publishing / saving
        # self.timer = self.create_timer(2.0, self.publish_downsampled) # just publish on destruction, uses more memory but whatever

        self.get_logger().info("AggregatedPointCloud node started.")

    def destroy_node(self):
        """Flush final cloud on shutdown."""
        self.get_logger().info("Shutting down — saving final aggregated cloud...")
        self.publish_downsampled()
        super().destroy_node()

    def pose_callback(self, msg: PoseStamped):
        self.latest_pose = msg

    def cloud_callback(self, msg: PointCloud2):
        if self.latest_pose is None:
            return

        points, colors = self._ros2_to_numpy(msg)

        if points.shape[0] == 0:
            return

        # Transform to world frame using latest pose
        T = pose_to_matrix(self.latest_pose)
        pts_h = np.hstack((points, np.ones((points.shape[0], 1))))
        pts_world = (T @ pts_h.T).T[:, :3]

        # Add to Open3D cloud
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(pts_world)
        if colors is not None:
            o3d_cloud.colors = o3d.utility.Vector3dVector(colors)

        self.aggregated_cloud += o3d_cloud

    def publish_downsampled(self):
        if len(self.aggregated_cloud.points) == 0:
            self.get_logger().info("No points in pointcloud... exiting.")
            return

        voxel_size = self.get_parameter("voxel_size").value
        down = self.aggregated_cloud.voxel_down_sample(voxel_size)

        self.get_logger().info(
            f"Aggregated cloud has {len(self.aggregated_cloud.points)} pts "
            f"-> downsampled to {len(down.points)} pts."
        )

        # Save to disk (PLY file with colors if present)
        o3d.io.write_point_cloud("~/aggregated_cloud.ply", down)

        # Option: Publish back to ROS2 topic
        # cloud_msg = self.o3d_to_ros2(down)
        # self.publisher.publish(cloud_msg)

    def _ros2_to_numpy(self, msg: PointCloud2):
        """Extract XYZ and RGB from PointCloud2."""
        points = []
        colors = []

        has_color = any(f.name in ["rgb", "rgba"] for f in msg.fields)

        for p in pc2.read_points(msg, skip_nans=True):
            x, y, z = p[0:3]
            points.append([x, y, z])

            if has_color:
                rgb_val = p[3]
                # rgb is packed float32; unpack into [r,g,b]
                if isinstance(rgb_val, float):
                    rgb_int = int(np.frombuffer(np.float32(rgb_val).tobytes(), dtype=np.uint32)[0])
                else:
                    rgb_int = int(rgb_val)

                r = (rgb_int >> 16) & 255
                g = (rgb_int >> 8) & 255
                b = rgb_int & 255
                colors.append([r / 255.0, g / 255.0, b / 255.0])

        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32) if has_color else None

        return points, colors

    def o3d_to_ros2(self, cloud: o3d.geometry.PointCloud) -> PointCloud2:
        pts = np.asarray(cloud.points)
        cols = np.asarray(cloud.colors) if cloud.has_colors() else None

        if cols is not None and pts.shape[0] == cols.shape[0]:
            data = []
            for i in range(len(pts)):
                x, y, z = pts[i]
                r, g, b = (cols[i] * 255).astype(np.uint8)
                rgb = (r << 16) | (g << 8) | b
                data.append([x, y, z, rgb])
            fields = [
                pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name="rgb", offset=12, datatype=pc2.PointField.UINT32, count=1),
            ]
        else:
            data = pts.tolist()
            fields = [
                pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
                pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
            ]

        header = self.get_clock().now().to_msg()
        return pc2.create_cloud(header, fields, data)


def main(args=None):
    rclpy.init(args=args)
    node = AggregatedPointCloud()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl-C received, shutting down...")
    finally:
        # We want to write things out before shutting down
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
