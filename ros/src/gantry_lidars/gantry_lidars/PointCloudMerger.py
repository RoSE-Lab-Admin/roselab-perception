import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_pc2

from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.time import Time

class PointCloudMerger(Node):
    def __init__(self):
        super().__init__('pointcloud_merger')

        # Parameters
        self.declare_parameter('target_frame', 'base_link')
        self.target_frame = self.get_parameter('target_frame').value

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers to 3 cloud topics
        self.pc1_sub = Subscriber(self, PointCloud2, '/l515_center/depth/color/points')
        self.pc2_sub = Subscriber(self, PointCloud2, '/l515_west/depth/color/points')
        self.pc3_sub = Subscriber(self, PointCloud2, '/l515_east/depth/color/points')

        self.sync = ApproximateTimeSynchronizer(
            [self.pc1_sub, self.pc2_sub, self.pc3_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.clouds_callback)

        self.publisher = self.create_publisher(PointCloud2, '/cloud_merged', 10)
        self.get_logger().info("PointCloudMerger initialized")

    def transform_cloud(self, cloud):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                cloud.header.frame_id,
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            return tf2_pc2.do_transform_cloud(cloud, transform)
        except Exception as e:
            self.get_logger().warn(f"Failed to transform cloud from {cloud.header.frame_id} to {self.target_frame}: {e}")
            return None

    def clouds_callback(self, cloud1, cloud2, cloud3):
        try:
            # Transform clouds to target frame
            c1 = self.transform_cloud(cloud1)
            c2 = self.transform_cloud(cloud2)
            c3 = self.transform_cloud(cloud3)

            if None in (c1, c2, c3):
                self.get_logger().warn("Skipping merge: one or more clouds could not be transformed")
                return

            # Merge all points
            merged_points = []
            for cloud in (c1, c2, c3):
                merged_points.extend(pc2.read_points(cloud, skip_nans=True))

            merged = pc2.create_cloud(
                header=c1.header,  # All transformed to same frame
                fields=c1.fields,
                points=merged_points
            )

            self.publisher.publish(merged)

        except Exception as e:
            self.get_logger().warn(f"Point Cloud Merge failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudMerger()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
