import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

class SessionRunner(Node):
    def __init__(self):
        super().__init__('perception_cli')

        # command line publishers
        self.pub_start_lidar = self.create_publisher(Bool, '/start_lidar', self.start_lidar, 10)
        # call from cli: ros2 topic pub --once /start_lidar std_msgs/msg/Bool "{data: true}"
        self.pub_start_mastcam = self.create_publisher(Bool, '/start_mastcam', self.start_mast_start, 10)
        # call from cli: ros2 topic pub --once /start_mastcam std_msgs/msg/Bool "{data: true}"
        self.pub_stop_mastcam = self.create_publisher(Bool, '/stop_mastcam', self.start_mast_stop, 10)
        # call from cli: ros2 topic pub --once /stop_mastcam std_msgs/msg/Bool "{data: true}"
        self.pub_start_rosey_bag = self.create_publisher(Bool, '/start_rosey_bag', self.start_rosey_bag, 10)
        # call from cli: ros2 topic pub --once /start_rosey_bag std_msgs/msg/Bool "{data: true}"
        self.pub_stop_rosey_bag = self.create_publisher(Bool, '/stop_rosey_bag', self.stop_rosey_bag, 10)
        # call from cli: ros2 topic pub --once /stop_rosey_bag std_msgs/msg/Bool "{data: true}"

        self.msg = Bool()
        self.msg.data = True

        # Gather session parameters from the launched ground control node
        # Need directory we're saving to for metadata.text file

        self.get_logger().info('Perception CLI Node has been started.')

    # Template for publishing Bool messages
    def run_session(self):
        raw = input("")
        self.pub_start_lidar.publish(self.msg)
        self.get_logger().info('Published start_lidar command.')

        raw = input("")
        self.pub_start_mastcam.publish(self.msg)
        self.get_logger().info('Published start_mastcam command.')
        self.pub_start_rosey_bag.publish(self.msg)
        self.get_logger().info('Published start_rosey_bag command.')

        raw = input("")
        self.pub_stop_mastcam.publish(self.msg)
        self.get_logger().info('Published stop_mastcam command.')
        self.pub_stop_rosey_bag.publish(self.msg)
        self.get_logger().info('Published stop_rosey_bag command.')


if __name__ == '__main__':
    rclpy.init()
    perception_cli = SessionRunner()
    rclpy.spin(perception_cli)
    perception_cli.destroy_node()
    rclpy.shutdown()