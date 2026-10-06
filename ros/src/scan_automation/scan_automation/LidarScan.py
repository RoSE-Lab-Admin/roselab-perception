import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point

from ament_index_python.packages import get_package_share_directory
from scan_automation.lidar_capture import LidarCapture
from gantry_interfaces.msg import GantryState

from builtin_interfaces.msg import Time
from pathlib import Path as pth
import time
from datetime import datetime
import os


class LidarScan(Node):
    def __init__(self):
        super().__init__('lidar_scan')

        # slade output folder
        self.declare_parameter("data_file")
        data_base = pth(self.get_parameter("data_file").value).expanduser().resolve()
        self.day = datetime.now().strftime("%m%d%Y")
        hour = datetime.now().strftime("%H-%M-%S")
        self.data_file = pth(data_base) / self.day / hour
        self.data_file.mkdir(parents=True, exist_ok=True)

        # lattepanda folder
        self.declare_parameter("panda_file", "lidar_bags")
        self.panda_file = self.get_parameter("panda_file").value

        # duration paramater
        self.declare_parameter("duration", 60.0)
        self.duration = self.get_parameter("duration").value

        # lidar recording on the gantry's remote_data_collector (replaces gantry_capture_service)
        self.declare_parameter("collector", "/gantry")
        self.declare_parameter("lidar_profiles",
                               ["gantry_state", "p_l515_center", "p_l515_west", "p_l515_east"])
        self.lidar = LidarCapture(remote=self.get_parameter("collector").value,
                                  bag_name=self.panda_file,
                                  profiles=self.get_parameter("lidar_profiles").value,
                                  logger=self.get_logger())

        # state variables
        self.tolerance = .02

        # wait for services to be available
        
        self.get_logger().info("Waiting for the gantry data collector...")
        self.lidar.wait_until_available()
        self.get_logger().info("Data collector ready.")

        self.start_lidar()

    def start_lidar(self):
        input("Press enter to start lidar scan")

        self.get_logger().info(f"Starting LIDAR capture for {self.duration} seconds.")
        self.lidar.start(float(self.duration) + 10.0)

        self.end_scan()

    def end_scan(self):
        # wait for the recording to finish and download it (kept on the gantry, as before)
        self.lidar.finish(self.data_file, delete=False)
        self.get_logger().info("Download complete.")

        input("Press enter to start lidar scan")
        
        self.start_lidar()


def main(args=None):
    rclpy.init(args=args)
    node = LidarScan()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




