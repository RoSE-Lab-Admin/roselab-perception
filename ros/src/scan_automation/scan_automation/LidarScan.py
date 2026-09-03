import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point

from ament_index_python.packages import get_package_share_directory
from gantry_lidar_interfaces.srv import Capture, DownloadName, DeleteName
from gantry_interfaces.msg import GantryState

import json
from builtin_interfaces.msg import Time
import subprocess
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

        # setting up service clients
        self.gant_capture = self.create_client(Capture, "gantry_capture_service/capture")
        self.gant_download = self.create_client(DownloadName, "gantry_capture_service/download/name")
        self.gant_delete = self.create_client(DeleteName, "gantry_capture_service/delete/name")

        # state variables
        self.tolerance = .02

        # wait for services to be available
        
        self.get_logger().info("Waiting for required services...")
        while not self.gant_capture.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for capture service...")
        while not self.gant_download.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for download service...")
        while not self.gant_delete.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for delete service...")
        self.get_logger().info("All services ready.")

        self.start_lidar()

    def start_lidar(self):
        input("Press enter to start lidar scan")

        capture_request = Capture.Request()
        capture_request.outname = self.panda_file
        capture_request.sensors = ["p_l515_center", "p_l515_west", "p_l515_east"]
        capture_request.duration = float(self.duration) + 10.0

        self.get_logger().info(f"Starting LIDAR capture for {self.duration} seconds.")
        self.future_cap = self.gant_capture.call_async(capture_request)

        self.end_scan()

    def end_scan(self):

        if self.future_cap and not self.future_cap.done():
            self.get_logger().info("Waiting for LIDAR capture to finish...")
            rclpy.spin_until_future_complete(self, self.future_cap)

        cap_response = self.future_cap.result()
        lidar_cap_data = json.loads(cap_response.outdata)

        name_request = DownloadName.Request()
        name_request.name = lidar_cap_data["outname"]
        self.get_logger().info(f"Lidar bag name: {name_request.name}")

        future_name = self.gant_download.call_async(name_request)
        rclpy.spin_until_future_complete(self, future_name)
        name_response = future_name.result()
        name_response_dict = json.loads(name_response.outdata)

        cap_url = name_response_dict["url"]
        self.get_logger().info(f"Downloading from {cap_url} ...")
        subprocess.Popen(["wget", "-r", "-P", f"{self.data_file}", f"{cap_url}"])
        time.sleep(10)

        self.get_logger().info("Download complete.")
        self.get_logger().info(f"Bag saved to {self.data_file}")

        # delete_req = DeleteName.Request()
        # delete_req.name = lidar_cap_data["outname"]
        # future_delete = self.gant_delete.call_async(delete_req)
        # rclpy.spin_until_future_complete(self, future_delete)
        # self.get_logger().info("Deleted bag from LattePanda.")

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




