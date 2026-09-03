import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from mastcam_interfaces.srv import Capture, DeleteName, DownloadName

from pathlib import Path
from datetime import datetime
import json
import subprocess
import time

class MastcamClient(Node):
    def __init__(self):
        super().__init__('mastcam_client')

        # output folder on slade - d drive perception data/day/hour
        data_file = "/mnt/d/perception-data"
        data_base = Path(data_file).expanduser().resolve()
        self.day = datetime.now().strftime("%m%d%Y")
        hour = datetime.now().strftime("%H-%M-%S")
        self.data_file = Path(data_base) / self.day / hour
        self.data_file.mkdir(parents=True, exist_ok=True)

        # output folder on pi 
        self.pi_file = "mastcam_bags"

        # service clients
        self.download = self.create_client(DownloadName, "mastcam_capture_service/download/name")
        self.delete = self.create_client(DeleteName, "mastcam_capture_service/delete/name")
        self.stop = self.create_client(Trigger, "mastcam_capture_service/stop")
        self.start = self.create_client(Capture, "mastcam_capture_service/start")

        # wait for services
        self.get_logger().info("Waiting for services...")
        while not self.download.wait_for_service(timeout_sec=1.0):
            pass
        while not self.delete.wait_for_service(timeout_sec=1.0):
            pass
        while not self.stop.wait_for_service(timeout_sec=1.0):
            pass
        while not self.start.wait_for_service(timeout_sec=1.0):
            pass
        self.get_logger().info("Services ready")


        self.start_capture()

    def start_capture(self):
        input("Press enter to start capture")

        capture_request = Capture.Request()
        capture_request.outname = self.pi_file
        capture_request.duration = 60.0 # dummy val
        self.cap_future = self.start.call_async(capture_request)

        self.get_logger().info("Mastcam capture started")
        self.stop_capture()

    def stop_capture(self):
        input("Press enter to start capture")

        stop_request = Trigger.Request()
        stop_future = self.stop.call_async(stop_request)
        rclpy.spin_until_future_complete(self, stop_future)

        self.get_logger().info("Bagging stopped")
        
        # process capture json result with outname
        self.cap_rep = self.cap_future.result()
        cap_json = json.loads(self.cap_rep.outdata)

        # call download name service
        download_request = DownloadName.Request()
        download_request.name = cap_json["outname"]
        self.get_logger().info(f"Bag name: {download_request.name}")
        download_future = self.download.call_async(download_request)
        rclpy.spin_until_future_complete(self, download_future)
        download_name = download_future.result()

        # download from https
        name_json = json.loads(download_name.outdata)
        cap_url = name_json["url"]
        self.get_logger().info(f"Downloading from {cap_url}")
        subprocess.Popen(["wget", "-r", "-P", f"{self.data_file}", f"{cap_url}"])
        time.sleep(10)

        self.get_logger().info(f"Bag saved to {self.data_file}")

        # delete from pi
        delete_req = DeleteName.Request()
        delete_req.name = cap_json["outname"]
        delete_future = self.delete.call_async(delete_req)
        rclpy.spin_until_future_complete(self, delete_future)
        self.get_logger().info("Deleted bag from pi")

        input("Press enter to start capture")
        self.start_capture()

def main(args=None):
    rclpy.init(args=args)
    node = MastcamClient()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()






        
 