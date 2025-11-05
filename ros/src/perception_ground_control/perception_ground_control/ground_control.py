import rclpy
from rclpy.node import Node

from gantry_lidar_interfaces.srv import (
    Capture as LidarCapture, 
    DownloadName as LidarDownloadName, 
    DeleteName as LidarDeleteName
    )
from mastcam_interfaces.srv import (
    Capture as MastCapture,
    DeleteName as MastDeleteName,
    DownloadName as MastDownloadName
    )
from std_srvs.srv import Trigger
from std_msgs.msg import Bool

from pathlib import Path
from datetime import datetime
import subprocess
import json
import time
import signal

'''
Process flow:
- take lidar scan
- mastcam + bagging other needed topics
- another lidar scan at end of session
'''


class groundcontrol(Node):
    def __init__(self):
        super().__init__('ground_control')

        # parameters
        # output file for slade slade
        self.declare_parameter('slade_root', "/mnt/d/perception-data")
        slade_root = Path(self.get_parameter('slade_root').value).expanduser().resolve()
        self.day = datetime.now().strftime("%m%d%Y")
        hour = datetime.now().strftime("%H-%M-%S")
        self.data_file = Path(slade_root) / self.day / hour
        self.data_file.mkdir(parents=True, exist_ok=True)
        # output file for lattepanda
        self.declare_parameter('panda_file', "lidar_bags")
        self.panda_file = self.get_parameter("panda_file").value
        # output file for pi
        self.declare_parameter('pi_file', "mastcam_bags")
        self.pi_file = self.get_parameter("pi_file").value
        # duration of lidar scan
        self.declare_parameter('duration', 60.0)
        self.duration = self.get_parameter('duration').value

        # setting up services
        # lidar services
        self.lidar_capture = self.create_client(LidarCapture, "gantry_capture_service/capture")
        self.lidar_download = self.create_client(LidarDownloadName, "gantry_capture_service/download/name")
        self.lidar_delete = self.create_client(LidarDeleteName, "gantry_capture_service/delete/name")
        # mastcam services
        self.mast_download = self.create_client(MastDownloadName, "mastcam_capture_service/download/name")
        self.mast_delete = self.create_client(MastDeleteName, "mastcam_capture_service/delete/name")
        self.mast_stop = self.create_client(Trigger, "mastcam_capture_service/stop")
        self.mast_start = self.create_client(MastCapture, "mastcam_capture_service/start")

        # command line subscriptions
        self.create_subscription(Bool, '/start_lidar', self.start_lidar, 10)
        # call from cli: ros2 topic pub --once /start_lidar std_msgs/msgs/Bool "{data: true}"
        self.create_subscription(Bool, '/start_mastcam', self.start_mast, 10)
        # call from cli: ros2 topic pub --once /start_mastcam std_msgs/msgs/Bool "{data: true}"
        self.create_subscription(Bool, '/stop_mastcam', self.stop_mast, 10)
        # call from cli: ros2 topic pub --once /stop_mastcam std_msgs/msgs/Bool "{data: true}"
        self.create_subscription(Bool, '/start_rosey_bag', self.start_rosey_bags, 10)
        # call from cli: ros2 topic pub --once /start_rosey_bag std_msgs/msgs/Bool "{data: true}"
        self.create_subscription(Bool, '/stop_rosey_bag', self.stop_rosey_bags, 10)
        # call from cli: ros2 topic pub --once /stop_rosey_bag std_msgs/msgs/Bool "{data: true}"

        # wait for services
        self.get_logger().info("Waiting for services...")
        while not self.mast_download.wait_for_service(timeout_sec=1.0):
            pass
        while not self.mast_delete.wait_for_service(timeout_sec=1.0):
            pass
        while not self.mast_stop.wait_for_service(timeout_sec=1.0):
            pass
        while not self.mast_start.wait_for_service(timeout_sec=1.0):
            pass
        while not self.lidar_capture.wait_for_service(timeout_sec=1.0):
            pass
        while not self.lidar_download.wait_for_service(timeout_sec=1.0):
            pass
        while not self.lidar_delete.wait_for_service(timeout_sec=1.0):
            pass
        self.get_logger().info("Services ready")

    # do first scan for lidar
    def start_lidar(self, msg: Bool):

        lidar_request = LidarCapture.Request()
        lidar_request.outname = self.panda_file
        lidar_request.sensors = ["p_l515_center", "p_l515_west", "p_l515_east"]
        lidar_request.duration = float(self.duration) 

        self.get_logger().info(f"Starting LIDAR capture")
        self.future_lidar = self.lidar_capture.call_async(lidar_request)
        rclpy.spin_until_future_complete(self, self.future_lidar)

        self.end_lidar()


    def end_lidar(self, msg: Bool):
        self.get_logger().info("Stopping LIDAR capture")
        # get outname from service repsonse
        lidar_response = self.future_lidar.result()
        lidar_outname = json.loads(lidar_response.outdata)

        # request https of bag
        name_request = LidarDownloadName.Request()
        name_request.name = lidar_outname["outname"]
        self.get_logger().info(f"Lidar bag name: {name_request.name}")
        # wait until return
        future_name = self.lidar_download.call_async(name_request)
        rclpy.spin_until_future_complete(self, future_name)
        name_response = future_name.result()
        name_response_dict = json.loads(name_response.outdata)
        cap_url = name_response_dict["url"]
        self.get_logger().info(f"Downloading from {cap_url} ...")
        subprocess.Popen(["wget", "-r", "-P", f"{self.data_file}", f"{cap_url}"])
        time.sleep(10)

        self.get_logger().info("Download complete.")
        self.get_logger().info(f"Bag saved to {self.data_file}")

        # delete_req = LidarDeleteName.Request()
        # delete_req.name = lidar_cap_data["outname"]
        # future_delete = self.gant_delete.call_async(delete_req)
        # rclpy.spin_until_future_complete(self, future_delete)
        # self.get_logger().info("Deleted bag from LattePanda.")

        # if session is over, terminate, else start mast cam

    def start_mast(self, msg: Bool):

        capture_request = MastCapture.Request()
        capture_request.outname = self.pi_file
        capture_request.duration = 60.0 # dummy val
        self.cap_future = self.mast_start.call_async(capture_request)

        self.get_logger().info("Mastcam capture started")


    def stop_mast(self, msg: Bool):

        stop_request = Trigger.Request()
        stop_future = self.mast_stop.call_async(stop_request)
        rclpy.spin_until_future_complete(self, stop_future)

        self.get_logger().info("Bagging stopped")
        
        # process capture json result with outname
        self.cap_rep = self.cap_future.result()
        cap_json = json.loads(self.cap_rep.outdata)

        # call download name service
        download_request = MastDownloadName.Request()
        download_request.name = cap_json["outname"]
        self.get_logger().info(f"Bag name: {download_request.name}")
        download_future = self.mast_download.call_async(download_request)
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
        delete_req = MastDeleteName.Request()
        delete_req.name = cap_json["outname"]
        delete_future = self.mast_delete.call_async(delete_req)
        rclpy.spin_until_future_complete(self, delete_future)
        self.get_logger().info("Deleted bag from pi")

    def start_rosey_bags(self, msg: Bool):
            # Format filename
            self.filename = f"RoseyBag"

            # Set Topics
            topics = ["/CubeRover_V1/pose", "/cmd_vel", 
                      "/dynamic_joint_states", "/initialpose",
                      "/joint_states", "/joy", "/robot_description",
                      "/rosout", "/tf", "tf_static",
                      "/Rover/camera/image_raw/compressed"]
            topics.append(rclpy.get_published_topics(namespace='/bno055/'))
            topics.append(rclpy.get_published_topics(namespace='/joy/'))
            topics.append(rclpy.get_published_topics(namespace='/roseybot_base_controller/'))           

            # Capture Bag
            bag_path = (self.data_file / self.filename).resolve()
            self.get_logger().info(f"Capturing data from {topics}, output: {str(bag_path)}")
            cmd = ['ros2', 'bag', 'record', '-o', str(bag_path)] + topics
            self.record_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info(f"Started recording bag: {self.filename}")

    def stop_rosey_bags(self, msg: Bool):
            # End capture
            self.record_process.send_signal(signal.SIGINT)
            self.record_process.wait()
            self.record_process = None

            self.get_logger().info(f"Stopped recording bag: {self.filename}")


def main(args=None):
    rclpy.init(args=args)
    node = groundcontrol()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
        
