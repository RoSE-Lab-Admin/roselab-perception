import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

from gantry_lidar_interfaces.srv import Capture, DownloadName, DeleteName

import yaml
import json
from builtin_interfaces.msg import Time
import subprocess
from pathlib import Path as pth
import time

#runGantryScan shell script

#LIDAR is always running, just subscribe to the topics maybe or make it event driven perhaps
#wait i can just use the hold mode
#TODO: look at goto mode for gantry
#pattern input automatic, potentially a lot of waypoints
class GantryCommand(Node):
    def __init__(self):
        super().__init__('gantry_command')

        #service client setup
        self.gant_capture = self.create_client(Capture, "gantry_capture_service/capture")
        self.gant_download = self.create_client(DownloadName, "gantry_capture_service/download/name")
        self.gant_delete = self.create_client(DeleteName, "gantry_capture_service/delete/name")

        #publishers to gantry control
        self.trajectory_pub = self.create_publisher(Path, 'gantry/SetTrajectory', 10)
        self.mode_pub = self.create_publisher(String, '/gantry/setMode', 10) 

        #wait for services to be ready
        while not self.gant_capture.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for capture service...")
        while not self.gant_download.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for download service...")
        while not self.gant_delete.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for delete service...")

        #where bags will be stored
        self.path = str(pth.home() / "rosbags" / "912-test")
        pth(self.path).mkdir(parents=True, exist_ok=True)

        #user file name choice
        self.test_name = str(input("File name: "))

        with open(f"{self.test_name}.yaml", "r") as f:
            waypoints = yaml.safe_load(f)

        self.path_msg = Path()

        #read trajectory in from file and store in path message
        self.path_msg.header.frame_id = "map"
        for point in waypoints:
            pose = PoseStamped()
            pose.header.frame_id="map"
            pose.pose.position.x = float(point["position_x"])
            pose.pose.position.y = float(point["position_y"])
            pose.pose.position.z = 0  
            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = 0
            pose.pose.orientation.w = 1.0

            sec = int(point["time"])
            nsec=int((point["time"]-sec)*1e9)
            pose.header.stamp = Time(sec=sec, nanosec=nsec)

            self.path_msg.poses.append(pose)
        
        #start trial
        self.start_scan()

    def start_scan(self):
        #start lidar
        future_cap = self.start_lidar()
        time.sleep(5)

        #start moving the gantry
        mode_msg = String(data = "TRAJECTORY")
        self.mode_pub.publish(mode_msg)
        self.trajectory_pub.publish(self.path_msg)
        #give it a second to publish messages
        rclpy.spin_once(self, timeout_sec=.1)

        #wait until lidar scan done
        rclpy.spin_until_future_complete(self, future=future_cap)

        #start downloading and ending processes
        self.end_scan(future_cap)

    def end_scan(self, future_cap):
        #get capture service repsonse
        cap_response = future_cap.result()

        #lidar capture data
        lidar_cap_data = json.loads(cap_response.outdata)

        # request download url name
        name_request = DownloadName.Request()
        name_request.name = lidar_cap_data["outname"]
        self.get_logger().info(f"Lidar bag 1 name: {name_request.name}")

        #recieve name
        future_name = self.gant_download.call_async(name_request)
        #spin until recieved
        rclpy.spin_until_future_complete(self, future_name)
        #recieved name
        name_response = future_name.result()
        self.get_logger().info("got service call name")

        #process response
        name_response_dict = json.loads(name_response.outdata)
        #get url
        cap_url = name_response_dict["url"]
        self.get_logger().info(f"url: {cap_url}")

        #download data from url
        subprocess.Popen(["wget", "-r", "-P", f"{self.path}", f"{cap_url}"])
        time.sleep(10)

        self.get_logger().info("trial complete.")
        self.get_logger().info(f"bag saved to {self.path}")

    #helper function
    def start_lidar(self):
        capture_request = Capture.Request()
        capture_request.outname = self.test_name + "_lidar"
        capture_request.sensors = ["l515_center", "l515_west", "l515_east"]
        capture_request.duration = 10.0
        future_cap = self.gant_capture.call_async(capture_request)
        return future_cap
    
def main(args=None):
    rclpy.init()
    try:
        GantryCommand()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()




