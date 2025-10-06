import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point

from gantry_lidar_interfaces.srv import Capture, DownloadName, DeleteName

import yaml
import json
from builtin_interfaces.msg import Time
import subprocess
from pathlib import Path as pth
import time
from datetime import datetime

#runGantryScan shell script

#LIDAR is always running, just subscribe to the topics maybe or make it event driven perhaps
#wait i can just use the hold mode
#TODO: look at goto mode for gantry
#pattern input automatic, potentially a lot of waypoints
class GantryCommand(Node):
    def __init__(self):
        super().__init__('gantry_command')

        #set up output file
        self.declare_parameter("data_file", "D:/perception_data/default")
        data_base = self.get_parameter("data_file").value
        #name folders as dates and times
        self.day = datetime.now().strftime("%m%d%Y")
        hour = datetime.now().strftime("/%H-%M-%S")
        self.data_file = pth(data_base) / self.day / hour
        #if folder doesnt exist, make it
        self.data_file.mkdir(parents=True, exist_ok=True)

        #set up trajectory input file
        self.declare_parameter("trajectory_file", "ros/src/scan_automation/scan_automation/path_files/RH_test.yaml")
        self.path_file = self.get_parameter("trajectory_file").value

        #setup lattepanda output file
        self.declare_parameter("panda_file", "lidar_bags")
        self.panda_file = self.get_parameter("panda_file").value

        #service client setup
        self.gant_capture = self.create_client(Capture, "gantry_capture_service/capture")
        self.gant_download = self.create_client(DownloadName, "gantry_capture_service/download/name")
        self.gant_delete = self.create_client(DeleteName, "gantry_capture_service/delete/name")

        #publishers to gantry control
        self.trajectory_pub = self.create_publisher(Path, '/gantry/setTrajectory', 10)
        self.mode_pub = self.create_publisher(String, '/gantry/setMode', 10) 

        #wait for services to be ready
        while not self.gant_capture.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for capture service...")
        while not self.gant_download.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for download service...")
        while not self.gant_delete.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for delete service...")

        with open(self.path_file, "r") as f:
            waypoints = yaml.safe_load(f)

        self.path_msg = Path()

        #read trajectory in from file and store in path message
        self.path_msg.header.frame_id = "map"
        for point in waypoints:
            pose = PoseStamped()
            pose.header.frame_id="map"
            pose.pose.position.x = float(point["position_x"])
            pose.pose.position.y = float(point["position_y"])

            # RH: All of these fields MUST be floats!!!
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.
            pose.pose.orientation.y = 0.
            pose.pose.orientation.z = 0.
            pose.pose.orientation.w = 1.0

            sec = int(point["time"])
            nsec=int((point["time"]-sec)*1e9)
            pose.header.stamp = Time(sec=sec, nanosec=nsec)

            self.path_msg.poses.append(pose)
        
        #start trial
        # RH: Commenting out lidar capture (which is basically working!) until trajectory is actually being used
        #self.start_scan()
        self.move_gantry()

    # RH: Maybe we should support goto through this interface as well... takes a GeometryMsg Point type
    def move_gantry(self):
        #start moving the gantry
        self.get_logger().info(f"Publishing Trajectory to Gantry: {self.path_msg}")
        self.trajectory_pub.publish(self.path_msg)

        # RH: mode might need to come second?
        mode_msg = String(data = "TRAJECTORY")
        self.mode_pub.publish(mode_msg)

        #give it a second to publish messages
        rclpy.spin_once(self, timeout_sec=3.) # Upping the timeout since I think that's the issue

        # RH: WAIT FOR TRAJECTORY TO COMPLETE!!! HARDCODING FOR NOW
        duration = 30
        time.sleep(duration)

        # Finally set to HOLD for good measure
        mode_msg = String(data = "HOLD")
        self.mode_pub.publish(mode_msg)

        #give it a second to publish messages
        rclpy.spin_once(self, timeout_sec=3.) # Upping the timeout since I think that's the issue

    def start_scan(self):
        #start lidar
        future_cap = self.start_lidar()
        time.sleep(5)

        # Move gantry
        self.move_gantry()

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
        subprocess.Popen(["wget", "-r", "-P", f"{self.data_file}", f"{cap_url}"])
        time.sleep(10)

        self.get_logger().info("trial complete.")
        self.get_logger().info(f"bag saved to {self.data_file}")

    #helper function to send requests to bagger service
    def start_lidar(self):
        capture_request = Capture.Request()
        capture_request.outname = self.panda_file 
        capture_request.sensors = ["l515_center"] #, "l515_west", "l515_east"]

        # Capture duration should be length of trajectory + 2 * padding
        capture_request.duration = 60.0
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




