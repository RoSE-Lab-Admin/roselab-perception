import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point

from ament_index_python.packages import get_package_share_directory
from gantry_lidar_interfaces.srv import Capture, DownloadName, DeleteName
from gantry_interfaces.msg import GantryState

import yaml
import json
from builtin_interfaces.msg import Time
import subprocess
from pathlib import Path as pth
import time
from datetime import datetime
import os

#runGantryScan shell script

#LIDAR is always running, just subscribe to the topics maybe or make it event driven perhaps
#pattern input automatic, potentially a lot of waypoints
class GantryCommand(Node):
    def __init__(self):
        super().__init__('gantry_command')

        #determine mode
        self.declare_parameter("movement_mode", "TRAJECTORY")
        movement_mode = self.get_parameter("movement_mode").value

        #set up output file
        self.declare_parameter("data_file")
        data_base = pth(self.get_parameter("data_file").value).expanduser().resolve()
        #name folders as dates and times
        self.day = datetime.now().strftime("%m%d%Y")
        hour = datetime.now().strftime("%H-%M-%S")
        self.data_file = pth(data_base) / self.day / hour
        #if folder doesnt exist, make it
        self.data_file.mkdir(parents=True, exist_ok=True)

        #set up trajectory input file
        current_file = get_package_share_directory("scan_automation")
        traj_folder = os.path.join(current_file, "path_files")
        self.declare_parameter("trajectory_file", "goto_cont_test.yaml")
        traj_file = self.get_parameter("trajectory_file").value
        self.path_file = os.path.join(traj_folder, traj_file)

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
        self.goto_pub = self.create_publisher(Point, '/gantry/gotoLocation', 10)

        #subscriber to gantry_control mode
        self.mode_sub = self.create_subscription(GantryState, '/gantry/gantry_status/gantry_state', self.read_mode, 10)

        #state variables
        self.tolerance = .02
        self.gantry_mode = None
        self.gantry_posx = None
        self.gantry_posy = None


        #wait for services to be ready
        while not self.gant_capture.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for capture service...")
        while not self.gant_download.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for download service...")
        while not self.gant_delete.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for delete service...")

        with open(self.path_file, "r") as f:
            self.waypoints = yaml.safe_load(f)

        if movement_mode == "TRAJECTORY":
            self.traj_mode_start()
        else:
            self.goto_mode_start()

        

    def read_mode(self, msg: String):
        self.gantry_mode = msg.controller_mode
        self.gantry_posx = msg.obs_gantry_position_c
        dir= msg.obs_gantry_position_e + msg.obs_gantry_position_w
        self.gantry_posy = dir/2


    def traj_mode_start(self):
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        first = Point()

        first_point = self.waypoints[0]

        last_point = self.waypoints[-1]
        self.duration = float(last_point["time"])

        first.x = float(first_point['position_x'])
        first.y = float(first_point['position_y'])
        first.z = 0.0

        go_string = String(data="GOTO")
        self.mode_pub.publish(go_string)
        rclpy.spin_once(self, timeout_sec=2.0)

        self.goto_pub.publish(first)

        # wait until goto is finished
        self.wait_for_end(first)
        
        for point in self.waypoints:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = float(point["position_x"])
            pose.pose.position.y = float(point["position_y"])

            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

            sec = int(point["time"])
            nsec = int((point["time"]-sec)*1e9)
            pose.header.stamp = Time(sec=sec, nanosec=nsec)

            self.path_msg.poses.append(pose)


        self.start_lidar()

        #start moving gantry
        self.trajectory_pub.publish(self.path_msg)
        self.get_logger().info("Published trajectory")

        mode_msg = String(data="TRAJECTORY")
        self.mode_pub.publish(mode_msg)

        rclpy.spin_once(self, timeout_sec=3.0)

        #wait until trajectory finished
        while self.gantry_mode != "HOLD":
            rclpy.spin_once(self, timeout_sec=3.0)
            self.get_logger().info("waiting to stop moving")

        self.end_scan()


    def goto_mode_start(self):
        self.path_msg = Path()

        self.goto_msgs = []
        self.stop_times = []

        #read trajectory in from file and store in point message list
        for point in self.waypoints:
            goal = Point()

            #reconstruct file into point message
            goal.x = float(point['position_x'])
            goal.y = float(point['position_y'])
            goal.z = 0.0
            scan_time = point['time']

            self.goto_msgs.append(goal)

            buffer_time = 5 * len(self.waypoints)

            #discerning between continuous and discrete
            if scan_time: #if not zero, aka discrete, add to scan time list, and find reasonable total scannig duration
                self.stop_times.append(scan_time)
                total_scan_time = sum(scan_time) #time its stopping
                #add five seconds for each point
                self.duration = buffer_time+total_scan_time

            else:
                self.duration = buffer_time

                
        #start lidar scan
        self.start_lidar()


        #if stop times are not the length of the points, it is continuous, otherwise its discrete
        if len(self.stop_times) != len(self.goto_msgs):
            self.go_to_cont()
        else:
            self.go_to_disc()
        

    def go_to_cont(self):
        self.get_logger().info('starting continuous path')
        #just making sure lol
        mode_hold = String(data='HOLD')
        self.mode_pub.publish(mode_hold)

        #give it a second to publish
        rclpy.spin_once(self, timeout_sec=3.0)

        mode_go = String(data='GOTO')

        for point in self.goto_msgs:
            #set mode, publish, wait until it gets to destination
            self.mode_pub.publish(mode_go)
            rclpy.spin_once(self, timeout_sec=3.0)
            self.goto_pub.publish(point)

            self.get_logger().info('point published')

            rclpy.spin_once(self, timeout_sec=3.0)

            #this logic might make it jerky, potentially should just manually
            #wait until movement is done
            self.wait_for_end(point)


            self.get_logger().info('ready to publish next point')

        self.mode_pub.publish(mode_hold)
        rclpy.spin_once(self, timeout_sec=3.0)
        self.end_scan()


    def go_to_disc(self):
        self.get_logger().info('starting discrete path')
        #just making sure lol
        mode_hold = String(data='HOLD')
        self.mode_pub.publish(mode_hold)

        rclpy.spin_once(self, timeout_sec=1.0)

        mode_go = String(data='GOTO')

        counter = 0

        for point in self.goto_msgs:
            #set mode, publish, wait until it gets there, begin count down

            self.mode_pub.publish(mode_go)
            rclpy.spin_once(self, timeout_sec=3.0)

            self.goto_pub.publish(point)

            #wait until reaches point
            self.wait_for_end(point)

            #wait until finished scanning
            time.sleep(self.stop_times[counter])
            counter += 1

        self.mode_pub.publish(mode_hold)
        rclpy.spin_once(self, timeout_sec=1.0)
        self.end_scan()

    def wait_for_end(self, goto_point):
        while (not self.gantry_posx) or (not self.gantry_posy):
            rclpy.spin_once(self, timeout_sec=1.0)
            self.get_logger().info("waiting for state publisher")

        while abs(self.gantry_posx-goto_point.x) >= self.tolerance and abs(self.gantry_posy-goto_point.y) >= self.tolerance: 
            rclpy.spin_once(self, timeout_sec=3.0)
            self.get_logger().info("waiting for goto position")

    #starts lidar bagging 
    def start_lidar(self):
        capture_request = Capture.Request()
        capture_request.outname = self.panda_file 
        capture_request.sensors = ["l515_center"] #, "l515_west", "l515_east"]

        # Capture duration should be length of trajectory + padding
        capture_request.duration = float(self.duration) + 10.0
        self.future_cap = self.gant_capture.call_async(capture_request)


    def end_scan(self):

        if self.future_cap and not self.future_cap.done():
            self.get_logger().info("lidar scanning.....")
            rclpy.spin_until_future_complete(self, self.future_cap)

        #get capture service repsonse
        cap_response = self.future_cap.result()

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

        #delete data from lattepanda
        delete_req = DeleteName.Request()
        delete_req.name = lidar_cap_data["outname"]
        future_delete = self.gant_delete.call_async(delete_req)
        rclpy.spin_until_future_complete(self, future_delete)

    
def main(args=None):
    rclpy.init()
    try:
        GantryCommand()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()




