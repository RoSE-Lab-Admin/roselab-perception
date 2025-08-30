import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from gantry_lidar_interfaces.srv import Capture, DownloadName, DownloadTimeRange, DeleteName, DeleteTimeRange
import subprocess # For running ROS bag commands
import signal # For sending shutdown sig
import shutil

from datetime import datetime # For parsing by time range
from pathlib import Path
import json
import time

DATA_DIR = Path.home() / "RoSE_Bags"
TIME_STR = "%Y-%m-%dT%H-%M-%S"

def start_http_server():
    subprocess.Popen([
        "python3", "-m", "http.server", "8000",
        "--directory", str(DATA_DIR),
        "--bind", "192.168.2.4"
    ])

def parse_time(timestr):
        return datetime.strptime(timestr, TIME_STR)

class GantryCaptureService(Node):
    def __init__(self):
        super().__init__('gantry_capture_service')

        # Info
        self.create_service(Trigger, 'gantry_capture_service/info', self.info_callback)
        
        # Capture sensor data
        self.create_service(Capture, 'gantry_capture_service/capture', self.capture_callback)

        # Download
        self.create_service(DownloadName, 'gantry_capture_service/download/name', self.download_name_callback)
        self.create_service(DownloadTimeRange, 'gantry_capture_service/download/timeRange', self.download_time_range_callback)

        # Delete
        self.create_service(DeleteName, 'gantry_capture_service/delete/name', self.delete_name_callback)
        self.create_service(DeleteTimeRange, 'gantry_capture_service/delete/timeRange', self.delete_time_range_callback)
        
        # Start HTTP server
        start_http_server()

        self.get_logger().info("Gantry capture service running")

    def info_callback(self, request, response):
        response.success = True
        response.message = "idk what to put here"
        return response

    def capture_callback(self, request, response):
        """
        Capture a bag for a set time and saves it locally, zipped
        Service Arguments
        - duration: how long to capture for
        - sensors: list of str names of sensors
        - outname: name of bag to save. bag will be saved as "outname_timestamp"
        """
        self.get_logger().info(str(DATA_DIR))
        try:
            #self.get_logger().info(request)
            #req_data = json.loads(request.data)
            duration = request.duration #req_data.get("duration", 10.0)
            sensors = request.sensors #req_data.get("sensors", [])
            outname = request.outname #req_data.get("outname", "")

            # Match directories like: "outname_*"
            #matches = list(DATA_DIR.glob(f"{outname}_*"))
            
            # If file exists 
            # if matches:
            #     self.get_logger().info(f"Filename already exists: {self.outname}")
            #     response.outdata = json.dumps({"status": "ERROR", "reason": "filename already exists"})
            #     return response
            
            # Format filename
            timestamp = datetime.now().strftime(TIME_STR)
            self.filename = f"{outname}_{timestamp}"

            # Set Topics
            topics = []
            topics.append("/tf")
            topics.append("/tf_static")
            
            for sensor in sensors:
                if (sensor == "l515_center") or (sensor == "l515_west") or (sensor == "l515_east"):
                    #topics.append("/"+sensor+"/depth/image_rect_raw")
                    topics.append("/"+sensor+"/aligned_depth_to_color/image_raw")
                    topics.append("/"+sensor+"/color/image_raw")
                    #topics.append("/"+sensor+"/depth/camera_info")
                    topics.append("/"+sensor+"/aligned_depth_to_color/camera_info")
                    topics.append("/"+sensor+"/color/camera_info")
                    topics.append("/"+sensor+"/extrinsics/depth_to_color")
                    #topics.append("/"+sensor+"/imu")
                    

            # Capture Bag
            bag_path = (DATA_DIR / self.filename).resolve()
            self.get_logger().info(f"Capturing data: {duration}s from {topics}, output: {str(bag_path)}")
            cmd = ['ros2', 'bag', 'record', '-o', str(bag_path)] + topics
            self.record_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info(f"Started recording bag: {self.filename}")

            time.sleep(duration) # Delay

            # End capture
            self.record_process.send_signal(signal.SIGINT)
            self.record_process.wait()
            self.record_process = None

            self.get_logger().info(f"Stopped recording bag: {self.filename}")
           
            # Zip bag
            #self.get_logger().info(f"Making .zip archive: {self.filename}.zip")
            #shutil.make_archive(self.filename, 'zip', DATA_DIR / self.filename)

            # Delete original folder
            #self.get_logger().info(f"Deleting original: {self.filename}")
            #shutil.rmtree(DATA_DIR / self.filename)

            response.outdata = json.dumps({
                "status": "ACK",
                "duration": duration,
                "sensors": sensors,
                "outname": self.filename
            })

        except Exception as e:
            response.outdata = json.dumps({"status": "ERROR", "reason": str(e)})
        return response
    
    def download_name_callback(self, request, response):
        """
        Download bags by name. All bags with matching name will be downloaded (although there should only be one)
        """
        try:
            #data = json.loads(request.data)
            outname = request.name

            # Get matches (Should only have one)
            matches = sorted(DATA_DIR.glob(f"{outname}"), reverse=True)
            if not matches:
                self.get_logger().info(f"Download name request failed, file not found.")
                response.outdata = json.dumps({"success": False, "error": "Not found"})
                return response

            # Figure out what the http path to the zip is
            folder = matches[0].name
            ip = "192.168.2.4"
            url = f"http://{ip}:8000/{folder}"

            self.get_logger().info(f"Download name request with: {url}")

            # Return the zip path for wget by client
            response.outdata = json.dumps({
                "success": True,
                "url": url
            })

        except Exception as e:
            response.outdata = json.dumps({"success": False, "error": str(e)})
        return response

    def download_time_range_callback(self, request, response):
        """
        Download bags by time range. All bags between start and end will be downloaded.
        Time stamp should be formated like "yyyy-mm-ddThh-mm-ss" in 24hr time
        """
        try:
            #data = json.loads(request.data)
            start = request.start #data.get("start")
            end = request.end #data.get("end")
            response.outdata = f"Downloaded files from {start} to {end}"
        except Exception as e:
            response.outdata = f"Error parsing request: {str(e)}"
        return response

    def delete_name_callback(self, request, response):
        """
        Delete bags by name. All bags with matching name will be deleted (although there should only be one)
        """
        try:
            #data = json.loads(request.data)
            outname = request.name#data.get("name")
            if outname == "":
                self.get_logger().info(f"Must delete a named file if deleting by name. Delete by time range instead.")
                response.outdata = json.dumps({"status": "ERROR", "reason": "Can't delete unnamed file. Delete by time range instead."})
                return response
            # Match directories like: "outname"
            matches = list(DATA_DIR.glob(f"{outname}"))
            
            # No matches
            if not matches:
                response.outdata = json.dumps({
                    "success": False,
                    "error": f"No bag found with name '{outname}'"
                })
                return response

            # Get list of files to be deleted (should only be one)
            deleted = []
            for path in matches:
                if path.is_dir():
                    shutil.rmtree(path)
                    deleted.append(path.name)

            self.get_logger().info(f"Deleted {outname}")
                

            response.outdata = json.dumps({
                "success": True,
                "deleted": deleted
            })

        except Exception as e:
            response.outdata = json.dumps({"success": False, "error": str(e)})
        return response



    def delete_time_range_callback(self, request, response):
        """
        Delete bags by time range. All bags between start and end will be deleted.
        Time stamp should be formated like yyyy-mm-ddThh-mm-ss" in 24hr time
        """
        try:
            #data = json.loads(request.data)
            start = datetime.strptime(request.start, TIME_STR)
            end = datetime.strptime(request.end, TIME_STR)

            deleted = []
            for path in DATA_DIR.iterdir():
                if not path.is_dir():
                    continue
                try:
                    # Extract timestamp from last underscore
                    ts_str = path.name.split('_')[-1]
                    ts = datetime.strptime(ts_str, TIME_STR)
                    if start <= ts <= end:
                        shutil.rmtree(path)
                        deleted.append(path.name)
                except Exception:
                    continue

            response.outdata = json.dumps({
                "success": True,
                "deleted": deleted
            })

        except Exception as e:
            response.outdata = json.dumps({"success": False, "error": str(e)})
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GantryCaptureService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



"""
TODO
- Troubleshoot LiDAR
    - Poke around and getting them working again
- Gather throughput measurements
    - Latte Panda LiDAR (single with two not launched), do we see bandwidth issues. 
    - ROS w/ 1 LiDAR, 3 LiDAR
    - Realsense SDK w/ 1 LiDAR, 3 LiDAR
    - Measure with wireshark or something like that (USB Top) ( measuring usb ? )
    - Vary depth resolution, fps (30fps & 15fps & whatever else) Depth and color
    - Get quantitatify measurements
    - Hardware sync
    - Looking at dropped frames
- Investigate hardware sync
- lidar calibration (color data also needs calibration if possible?) (bonus)

! NOTES
- Ultimately we make the sdk pull data and write our own publisher.
- USB C direction affects to bandwidth
- Figure out HOW im going to do bandwidth measurement

? What all should I put in the matrix
- 15 and 30 fps, various res
- Two tables, one with HW sync and one without

? Is the matrix all depth + color
- Look at point clouds, but focus on depth and color
- Find out if it happens on chip

"""