# mastcam_capture_service.py
# Adds explicit START (no duration) and STOP services alongside your existing timed Capture.
# START uses the same Capture.srv (ignores duration) so you can still pass sensors + outname.
# STOP uses std_srvs/Trigger to end the active recording gracefully.

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from mastcam_interfaces.srv import Capture, DownloadName, DeleteName
import subprocess
import signal
import shutil

from datetime import datetime
from pathlib import Path
import json
import time

# --- Config ---
DATA_DIR = Path("D:/perception_data")  # Use Path consistently
TIME_STR = "%Y-%m-%dT%H-%M-%S"
HTTP_PORT = "8000"

import socket

def get_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return local_ip

def start_http_server():
    # Serves DATA_DIR over HTTP for simple downloads
    subprocess.Popen([
        "python3", "-m", "http.server", HTTP_PORT,
        "--directory", str(DATA_DIR),
        "--bind", get_local_ip()
    ])

def parse_time(timestr):
    return datetime.strptime(timestr, TIME_STR)

class MastcamCaptureService(Node):
    def __init__(self):
        super().__init__('mastcam_capture_service')

        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Runtime state for recording lifecycle
        self.record_process = None         # subprocess.Popen handle
        self.filename = None               # "outname_timestamp"
        self.active_topics = []            # list[str] recorded in the current session

        # --- Service Endpoints ---
        # Info
        self.create_service(Trigger, 'mastcam_capture_service/info', self.info_callback)

        # Timed capture (your existing API)
        self.create_service(Capture, 'mastcam_capture_service/capture', self.capture_callback)

        # Start (no duration): begins recording until STOP is called
        # Reuse Capture.srv so caller can pass outname + sensors, we IGNORE duration here.
        self.create_service(Capture, 'mastcam_capture_service/start', self.start_no_duration_callback)

        # Stop: ends any active recording started by START
        self.create_service(Trigger, 'mastcam_capture_service/stop', self.stop_callback)

        # Download (by name)
        self.create_service(DownloadName, 'mastcam_capture_service/download/name', self.download_name_callback)
        # self.create_service(DownloadTimeRange, 'mastcam_capture_service/download/timeRange', self.download_time_range_callback)

        # Delete (by name)
        self.create_service(DeleteName, 'mastcam_capture_service/delete/name', self.delete_name_callback)
        # self.create_service(DeleteTimeRange, 'mastcam_capture_service/delete/timeRange', self.delete_time_range_callback)

        # Start HTTP server for simple file serving
        start_http_server()

        self.get_logger().info("MastCam capture service running")

    # ----------------------------
    # Helpers
    # ----------------------------
    def _build_topics(self):
        """
        Construct the topic list from sensor names.
        """
        topics = ["/tf", "/tf_static"]
        topics.append("/MastCam/Front/color/image_raw")
        topics.append("/MastCam/Front/color/camera_info")
        topics.append("/MastCam/Front/extrinsics/depth_to_color")
        topics.append("/MastCam/Front/aligned_depth_to_color/image_raw")
        topics.append("/MastCam/Front/aligned_depth_to_color/camera_info")
        topics.append("/MastCam/Front/depth/image_rect_raw")
        topics.append("/MastCam/Front/depth/camera_info")
        
        return topics

    def _start_bag(self, outname):
        """
        Start a ros2 bag record subprocess. Stores handle + filename + topics.
        """

        # Format filename: "outname_timestamp"
        ts = datetime.now().strftime(TIME_STR)
        self.filename = f"{outname}_{ts}"

        # Build topics and bag path
        self.active_topics = self._build_topics()
        bag_path = (DATA_DIR / self.filename).resolve()

        # Launch ros2 bag record
        cmd = ['ros2', 'bag', 'record', '-o', str(bag_path)] + self.active_topics
        self.get_logger().info(f"Recording to: {bag_path}")
        self.get_logger().info("Topics:\n  " + "\n  ".join(self.active_topics))

        # Start process; suppress stdout/stderr to keep node logs clean
        self.record_process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def _stop_bag(self):
        """
        Stop the active ros2 bag record, if any, and wait for exit.
        """
        if not self.record_process:
            return False

        try:
            # Send SIGINT for clean closure (ros2 bag handles ctrl-c)
            self.record_process.send_signal(signal.SIGINT)
            self.record_process.wait(timeout=15)
        except Exception:
            # Escalate if needed
            try:
                self.record_process.terminate()
                self.record_process.wait(timeout=5)
            except Exception:
                self.record_process.kill()
                self.record_process.wait()

        # Clear state
        self.record_process = None
        self.active_topics = []
        return True

    # ----------------------------
    # Services
    # ----------------------------
    def info_callback(self, request, response):
        response.success = True
        response.message = "MastCam service ready (endpoints: capture, start, stop, download/name, delete/name)"
        return response

    def capture_callback(self, request, response):
        """
        Timed capture: start bag, sleep for duration, then stop, return JSON with info.
        Request:
          - float32 duration
          - string[] sensors
          - string outname
        Response:
          - string outdata (JSON)
        """
        try:
            duration = float(request.duration)
            sensors = list(request.sensors)
            outname = str(request.outname)

            if self.record_process is not None:
                # Prevent overlapping sessions
                msg = {"status": "ERROR", "reason": "Recording already active. Stop first."}
                response.outdata = json.dumps(msg)
                return response

            self._start_bag(outname, sensors)
            self.get_logger().info(f"Started timed recording: {self.filename} for {duration}s")

            time.sleep(duration)  # Blocking wait (simple)

            stopped = self._stop_bag()
            self.get_logger().info(f"Stopped timed recording: {self.filename}")

            response.outdata = json.dumps({
                "status": "ACK",
                "mode": "timed",
                "duration": duration,
                "sensors": sensors,
                "outname": self.filename,
                "stopped": stopped
            })

        except Exception as e:
            response.outdata = json.dumps({"status": "ERROR", "reason": str(e)})
        return response

    def start_no_duration_callback(self, request, response):
        """
        START (no duration): begins recording until STOP is called.
        We reuse Capture.srv so the caller can pass outname.
        The 'duration' field is IGNORED.
        """
        sensors = ["mastcam"]
        try:
            outname = str(request.outname)

            if self.record_process is not None:
                msg = {"status": "ERROR", "reason": "Recording already active. Stop first."}
                response.outdata = json.dumps(msg)
                return response
        

            self._start_bag(outname, sensors)
            self.get_logger().info(f"Started continuous recording: {self.filename}")

            response.outdata = json.dumps({
                "status": "ACK",
                "mode": "continuous",
                "sensors": sensors,
                "outname": self.filename
            })
        except Exception as e:
            response.outdata = json.dumps({"status": "ERROR", "reason": str(e)})
        return response

    def stop_callback(self, request, response):
        """
        STOP: ends the active recording started by START (or any active capture).
        Uses std_srvs/Trigger.
        """
        try:
            if self.record_process is None:
                response.success = False
                response.message = json.dumps({"status": "ERROR", "reason": "No active recording"})
                return response

            current_name = self.filename  # capture for the response before clearing
            stopped = self._stop_bag()
            self.get_logger().info(f"Stopped recording: {current_name}")

            response.success = True
            response.message = json.dumps({
                "status": "ACK",
                "stopped": stopped,
                "outname": current_name
            })
        except Exception as e:
            response.success = False
            response.message = json.dumps({"status": "ERROR", "reason": str(e)})
        return response

    def download_name_callback(self, request, response):
        """
        Download bags by exact folder name (e.g., 'mytrial_2025-10-06T12-34-56').
        Returns a URL served by the local HTTP server.
        """
        try:
            outname = request.name
            matches = sorted(DATA_DIR.glob(f"{outname}"), reverse=True)
            if not matches:
                self.get_logger().info("Download name request failed, file not found.")
                response.outdata = json.dumps({"success": False, "error": "Not found"})
                return response

            folder = matches[0].name
            url = f"http://192.168.2.104:{HTTP_PORT}/{folder}"
            self.get_logger().info(f"Download name request: {url}")

            response.outdata = json.dumps({"success": True, "url": url})
        except Exception as e:
            response.outdata = json.dumps({"success": False, "error": str(e)})
        return response

    def download_time_range_callback(self, request, response):
        try:
            start = request.start
            end = request.end
            response.outdata = f"Downloaded files from {start} to {end}"
        except Exception as e:
            response.outdata = f"Error parsing request: {str(e)}"
        return response

    def delete_name_callback(self, request, response):
        """
        Delete a bag folder by name.
        """
        try:
            outname = request.name
            if outname == "":
                msg = {"status": "ERROR", "reason": "Empty name. Delete by time range instead."}
                response.outdata = json.dumps(msg)
                return response

            matches = list(DATA_DIR.glob(f"{outname}"))
            if not matches:
                response.outdata = json.dumps({"success": False, "error": f"No bag found with name '{outname}'"})
                return response

            deleted = []
            for path in matches:
                if path.is_dir():
                    shutil.rmtree(path)
                    deleted.append(path.name)

            self.get_logger().info(f"Deleted {outname}")
            response.outdata = json.dumps({"success": True, "deleted": deleted})
        except Exception as e:
            response.outdata = json.dumps({"success": False, "error": str(e)})
        return response

    def delete_time_range_callback(self, request, response):
        """
        Delete all bag folders whose timestamp suffix lies within [start, end].
        Folder name format expected: "<prefix>_<YYYY-mm-ddTHH-MM-SS>"
        """
        try:
            start = datetime.strptime(request.start, TIME_STR)
            end = datetime.strptime(request.end, TIME_STR)

            deleted = []
            for path in DATA_DIR.iterdir():
                if not path.is_dir():
                    continue
                try:
                    ts_str = path.name.split('_')[-1]
                    ts = datetime.strptime(ts_str, TIME_STR)
                    if start <= ts <= end:
                        shutil.rmtree(path)
                        deleted.append(path.name)
                except Exception:
                    continue

            response.outdata = json.dumps({"success": True, "deleted": deleted})
        except Exception as e:
            response.outdata = json.dumps({"success": False, "error": str(e)})
        return response


def main(args=None):
    rclpy.init(args=args)
    node = MastcamCaptureService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
