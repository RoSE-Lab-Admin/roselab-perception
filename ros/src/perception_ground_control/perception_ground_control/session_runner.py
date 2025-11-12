import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from mastcam_interfaces.srv import Capture as MastCapture, DeleteName as MastDeleteName, DownloadName as MastDownloadName, Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

class SessionRunner(Node):
    def __init__(self):
        super().__init__('perception_cli')

        # command line publishers
        self.pub_start_lidar = self.create_client(Duration, 'ground_control_service/lidar/start')
        # call from cli: ros2 topic pub --once /start_lidar std_msgs/msg/Bool "{data: true}"
        self.pub_start_mastcam = self.create_client(Trigger, 'ground_control_service/start/mastcam')
        # call from cli: ros2 topic pub --once /start_mastcam std_msgs/msg/Bool "{data: true}"
        self.pub_stop_mastcam = self.create_client(Trigger, 'ground_control_service/stop/mastcam')
        # call from cli: ros2 topic pub --once /stop_mastcam std_msgs/msg/Bool "{data: true}"
        self.pub_start_rosey_bag = self.create_client(Trigger, 'ground_control_service/start/rosey_bag')
        # call from cli: ros2 topic pub --once /start_rosey_bag std_msgs/msg/Bool "{data: true}"
        self.pub_stop_rosey_bag = self.create_client(Trigger, 'ground_control_service/stop/rosey_bag')
        # call from cli: ros2 topic pub --once /stop_rosey_bag std_msgs/msg/Bool "{data: true}"
        self.pub_update_savedir = self.create_client(Trigger, 'ground_control_service/update_savedir')

        
        # Gather session parameters from the launched ground control node
        # Need directory we're saving to for metadata.text file
        fut = self.pub_update_savedir.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut)
        self.run_session()

        self.get_logger().info('Perception CLI Node has been started.')

    # Template for publishing Bool messages
    def run_session(self):
        self.get_logger().info("TURN OFF CAMERAS IN MOTIVE!!!")
        raw = input("Please enter duration (in seconds) of Lidar Scan: ")
        lidar_request = Duration.Request()
        lidar_request.duration = float(raw)
        lidar_future = self.pub_start_lidar.call_async(lidar_request)
        rclpy.spin_until_future_complete(self, lidar_future)
        self.get_logger().info('Published start_lidar command.')

        Ntrials = input("Enter number of trials: ")
        for n in range(int(Ntrials)):
            self.get_logger().info("TURN ON CAMERAS IN MOTIVE!!!")
            raw = input("Hit [ENTER] to start Mastcam and Rosey")       
            bag_request = Trigger.Request()
            future_bag = self.pub_start_mastcam.call_async(bag_request)
            future_ros = self.pub_start_rosey_bag.call_async(Trigger.Request())
            self.get_logger().info("sent")
 
            rclpy.spin_until_future_complete(self, future_bag)
            self.get_logger().info('Published start_mastcam command.')
            rclpy.spin_until_future_complete(self, future_ros)
            self.get_logger().info('Published start_rosey_bag command.')

            raw = input("Hit [ENTER] to stop mastcam and rosey")
            future_stop = self.pub_stop_mastcam.call_async(bag_request)            
            future_sto = self.pub_stop_rosey_bag.call_async(bag_request)
            rclpy.spin_until_future_complete(self, future_stop)
            self.get_logger().info('Published stop_mastcam command.')
            rclpy.spin_until_future_complete(self, future_sto)
            self.get_logger().info('Published stop_rosey_bag command.')

            self.get_logger().info("TURN OFF CAMERAS IN MOTIVE!!!")
            raw = input("Please enter duration (in seconds) of Lidar Scan: ")
            lidar_request = Duration.Request()
            lidar_request.duration = float(raw)
            lidar_future = self.pub_start_lidar.call_async(lidar_request)
            rclpy.spin_until_future_complete(self, lidar_future)
            self.get_logger().info('Published start_lidar command.')

        self.get_logger().info('Session complete! Closing...')

def main():
    rclpy.init()
    perception_cli = SessionRunner()
    rclpy.spin(perception_cli)
    perception_cli.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()