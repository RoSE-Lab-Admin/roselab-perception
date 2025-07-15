# ROS
import rclpy
import rosbag2_py
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

# Math and Vision
import numpy as np
import cv2
import matplotlib.pyplot as plt

# System/Misc
from pathlib import Path
import time
import threading
from tqdm import tqdm

from rosbags.rosbag2 import Reader

bag1_path = Path("/home/ryan/bags/07102025_15_07_51/192.168.2.4:8000")
bag2_path = Path("/home/ryan/bags/07102025_15_07_51/pose")

class OriginFusion():
    def __init__(self):
        self.color_images = []
        self.depth_images = []
        self.median_color_img = None
        self.median_depth_img = None

        self.pose_positions = np.array(0)
        self.pose_orientations = np.array(0)
        self.pose_positions = (0,0,0)
        self.pose_orientation = (0,0,0,0)

    def LoadBag(self, bag_path):
        """
        Given a bag path, update self.median_color_img and self.median_depth_img.
        """
        with AnyReader([bag1_path]) as reader:
            # Make list of color and depth images
            for connection, timestamp, rawdata in tqdm(reader.messages(), total=2452):
                # Color
                if connection.topic == '/l515_center/color/image_raw':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    img = message_to_cvimage(msg, 'rgb8')
                    self.color_images.append(img)
                # Depth
                if connection.topic == '/l515_center/aligned_depth_to_color/image_raw':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    img = message_to_cvimage(msg, '16UC1')
                    print(self.color_images[0].shape[:2])
                    if(img.shape == self.color_images[0].shape[:2]): self.depth_images.append(img) # Take only the aligned ones
        #with AnyReader([bag2_path]) as reader2:
            # Make list of color and depth images
            #for connection, timestamp, rawdata in tqdm(reader2.messages()):
                # Pose
                if connection.topic == '/Origin_Square/pose':
                    print(rawdata)
                    msg = reader.deserialize(rawdata, connection.msgtype)  # PoseStamped
                    p = msg.pose.position
                    q = msg.pose.orientation
                    self.pose_positions.append((p.x, p.y, p.z))
                    self.pose_orientations.append((q.x, q.y, q.z, q.w))

        # Run your stacking and median in a thread (so we can show elapsed time)
        t = threading.Thread(target=self.stack_task)
        t.start()
        with tqdm(total=0, bar_format="{desc} {elapsed}") as pbar:
            while t.is_alive():
                time.sleep(0.1)
                pbar.set_description("Stacking and Getting Median")
                pbar.update(0)
        t.join()

    def stack_task(self):
        """
        Helper function. Using image arrays, stack, then take the median.
        Also takes mean of position and average orientation
        """
        # Make stacked and median images
        stacked_color = np.stack(self.color_images, axis=0)
        self.median_color_img = np.median(stacked_color, axis=0).astype(np.uint8)
        stacked_depth = np.stack(self.depth_images, axis=0)
        self.median_depth_img = np.median(stacked_depth, axis=0).astype(np.uint16)
        #cv2.imwrite("/mnt/c/Users/ryan1/Downloads/img.tiff", self.median_depth_img)

        # Take average postiion
        # self.pose_position = np.average(self.pose_positions)
        
        # # Form the symmetric accumulator matrix
        # A = np.zeros((4, 4))
        # print(self.pose_orientations.shape)
        # M = self.pose_orientations.shape[0]
        # count = 0
        # for i in range(M):
        #     q = self.pose_orientations[i, :]
        #     A += (np.outer(q, q)) # Rank 1 update
        #     count += 1
        # # Scale
        # A /= count
        # # Get the eigenvector corresponding to largest eigen value
        # self.pose_orientation = np.linalg.eigh(A)[1][:, -1]
        # print(self.pose_orientation)
    
    def PlotImages(self):
        """
        Plot both median images with matplotlib
        """
        # Plot median images
        fig, axes=plt.subplots(1,2)
        axes[0].imshow(self.median_color_img)
        axes[1].imshow(self.median_depth_img, cmap="inferno", vmin=2300.0, vmax=2400.0)
        #axes[1][1].hist(self.median_depth_img.flat, bins=100)
        plt.show()

    def GetOrigin(self):
        # Get Aruco (pose in camera) (Ryan H)

        # Plane fit to get normal vector (Ryan H)
        
        return #translation and rotation matricies
    
class StaticArucoToWorld(Node):
    def __init__(self):
        super().__init__('static_aruco_to_world')
        # create the broadcaster
        self._broadcaster = StaticTransformBroadcaster(self)

        # fill in your manual measurements here
        translation = {
            'x': 0.0687,  # meters from ArUco to world origin along X
            'y': -0.007, # meters along Y
            'z': 0.1309   # meters along Z
        }
        rotation = {
            'x': 0.0,   # quaternion x
            'y': 1.0,   # quaternion y
            'z': 0, # quaternion z
            'w': 1.5707963  # quaternion w
        }

        # create and publish the static transform
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'      # parent frame
        t.child_frame_id = 'aruco_frame' # child frame

        t.transform.translation.x = translation['x']
        t.transform.translation.y = translation['y']
        t.transform.translation.z = translation['z']

        t.transform.rotation.x = rotation['x']
        t.transform.rotation.y = rotation['y']
        t.transform.rotation.z = rotation['z']
        t.transform.rotation.w = rotation['w']

        # sendTransform works for static transforms too
        self._broadcaster.sendTransform(t)

if __name__ == "__main__":
    #bag_path = "/mnt/c/Users/ryan1/Downloads/07102025_15_07_51/07102025_15_07_51/"
    originFuser = OriginFusion()
    originFuser.LoadBag("")
    originFuser.PlotImages()

    rclpy.init()
    node = StaticArucoToWorld()
    rclpy.spin_once(node, timeout_sec=0.1)