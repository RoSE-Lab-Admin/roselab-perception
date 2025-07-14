# Load Bag, and fuse with time probably
# Color match the plate (research color matching)
# Plane fit the plate  (research plane fitting)
# Grab center (or origin) of plate, store as origin
# Get normal of plane, store as Y
# Determine orientation of x, z
# Store all 
# Get transaltion based on camera frame (examples)
# Make translation matrix
# Rotation based on slope of plane
# Make rotation matrix
# Return both

# ROS
import rclpy
import rosbag2_py
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage

# Math and Vision
import numpy as np
import cv2
import matplotlib.pyplot as plt

# System/Misc
from pathlib import Path
import time
import threading
from tqdm import tqdm

class OriginFusion():
    def __init__(self):
        self.color_images = []
        self.depth_images = []

    def LoadBag(self, bag_path):
        with AnyReader([Path(bag_path)]) as reader:
            # Make list of color and depth images
            for connection, timestamp, rawdata in tqdm(reader.messages(), total=2452):
                if connection.topic == '/l515_center/color/image_raw':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    img = message_to_cvimage(msg, 'rgb8')
                    self.color_images.append(img)
                if connection.topic == '/l515_center/aligned_depth_to_color/image_raw':
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    img = message_to_cvimage(msg, '16UC1')
                    if(img.shape == (360,640)): self.depth_images.append(img)

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
        # Make stacked and median images
        stacked_color = np.stack(self.color_images, axis=0)
        self.median_color_img = np.median(stacked_color, axis=0).astype(np.uint8)
        stacked_depth = np.stack(self.depth_images, axis=0)
        self.median_depth_img = np.median(stacked_depth, axis=0).astype(np.uint16)
        #cv2.imwrite("/mnt/c/Users/ryan1/Downloads/img.tiff", self.median_depth_img)
    
    def PlotImages(self):
        # Plot median images
        fig, axes=plt.subplots(1,2)
        axes[0].imshow(self.median_color_img)
        axes[1].imshow(self.median_depth_img, cmap="inferno", vmin=2300.0, vmax=2400.0)
        #axes[1][1].hist(self.median_depth_img.flat, bins=100)
        plt.show()

    def GetOrigin(self):
        # Get Arcuo

        # Find Static TF for arcu in world frame (frame cad)

        # Plane fit to get normal vector (Ryan H)
        
        return #translation and rotation matricies
    
if __name__ == "__main__":
    bag_path = "/mnt/c/Users/ryan1/Downloads/07102025_15_07_51/07102025_15_07_51/192.168.2.4:8000"
    originFuser = OriginFusion()
    originFuser.LoadBag(bag_path)
    originFuser.PlotImages()
    # Having the surface characterization run as its own node makes sense. Maybe i should just make this a service outright and skip the joining service.
    #originFuser.GetOrigin()