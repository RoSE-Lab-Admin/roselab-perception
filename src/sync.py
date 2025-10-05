import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosbags.highlevel import AnyReader
from sensor_msgs.msg import Image, PointCloud2
import os
import sys
from pathlib import Path
from tqdm import tqdm

class BagSynchronizer:
    def __init__(self, bag_path, topic1, topic2, topic3, *cb_args, callback=None, info=None, tol=50000000, **cb_kwargs):
        self.bag_path = Path(bag_path)
        self.topic1 = topic1
        self.topic2 = topic2
        self.camera_info_topic = topic3
        self.callback = callback if callback else lambda *cb_args, **cb_kwargs: None
        self.tol = tol

        self.buffer1 = []
        self.buffer2 = []
        self.info = info

    def synchronize(self):
        # Infer types from IDL files or .msg files if using custom messages
        # You may need to provide custom type paths, but AnyReader handles
        # many common message types automatically.
        try:
            with AnyReader([self.bag_path]) as reader:
                # Store connections for easy lookup
#                connections = {conn.id: conn for conn in reader.connections}

                for connection, timestamp, rawdata in tqdm(reader.messages()):
                    # Deserialize message data
#                    msg = deserialize_message(rawdata, connections[connection.id].msgtype)
                    msg = reader.deserialize(rawdata, connection.msgtype)

                    # Add messages to respective buffers
                    if connection.topic == self.topic1:
                        self.buffer1.append((timestamp, msg))
                    elif connection.topic == self.topic2:
                        self.buffer2.append((timestamp, msg))
                    elif connection.topic == self.camera_info_topic:
                        self.info = msg

                    if self.info is None:
                        # Wait until we have intrinsics to apply
                        continue

                    # Simple synchronization logic:
                    # Find the oldest message in each buffer that is close in time
                    while self.buffer1 and self.buffer2:
                        ts1, msg1 = self.buffer1[0]
                        ts2, msg2 = self.buffer2[0]

                        # Define a time threshold (e.g., 50ms = 50,000,000 ns)
                        if abs(ts1 - ts2) < self.tol:
                            print(f"Synchronized pair found at timestamps: {ts1} and {ts2}")
                            print(f"  {self.topic1} message timestamp: {msg1.header.stamp.sec}.{msg1.header.stamp.nanosec}")
                            print(f"  {self.topic2} message timestamp: {msg2.header.stamp.sec}.{msg2.header.stamp.nanosec}")
                            # Process the synchronized messages here
                            self.process_synchronized_messages() # Could probably just ref self.buffer1 and self.buffer2 instead within this method

                            # Remove the processed messages from the buffers
                            self.buffer1.pop(0)
                            self.buffer2.pop(0)
                        elif ts1 < ts2:
                            # Discard the older message if no match is found
                            self.buffer1.pop(0)
                        else:
                            self.buffer2.pop(0)

                print("Finished processing the bag file.")

        except FileNotFoundError:
            print(f"Error: The bag file was not found at {self.bag_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    # Instead of making this as general as possible, I should probably just write one that works for what I'm trying to do here.
    # Explicitly write the class to synchronize based on color/depth, and then choose nearest neighbor camera_info for alignment and nearest pose for projection to point cloud 
    def process_synchronized_messages(self):
        """
        Callback-like function to handle the synchronized messages.
        Replace this with your desired logic.
        """
        # Example: Save images, perform analysis, etc.
        # RH: implement rgbd handling here: register, convert to o3d, project into point cloud
        self.callback()

if __name__ == '__main__':
    # Replace with the actual path to your bag file and topic names
    bag_file = sys.argv[1]
    color_topic = sys.argv[2]
    depth_topic = sys.argv[3]
    cam_info_topic = sys.argv[4]

    synchronizer = BagSynchronizer(bag_file, color_topic, depth_topic, cam_info_topic)
    synchronizer.synchronize()
