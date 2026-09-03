import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from evo.tools import file_interface
from evo.core.trajectory import PoseTrajectory3D

def run_conversion(bag_paths, output_dir, rgb_topic, depth_topic=None, pose_topic=None):
    output_path = Path(output_dir)
    rgb_dir = output_path / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    
    # Only create depth directory if topic is provided
    depth_dir = None
    if depth_topic:
        depth_dir = output_path / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)

    # Data containers for evo
    timestamps, positions, quaternions = [], [], []

    # Filter connections for speed
    target_topics = [rgb_topic]
    if depth_topic: target_topics.append(depth_topic)
    if pose_topic: target_topics.append(pose_topic)

    print(f"Reading {len(bag_paths)} bag(s)...")

    with AnyReader([Path(p) for p in bag_paths]) as reader:
        connections = [c for c in reader.connections if c.topic in target_topics]
        
        # Use contextlib ExitStack or simple conditional opening
        rgb_f = open(output_path / "rgb.txt", "w")
        depth_f = open(output_path / "depth.txt", "w") if depth_topic else None
            
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            t_sec = timestamp / 1e9  # ROS2 nanoseconds to TUM seconds

            # 1. Process RGB
            if connection.topic == rgb_topic:
                cv_img = message_to_cvimage(msg)
                img_name = f"{t_sec:.6f}.png"
                cv2.imwrite(str(rgb_dir / img_name), cv_img)
                rgb_f.write(f"{t_sec:.6f} rgb/{img_name}\n")

            # 2. Process Depth (Optional)
            elif depth_topic and connection.topic == depth_topic:
                cv_depth = message_to_cvimage(msg)
                depth_name = f"{t_sec:.6f}.png"
                cv2.imwrite(str(depth_dir / depth_name), cv_depth)
                depth_f.write(f"{t_sec:.6f} depth/{depth_name}\n")

            # 3. Process Pose (Optional Ground Truth)
            elif pose_topic and connection.topic == pose_topic:
                pose_obj = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
                timestamps.append(t_sec)
                positions.append([pose_obj.position.x, pose_obj.position.y, pose_obj.position.z])
                quaternions.append([
                    pose_obj.orientation.x, pose_obj.orientation.y, 
                    pose_obj.orientation.z, pose_obj.orientation.w
                ])

        rgb_f.close()
        if depth_f: depth_f.close()

    # 4. Save Trajectory (Only if pose data was found)
    if pose_topic and timestamps:
        traj = PoseTrajectory3D(np.array(positions), np.array(quaternions), np.array(timestamps))
        file_interface.write_tum_trajectory_file(output_path / "groundtruth.txt", traj)
        print(f"Ground truth saved with {len(timestamps)} poses.")
    elif pose_topic:
        print("Warning: Pose topic was specified but no messages were found.")

    print(f"Conversion complete. Data saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Convert ROS2 bags to TUM-RGBD format.")
    parser.add_argument("bags", nargs="+", help="Path to one or more ROS2 bag files/folders")
    parser.add_argument("-o", "--output", default="tum_output", help="Output directory")
    parser.add_argument("--rgb", required=True, help="Topic name for RGB images")
    parser.add_argument("--depth", help="Topic name for Depth images (optional)")
    parser.add_argument("--pose", help="Topic name for Pose (optional ground truth)")

    args = parser.parse_args()
    run_conversion(args.bags, args.output, args.rgb, args.depth, args.pose)

if __name__ == "__main__":
    main()
