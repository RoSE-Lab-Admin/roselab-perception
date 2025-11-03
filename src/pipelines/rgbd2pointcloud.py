#!/usr/bin/env python3
"""
Offline RGBD -> Open3D point cloud aggregator (chunked)
Reads an MCAP bag with ROS2 Jazzy messages, synchronizes color/depth,
applies nearest pose, and writes partial PLY chunks to disk to control memory usage.
"""

import os
import math
import argparse
import numpy as np
import open3d as o3d

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header

#from nav_msgs.msg import Odometry

from pathlib import Path
from tqdm import tqdm
import warnings

# Use scipy instead for this shit
def quat_to_matrix(qx, qy, qz, qw):
    """Return 4x4 transform from quaternion."""
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T

# Add ability to return TWO nearest samples (one above, one below)
def find_nearest(sorted_msgs, t_ns, nearest=True):
    """Return the message whose timestamp is closest to t_ns, or closest messages before and after t_ns."""
    if nearest:
        if not sorted_msgs:
            return None

        # I should probably use np.searchsorted for this and keep vectors of the times of each topic, or at least pose
        return min(sorted_msgs, key=lambda x: abs(x[0] - t_ns))
    else:
        if not sorted_msgs:
            return [None, None]

        # Otherwise, let's find nearest neighbors before and after t_ns
        # Find index of nearest, determine sooner (negative) or later (positive) than t_ns
        nni = np.argmin(sorted_msgs, key=lambda x: abs(x[0] - t_ns))
        nn = sorted_msgs[nni]

        # Return (nn_sooner, nn_later)
        return sorted([nn, sorted_msgs[int(nni+np.sign(nn[0] - t_ns)))]], key=lambda x: x[0])

def build_intrinsic(cam_info: CameraInfo):
    fx, fy = cam_info.k[0], cam_info.k[4]
    cx, cy = cam_info.k[2], cam_info.k[5]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        cam_info.width, cam_info.height, fx, fy, cx, cy
    )
    return intrinsic


def process_rgbd_pair(color_msg, depth_msg, pose_msg, bridge, intrinsic, depth_scale, depth_trunc):
    """Create and transform a point cloud from synchronized RGBD and pose."""
    color_cv = bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
    depth_cv = bridge.imgmsg_to_cv2(depth_msg)
    depth_cv = depth_cv.astype(np.float32) / depth_scale

    color_o3d = o3d.geometry.Image(color_cv)
    depth_o3d = o3d.geometry.Image(depth_cv)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0,
        depth_trunc=depth_trunc, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    if pose_msg:
        if isinstance(pose_msg, PoseStamped):
            pos = pose_msg.pose.position
            ori = pose_msg.pose.orientation
        else:
            raise TypeError("Optitrack pose topic is not PoseStamped.")

        # RH: use scipy rotation instead
        T = quat_to_matrix(ori.x, ori.y, ori.z, ori.w)
        T[:3, 3] = [pos.x, pos.y, pos.z]
        pcd.transform(T)

    return pcd

def setup_parser():
    parser = argparse.ArgumentParser(description="Chunked offline RGBD -> Open3D point cloud aggregator")
    parser.add_argument("--bag", required=True, help="Path to .mcap file")
    parser.add_argument("--color", required=True, help="Color image topic")
    parser.add_argument("--depth", required=True, help="Depth image topic")
    parser.add_argument("--camera-info", required=True, help="CameraInfo topic")
    parser.add_argument("--pose", required=True, help="PoseStamped or Odometry topic")
    parser.add_argument("--out-dir", default="./chunks", help="Directory to save chunked PLYs")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-trunc", type=float, default=15.0) # Setting default to well over operating range of D456
    parser.add_argument("--slop", type=float, default=0.05, help="Approx sync slop in seconds")
    parser.add_argument("--chunk-size", type=int, default=100, help="Frames per chunk")
    parser.add_argument("--interp", type=bool, default=True, help="Whether to interpolate poses to aligned frames")
    return parser

# Simple weighted average of two pose msgs to a target time (which should be between their timestamps)
def interpolate_pose_msgs(pre, post, t_ns):
    # Compute weightings
    total = np.abs(post - pre)
    w_pre = 1. - abs(pre-t_ns) / total
    w_post = 1. - abs(post-t_ns) / total

    # Extract rotational components and average
    # Create a Slerp object: Slerp(keyframe_times, keyframe_quats)
    pre_q = pre[1].pose.orientation
    post_q = post[1].pose.orientation

    slerp = Slerp([pre[0], post[0]], [pre.pose.orientation])

    # Interpolate at intermediate times
    interp_q = slerp(t_ns).as_quaternion()

    # Extract translational components and average

    # Construct new pose msg
    avg_pose_msg = PoseStamped()
    avg_pose_msg.header = Header()
    avg_pose_msg.header.frame_id = pre[1].header.frame_id
    avg_pose_msg.header.stamp = t_ns * 1e9 # This needs to be formatted into a proper TimeStamp object

    # Now fill with pose info

    return new_pose_msg

# This is so stupid... going to rewrite as a class and use queues soon.
def process_chunk(args, chunk_idx, color_msgs, depth_msgs, pose_msgs, camera_info, bridge, mode='nn'):
    if not camera_info:
        warnings.warn(f"No CameraInfo found in bag chunk /# {chunk_index}")

    intrinsic = build_intrinsic(camera_info)
    slop_ns = args.slop * 1e9

    color_msgs.sort(key=lambda x: x[0])
    depth_msgs.sort(key=lambda x: x[0])
    pose_msgs.sort(key=lambda x: x[0])

    print(f"Parsed: {len(color_msgs)} color frames, {len(depth_msgs)} depth frames, {len(pose_msgs)} poses.")

    current_chunk = o3d.geometry.PointCloud()

    if args.interp:
        # If interp is set to True, we will interpolate between poses (and possibly do a weighted average of frames?)
        for i, (c_time, c_msg) in tqdm(enumerate(color_msgs)):
            # Find approx depth match to color frame at t=c_time
            d_near = [d for d in depth_msgs if abs(d[0] - c_time) <= slop_ns]
            if not d_near:
                continue
            d_time, d_msg = min(d_near, key=lambda x: abs(x[0] - c_time))

            # Find TWO nearest poses to target time (above and below) and do weighted average to target time
            pose_sooner, pose_later = find_nearest(pose_msgs, c_time, nearest=False)
            if (not pose_sooner) or (not pose_later):
                continue
            else:
                # perform weighted average on these messages and return a new pose_msg at c_time
                pose_msg = interpolate_pose_msgs(pose_msg_sooner, pose_msg_later, c_time)

            # Process frame
            pcd = process_rgbd_pair(
                c_msg, d_msg, pose_msg, bridge,
                intrinsic, args.depth_scale, args.depth_trunc
            )
            current_chunk += pcd

    else:
        # Do our normal nearest neighbor approach
        for i, (c_time, c_msg) in tqdm(enumerate(color_msgs)):
            # Find approx depth match
            d_near = [d for d in depth_msgs if abs(d[0] - c_time) <= slop_ns]
            if not d_near:
                continue
            d_time, d_msg = min(d_near, key=lambda x: abs(x[0] - c_time))

            # Nearest pose
            pose_pair = find_nearest(pose_msgs, c_time)
            if not pose_pair:
                continue
            else:
                pose_msg = pose_pair[1]

            # Process frame
            pcd = process_rgbd_pair(
                c_msg, d_msg, pose_msg, bridge,
                intrinsic, args.depth_scale, args.depth_trunc
            )
            current_chunk += pcd

    # FILTER AND DOWNSAMPLE!!!!
    current_chunk, inds = current_chunk.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    current_chunk = current_chunk.voxel_down_sample(voxel_size=0.005) # Hardcoded to 5mm to start

    chunk_path = os.path.join(args.out_dir, f"chunk_{chunk_idx:03d}.ply")
    o3d.io.write_point_cloud(chunk_path, current_chunk)
    print(f"Saved {chunk_path} with {len(current_chunk.points)} points")

def main():
    parser = setup_parser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bridge = CvBridge()

    storage_options = StorageOptions(uri=args.bag, storage_id='mcap')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_types = {t: get_message(t) for t in topic_types.values()}

    color_msgs, depth_msgs, pose_msgs = [], [], []
    camera_info = None

    print("Reading messages...")

    CHUNK = args.chunk_size
    chunk_idx = 0

    # We should probably unpack the camera info directly first
    while reader.has_next():
        topic, data, t = reader.read_next()
        msg = deserialize_message(data, msg_types[topic_types[topic]])
        if topic == args.color:
            color_msgs.append((t, msg))
        elif topic == args.depth:
            depth_msgs.append((t, msg))
        elif topic == args.pose:
            pose_msgs.append((t, msg))
        elif topic == args.camera_info:
            camera_info = msg

        # break if color and depth topics have reached limit, and pose topic's latest message is after the latest color message
        if len(color_msgs) >= CHUNK and len(depth_msgs) >= CHUNK:
            if pose_msgs[-1][0] >= color_msgs[-1][0]:

                # Instead of breaking, let's turn all of the following into a chunk processing call, clear the messages, and let the while loop keep going
                process_chunk(args, chunk_idx, color_msgs, depth_msgs, pose_msgs, camera_info, bridge) # writes out chunk to disk

                # clear and continue while loop
                color_msgs, depth_msgs, pose_msgs = [], [], []
                chunk_idx += 1

                print(f"Processed {(chunk_idx+1) * CHUNK} RGBD pairs.")

    # Global merge
    chunk_files = sorted([Path(args.out_dir) / Path(f) for f in os.listdir(args.out_dir) if f.endswith(".ply")])

    print("Aggregating pointcloud chunks...")
    print(f"Filenames: {chunk_files}")

    aggregated = o3d.geometry.PointCloud()
    for f in tqdm(chunk_files):
        pcd = o3d.io.read_point_cloud(f)
        aggregated += pcd

    # Do a final filter and downsample?
    # Maybe we should have a TSDF volume fit here instead of simply adding point clouds together...
    # Then we'd extract mesh or point cloud without enforcing grid
    # aggregated, inds = aggregated.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    # aggregated = aggregated.voxel_down_sample(voxel_size=0.005) # 1mm resolution grid? might have a lot of empty voxels and backfaces.

    merged_path = os.path.join(args.out_dir, "aggregated.ply")
    o3d.io.write_point_cloud(merged_path, aggregated)
    print(f"Final aggregated cloud saved: {merged_path}")

    # Finally, let's clean things up by deleting all the chunk point clouds...
    print("Deleting temporary chunk files...")
    for ply_file in tqdm(chunk_files):
        try:
            ply_file.unlink()
        except Exception as e:
            print(f"Could not delete {ply_file}: {e}")
    print("Done!")

if __name__ == "__main__":
    main()
