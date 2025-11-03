#!/usr/bin/env python3
"""
offline_rgbd_tsdf_from_bag_memcapped.py

Parallel TSDF reconstruction with pose interpolation and per-worker memory cap.
Each worker integrates at most --chunk-size RGBD frames, then frees memory.
This keeps peak RAM predictable even with many cores.

Usage:
  python3 offline_rgbd_tsdf_from_bag_memcapped.py \
      --bag /path/to/bag \
      --workers 8 \
      --chunk-size 256
"""
import argparse
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import numpy as np
import open3d as o3d
from cv_bridge import CvBridge
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile, os, math, time
import tqdm

# ---------- Parameters ----------
COLOR_TOPIC = '/MastCam/Front/color/image_raw'
DEPTH_TOPIC = '/MastCam/Front/depth/image_rect_raw'
CAM_INFO_TOPIC = '/MastCam/Front/color/camera_info'
POSE_TOPIC = '/CubeRover_V1/pose'

DEPTH_SCALE = 1000.0
DEPTH_TRUNC = 3.0
VOXEL_LENGTH = 0.02
SDF_TRUNC = 0.06
POSE_MATCH_MAX = 0.2
# --------------------------------

# ---------- Pose helpers ----------
def quat_to_matrix(qx, qy, qz, qw):
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-8:
        return np.eye(3)
    s = 1.0 / n
    xx, yy, zz = qx*qx*s, qy*qy*s, qz*qz*s
    xy, xz, yz = qx*qy*s, qx*qz*s, qy*qz*s
    wx, wy, wz = qw*qx*s, qw*qy*s, qw*qz*s
    R = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ])
    return R

def pose_to_matrix(pose_msg):
    p = pose_msg.pose.position
    q = pose_msg.pose.orientation
    R = quat_to_matrix(q.x, q.y, q.z, q.w)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = np.array([p.x, p.y, p.z])
    return T

def quat_to_array(R):
    qw = np.sqrt(1 + np.trace(R)) / 2
    qx = (R[2,1] - R[1,2]) / (4*qw)
    qy = (R[0,2] - R[2,0]) / (4*qw)
    qz = (R[1,0] - R[0,1]) / (4*qw)
    return np.array([qx,qy,qz,qw])

def quat_to_rot(q):
    x,y,z,w = q
    return quat_to_matrix(x,y,z,w)

def slerp(q1, q2, t):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = q1 + t*(q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    s0 = np.sin((1-t)*theta_0)/np.sin(theta_0)
    s1 = np.sin(t*theta_0)/np.sin(theta_0)
    return s0*q1 + s1*q2

def interpolate_pose(t_target, poses):
    times = [p[0] for p in poses]
    if t_target <= times[0]:
        return poses[0][1]
    if t_target >= times[-1]:
        return poses[-1][1]
    for i in range(len(poses)-1):
        t0,T0 = poses[i]
        t1,T1 = poses[i+1]
        if t0 <= t_target <= t1:
            dt = t1-t0
            if dt > POSE_MATCH_MAX:
                return None
            alpha = (t_target - t0)/dt
            p0,p1 = T0[:3,3],T1[:3,3]
            p_interp = (1-alpha)*p0 + alpha*p1
            q0,q1 = quat_to_array(T0[:3,:3]), quat_to_array(T1[:3,:3])
            q_interp = slerp(q0,q1,alpha)
            R_interp = quat_to_rot(q_interp)
            T = np.eye(4)
            T[:3,:3] = R_interp
            T[:3,3] = p_interp
            return T
    return None

# ---------- Bag read ----------
def read_rosbag2(bag_path, topics):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in topics: continue
        msg_type = topics[topic]
        msg = deserialize_message(data, msg_type)
        yield topic, msg, t/1e9

# ---------- Worker ----------
def integrate_chunk(frames, cam_info, poses, chunk_id):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        cam_info.width, cam_info.height,
        cam_info.k[0], cam_info.k[4], cam_info.k[2], cam_info.k[5]
    )
    bridge = CvBridge()
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LENGTH,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    for (t_color, color_msg, depth_msg) in frames:
        pose_T = interpolate_pose(t_color, poses)
        if pose_T is None:
            continue
        color_cv = bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        depth_cv = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_cv),
            o3d.geometry.Image(depth_cv.astype(np.float32)),
            depth_scale=DEPTH_SCALE,
            depth_trunc=DEPTH_TRUNC,
            convert_rgb_to_intensity=False
        )
        try:
            tsdf.integrate(rgbd, intrinsic, np.linalg.inv(pose_T))
        except:
            raise Warning("Couldn't integrate with Open3D \\-'_'-/")

    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    tmpfile = tempfile.mktemp(prefix=f"chunk_{chunk_id}_", suffix=".ply")
    o3d.io.write_triangle_mesh(tmpfile, mesh)
    return tmpfile

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--workers", type=int, default=8, help="Max concurrent workers")
    parser.add_argument("--chunk-size", type=int, default=256, help="Frames per worker chunk")
    args = parser.parse_args()

    topics = {
        COLOR_TOPIC: Image,
        DEPTH_TOPIC: Image,
        CAM_INFO_TOPIC: CameraInfo,
        POSE_TOPIC: PoseStamped,
    }

    color_msgs, depth_msgs, poses = [], [], []
    cam_info = None
    print("Reading bag...")
    for topic, msg, t in read_rosbag2(args.bag, topics):
        if topic == COLOR_TOPIC: color_msgs.append((t, msg))
        elif topic == DEPTH_TOPIC: depth_msgs.append((t, msg))
        elif topic == CAM_INFO_TOPIC and cam_info is None: cam_info = msg
        elif topic == POSE_TOPIC: poses.append((t, pose_to_matrix(msg)))

    assert cam_info is not None, "No camera info found!"
    frames = []
    for t_color, color_msg in color_msgs:
        depth_match = min(depth_msgs, key=lambda x: abs(x[0]-t_color), default=None)
        if depth_match is None or abs(depth_match[0]-t_color) > 0.03:
            continue
        frames.append((t_color, color_msg, depth_match[1]))

    total = len(frames)
    chunks = [frames[i:i+args.chunk_size] for i in range(0, total, args.chunk_size)]
    print(f"Total frames: {total} → {len(chunks)} chunks of up to {args.chunk_size} each")
    # print(f"Example chunk: {chunks[0]}")

    tmpfiles = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(integrate_chunk, chunk, cam_info, poses, idx): idx for idx, chunk in enumerate(chunks[:4])}
        for f in tqdm.tqdm(as_completed(futures)):
            idx = futures[f]
            try:
                res = f.result()
                tmpfiles.append(res)
                print(f"✓ Finished chunk {idx+1}/{len(chunks)} → {res}")
            except Exception as e:
                print(f"✗ Chunk {idx} failed: {e}")

    print("Merging partial meshes...")
    meshes = [o3d.io.read_triangle_mesh(f) for f in tmpfiles]
    merged = o3d.geometry.TriangleMesh()
    for m in tqdm.tqdm(meshes):
        merged += m
    merged.merge_close_vertices(0.005)
    merged.compute_vertex_normals()

    o3d.io.write_triangle_mesh("scene_mesh.ply", merged)
    print("Saved scene_mesh.ply")

if __name__ == "__main__":
    main()
