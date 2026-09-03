import argparse
import cv2
import numpy as np
import bisect
import os
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from evo.tools import file_interface
from evo.core.trajectory import PoseTrajectory3D
from tqdm import tqdm

class ROS2RGBDStreamer:
    def __init__(self, bag_paths, rgb_topic, depth_topic=None, pose_topic=None, 
                 camera_info_topic=None, intrinsics=None, max_delta=0.033, sync_motion=False):
        self.bag_paths = [Path(p) for p in bag_paths]
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.pose_topic = pose_topic
        self.camera_info_topic = camera_info_topic
        self.intrinsics = intrinsics # [fx, fy, cx, cy]
        self.max_delta = max_delta
        self.sync_motion = sync_motion

        self.write_plan = {}
        self.interpolated_poses = {}
        self.target_stamps = []

    def _get_3d_points(self, pts2d, depth_img, fx, fy, cx, cy):
        pts3d, valid_idx = [], []
        for i, (u, v) in enumerate(pts2d):
            if v >= depth_img.shape[0] or u >= depth_img.shape[1]: continue
            z = depth_img[int(v), int(u)] / 1000.0
            if 0.1 < z < 10.0:
                pts3d.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
                valid_idx.append(i)
        return np.array(pts3d), valid_idx

    def _detect_visual_motion_rgbd(self, rgb_msgs, depth_msgs, threshold=0.02):
        if len(rgb_msgs) < 2 or not depth_msgs or not self.intrinsics: return None
        fx, fy, cx, cy = self.intrinsics

        ref_rgb = cv2.cvtColor(message_to_cvimage(rgb_msgs[0][1]), cv2.COLOR_BGR2GRAY)
        ref_depth = message_to_cvimage(depth_msgs[0][1])

        p0 = cv2.goodFeaturesToTrack(ref_rgb, maxCorners=500, qualityLevel=0.01, minDistance=10)
        pts3d_ref, valid_idx = self._get_3d_points(p0.reshape(-1, 2), ref_depth, fx, fy, cx, cy)
        p0 = p0[valid_idx]

        for i in range(1, len(rgb_msgs)):
            curr_rgb = cv2.cvtColor(message_to_cvimage(rgb_msgs[i][1]), cv2.COLOR_BGR2GRAY)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(ref_rgb, curr_rgb, p0, None)
            idx = (st.flatten() == 1)

            success, _, tvec = cv2.solvePnP(pts3d_ref[idx], p1[idx], 
                                            np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]), None)
            if success and np.linalg.norm(tvec) > threshold:
                return rgb_msgs[i][0]
        return None

    def _get_nearest_neighbor(self, target_t, timestamps):
        if not timestamps: return None
        idx = bisect.bisect_left(timestamps, target_t)
        candidates = []
        if idx > 0: candidates.append(timestamps[idx - 1])
        if idx < len(timestamps): candidates.append(timestamps[idx])
        if not candidates: return None
        
        best_t = min(candidates, key=lambda t: abs(t - target_t))
        if abs(best_t - target_t) <= self.max_delta: return best_t
        return None

    def _analyze_and_sync(self):
        """Pass 1: Scans timestamps, extracts initial intrinsics, and plans SLERP."""
        t_rgb, t_depth, t_cinfo, raw_poses = [], [], [], []
        cached_rgb, cached_depth = [], []
        CACHE_LIMIT = 200

        target_topics = [self.rgb_topic,
                        self.depth_topic,
                        self.pose_topic,
                        self.camera_info_topic]

        print(f"[INFO] Beginning analysis with target topics: ")
        for tt in target_topics:
            print(f"\t{tt}")

        t_sec = None
        tf = np.eye(4)

        with AnyReader(self.bag_paths) as reader:
            conns = [c for c in reader.connections if c.topic in target_topics]
            for conn, ts, raw in reader.messages(connections=conns):
                # RH: BAD!!! DON'T USE RECORDING TIME!!! USE MESSAGE HARDWARE TIMESTAMPS!!!!!!!!!!!
                # t_sec = ts / 1e9
                msg = reader.deserialize(raw, conn.msgtype)
                if hasattr(msg, 'header'):
                    # Access the header stamp fields (sec and nanosec)
                    t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                elif conn.msgtype == 'gantry_interfaces/msg/GantryState':
                    t_sec = ts / 1e9 # Evidently gantry doesnt' have a hardware timestamp recorded nor a header.... :/
                else:
                    continue

                if conn.topic == self.rgb_topic:
                    t_rgb.append(t_sec)
                    if self.sync_motion and len(cached_rgb) < CACHE_LIMIT:
                        cached_rgb.append((t_sec, reader.deserialize(raw, conn.msgtype)))
                
                elif conn.topic == self.depth_topic:
                    t_depth.append(t_sec)
                    if self.sync_motion and len(cached_depth) < CACHE_LIMIT:
                        cached_depth.append((t_sec, reader.deserialize(raw, conn.msgtype)))
                
                elif conn.topic == self.camera_info_topic:
                    # t_cinfo.append(t_sec)
                    # Auto-extract intrinsics from the very first message if none provided
                    if self.intrinsics is None:
                        msg = reader.deserialize(raw, conn.msgtype)
                        # ROS CameraInfo 'k' is a 9-element array: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                        self.intrinsics = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]
                        print(f"Auto-extracted initial intrinsics [fx, fy, cx, cy]: {self.intrinsics}")

                # RH: Add gantry state parsing in here too so we can generate dynamic lidar scans
                elif conn.topic == self.pose_topic:
                    msg = reader.deserialize(raw, conn.msgtype)

                    # RH: Check msgtype first
                    if conn.msgtype == 'gantry_interfaces/msg/GantryState':
                        # For now, assume 2D plane, orientation doesn't change
                        # tf[:3,:3] = np.eye(3)

                        # tf[:3,3] = np.r_[
                            # msg["encoder_c"], # Carriage is x dir
                            # np.nanmean([msg["encoder_e"], msg["encoder_w"]]), # Take average for y dir
                            # 0. # This would normally be set to the actual height (hopefully constant...) of the cart w.r.t MLSS origin
                        # ]
                        raw_poses.append([t_sec, msg.encoder_c, np.nanmean([msg.encoder_e, msg.encoder_w]), 0.0,
                            0.0, 0.0, 0.0, 1.0]) # Identity rotation (unless we eventually form a rotation about Z due to E-W asynchrony

                    else: # Assume regular posestamped msgtype
                        p = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose
                        raw_poses.append([t_sec, p.position.x, p.position.y, p.position.z,
                                      p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])

        t_rgb.sort()
        t_depth.sort()
        # t_cinfo.sort()
        pose_arr = np.array(raw_poses)

        print(f"[INFO] Found {len(t_rgb)} color frames, {len(t_depth)} depth frames, and {len(pose_arr)} poses.")
        print(f"\tColor start time = {t_rgb[0]} s")
        print(f"\tDepth start time = {t_depth[0]} s")
        print(f"\tPose start time  = {pose_arr[0][0]} s")

        # For Lidar system, hardware clock is used to stamp data but is not synced with system clock necessarily!!! More timing issues yayyyyyyyyyyyyyyyyy.................
        # RH: Quick fix, correct pose stream to earlier of color or depth feed (100s of ms error probably, but only matters relative to data which is synced from hardware clock)
        # RH:    Alternatively, we can just use our motion sync provided that the scan starts from rest...
        pose_arr_offset = pose_arr[0,0] - min(t_rgb[0], t_depth[0])
        pose_arr[:,0] -= pose_arr_offset

        # Geometric Sync
        if self.sync_motion and self.intrinsics and len(raw_poses) > 0:
            print("[INFO] Performing motion-based stream time synchronization offset.")
            t_vis = self._detect_visual_motion_rgbd(cached_rgb, cached_depth)
            dist = np.linalg.norm(np.diff(pose_arr[:, 1:4], axis=0), axis=1)
            m_idx = np.where(dist > 0.005)[0]
            if t_vis and len(m_idx) > 0:
                offset = t_vis - pose_arr[m_idx[0], 0]
                pose_arr[:, 0] += offset
            print(f"[INFO] Found offset = {offset:.3}s")

        del cached_rgb, cached_depth

        # Master-Slave Matching
        pose_start = pose_arr[0, 0] if len(pose_arr) > 0 else 0
        pose_end = pose_arr[-1, 0] if len(pose_arr) > 0 else float('inf')

        # Check len of rgb and depth and set smaller one to master
        comp = (len(t_rgb) < len(t_depth))
        master = t_rgb if comp else t_depth
        slave = t_depth if comp else t_rgb

        print("[INFO] Beginning nearest-neighbor message synchronization.")
        print(f"\tUsing master={'RGB' if comp else 'Depth'} and slave={'Depth' if comp else 'RGB'}.")

        for t_r in tqdm(master):
            if not (pose_start <= t_r <= pose_end):
                continue
            
            # RH: CHANGE THIS TO USE SMALLER SET OF IMAGES AS THE SYNC TARGET (AKA MASTER)
            # Match Slave topic to Master
            match_s = self._get_nearest_neighbor(t_r, slave) # ADD FLAG HERE FOR CHECKING FOR SAME IMAGE SHAPE???
            if match_s is None: continue
            self.write_plan[match_s] = t_r
            
            # Match Camera Info to Master
            # match_c = self._get_nearest_neighbor(t_r, t_cinfo)
            # if match_c is None: continue
            # self.write_plan[match_c] = t_r

            self.write_plan[t_r] = t_r
            self.target_stamps.append(t_r)

        # SLERP Interpolation
        if self.target_stamps:
            interp_tx = interp1d(pose_arr[:, 0], pose_arr[:, 1])
            interp_ty = interp1d(pose_arr[:, 0], pose_arr[:, 2])
            interp_tz = interp1d(pose_arr[:, 0], pose_arr[:, 3])
            
            rots = Rotation.from_quat(pose_arr[:, 4:8])
            slerp = Slerp(pose_arr[:, 0], rots)
            interp_quats = slerp(self.target_stamps).as_quat()
            
            for i, t in tqdm(enumerate(self.target_stamps)):
                self.interpolated_poses[t] = [
                    interp_tx(t), interp_ty(t), interp_tz(t),
                    interp_quats[i, 0], interp_quats[i, 1], interp_quats[i, 2], interp_quats[i, 3]
                ]

        print(f"[INFO] {len(self.target_stamps)} synced messages triplets found")

    def stream(self):
        """Pass 2 Generator: Yields (timestamp, pose_tuple, rgb_img, depth_img, intrinsics_list)."""
        # RH: Maybe add a flag or property which checks whether we have filled synced timestamp variables?
        #self._analyze_and_sync()

        expected_keys = ['rgb', 'depth']

        frame_buffer = {}

        target_topics = [self.rgb_topic, self.depth_topic] #, self.camera_info_topic]
        valid_topics = [t for t in target_topics if t is not None]

        t_sec = None

        with AnyReader(self.bag_paths) as reader:
            conns = [c for c in reader.connections if c.topic in valid_topics]
            for conn, ts, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                if hasattr(msg, 'header'):
                    # Access the header stamp fields (sec and nanosec)
                    t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

                if t_sec in self.write_plan:
                    master_t = self.write_plan[t_sec]
                    if master_t not in frame_buffer: frame_buffer[master_t] = {}

                    msg = reader.deserialize(raw, conn.msgtype)

                    if conn.topic == self.rgb_topic:
                        frame_buffer[master_t]['rgb'] = message_to_cvimage(msg)
                    elif conn.topic == self.depth_topic and 'depth' not in frame_buffer[master_t]:
                        # frame_buffer[master_t]['depth'] = np.clip(np.asarray(message_to_cvimage(msg)) * 5., 0.0, 2**16-1).astype(np.uint16) # RH: Hack to trick GSFusion into divide by !!!!!!!!
                        frame_buffer[master_t]['depth'] = message_to_cvimage(msg)
                    # elif conn.topic == self.camera_info_topic and 'cinfo' not in frame_buffer[master_t]:
                    #     frame_buffer[master_t]['cinfo'] = [msg.k[0], msg.k[4], msg.k[2], msg.k[5]]

                    # Yield if triplet/quadruplet is fully assembled
                    if all(k in frame_buffer[master_t] for k in expected_keys):
                        rgb_img = frame_buffer[master_t].get('rgb')
                        depth_img = frame_buffer[master_t].get('depth')
                        # cinfo = frame_buffer[master_t].get('cinfo', self.intrinsics)
                        pose = self.interpolated_poses.get(master_t)

                        yield (master_t, pose, rgb_img, depth_img)
                        del frame_buffer[master_t]

    def save_tum_dataset(self, output_dir):
        """Consumes the stream and writes perfectly aligned data to disk."""
        out = Path(output_dir)
        (out / "rgb").mkdir(parents=True, exist_ok=True)
        (out / "depth").mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Saving perfectly aligned TUM dataset to '{output_dir}'...")
        rgb_f = open(out / "rgb.txt", "w")
        depth_f = open(out / "depth.txt", "w")
        
        poses_for_evo = []
        timestamps_for_evo = []

        # Initialize for streaming
        print("[INFO] Analyze and synchronize messages.")
        self._analyze_and_sync()

        # Write out intrinsics!
        print("[INFO] Saving intrinsics.")
        with open(out / "intrinsics.txt", 'w') as f:
            f.write(f"fx: {self.intrinsics[0]}\n")
            f.write(f"fy: {self.intrinsics[1]}\n")
            f.write(f"cx: {self.intrinsics[2]}\n")
            f.write(f"cy: {self.intrinsics[3]}\n")

        # Consume the generator!
        print("[INFO] Writing TUM dataset...")
        for t, pose, rgb, depth in tqdm(self.stream(), total=len(self.target_stamps)):
            img_name = f"{t:.6f}.png"

            cv2.imwrite(str(out / "rgb" / img_name), rgb)
            rgb_f.write(f"{t:.6f} rgb/{img_name}\n")

            if self.depth_topic and depth is not None:
                cv2.imwrite(str(out / "depth" / img_name), depth)
                depth_f.write(f"{t:.6f} depth/{img_name}\n")

            if pose is not None:
                poses_for_evo.append(pose)
                timestamps_for_evo.append(t)

        rgb_f.close()
        depth_f.close()

        if poses_for_evo:
            pose_arr = np.array(poses_for_evo)
            traj = PoseTrajectory3D(pose_arr[:, 0:3], pose_arr[:, 3:7], np.array(timestamps_for_evo))
            file_interface.write_tum_trajectory_file(out / "groundtruth.txt", traj)

        print("TUM Dataset successfully generated!")

def setup_cli():
    parser = argparse.ArgumentParser(description="ROS2 RGB-D-Pose Extractor & Streamer")
    parser.add_argument("bags", nargs="+", help="ROS2 bag files/folders")
    parser.add_argument("--rgb", required=True, help="Master color topic")
    parser.add_argument("--depth", required=True, help="Slave depth topic")
    parser.add_argument("--pose", required=True, help="Slave pose topic")
    parser.add_argument("--info", required=True, help="Slave camera info topic")
    parser.add_argument("--max-delta", type=float, default=0.033)
    parser.add_argument("--sync-motion", action="store_true")
    parser.add_argument("--intrinsics", nargs=4, type=float, metavar=('FX', 'FY', 'CX', 'CY'))

    # Behavior flags
    parser.add_argument("--save-tum", action="store_true", help="Write outputs to TUM dataset format")
    parser.add_argument("--output", default="tum_dataset", help="Output directory if saving to TUM")

    return parser

def main():

    parser = setup_cli()
    args = parser.parse_args()

    # Initialize the engine
    streamer = ROS2RGBDStreamer(
        bag_paths=args.bags,
        rgb_topic=args.rgb,
        depth_topic=args.depth,
        pose_topic=args.pose,
        camera_info_topic=args.info,
        intrinsics=args.intrinsics,
        max_delta=args.max_delta,
        sync_motion=args.sync_motion
    )

    if args.save_tum:
        streamer.save_tum_dataset(args.output)
    else:
        # Example: Consuming the stream directly in another pipeline
        #   RH: Here we'd generally pipe over to a 3D reconstruction pipeline, but I want a unified CLI for our apps
        print("[INFO] Starting aligned stream (no disk writing)...")
        for i, (t, pose, rgb, depth) in enumerate(streamer.stream()):
            #status = f"Streamed Frame {i}: t={t:.3f} | RGB: {rgb.shape}"
            #if depth is not None: status += f" | Depth: {depth.shape}"
            #if pose is not None: status += f" | Pose: [x={pose[0]:.2f}, y={pose[1]:.2f}, z={pose[2]:.2f}]"
            #print(status, end="\r")
            pass
        print("\n[INFO] Stream complete.")

if __name__ == "__main__":
    main()
