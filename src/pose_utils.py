from pathlib import Path
from dataclasses import asdict
from typing import List, Tuple

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib as mpl
import matplotlib.pyplot as plt

import cv2
from tqdm import tqdm
from rosbags.highlevel import AnyReader

# Useful lib for abstracting loading and manipulating pose streams!!!
import robotdatapy as rdp

def rotation_magnitude(rot):
    return np.linalg.norm(R.from_matrix(rot).as_rotvec())

def translation_magnitude(trans):
    return np.linalg.norm(trans)

# SEP arg controls whether this func returns combined or R,t error components
def pose_error(p1, p2, alpha=0.5, beta=0.5, sep=False):
    dp = np.linalg.inv(p1) @ p2
    if sep:
        return rotation_magnitude(dp[:3,:3]), translation_magnitude(dp[:3,3])

    else:
        normed = np.array([alpha, beta])
        normed /= np.linalg.norm(normed)

        return normed[0]*rotation_magnitude(dp[:3,:3]) + normed[1]*translation_magnitude(dp[:3,3])

def are_quaternions_close(q1: np.ndarray, q2: np.ndarray) -> bool:
    """Check if two quaternions represent the same rotation direction."""
    return np.dot(q1, q2) >= 0.0

def inverse_sign_quaternion(q: np.ndarray) -> np.ndarray:
    return -q

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)

def average_quaternion(cumulative: np.ndarray, new_q: np.ndarray, first_q: np.ndarray, count: int) -> np.ndarray:
    """
    Incrementally averages quaternions using a cumulative sum and normalizing the result.

    Args:
        cumulative: np.ndarray of shape (4,), running sum of quaternions.
        new_q: np.ndarray of shape (4,), new quaternion to add.
        first_q: np.ndarray of shape (4,), first quaternion used for sign reference.
        count: int, number of quaternions added including this one.

    Returns:
        np.ndarray: normalized average quaternion of shape (4,)
    """
    # Since we expect them to be close, any non-close ones are due to the double-cover problem
    if not are_quaternions_close(new_q, first_q):
        new_q = inverse_sign_quaternion(new_q)

    cumulative += new_q
    average = cumulative / count
    return normalize_quaternion(average)


def get_average_pose(bag_path: str, topic:str):
    """
    Given a bag path, returns the average postiion and orientation
    Args:
    bag_path: string path to bag
    topic: string, topic name (unused currently)

    Returns:
    avg_position:np.array(3,1), avg_orientation:np.array(4,1) 
    """
    bag_path = Path(bag_path)
    all_data = []
    # Extract poses
    with AnyReader([bag_path]) as reader:
        for conn in reader.connections:
            rows = []
            desc = f"{bag_path.name}:{conn.topic}"
            for _, ts, raw in tqdm(reader.messages(connections=[conn]),
                                   total=conn.msgcount, desc=desc):
                msg = reader.deserialize(raw, conn.msgtype)
                row = {'stamp_ns': ts}
                row.update(pd.json_normalize(asdict(msg)).iloc[0].to_dict())
                rows.append(row)
            all_data.append(rows)

    # Average pose
    pos_x_sum=0
    pos_y_sum=0
    pos_z_sum=0
    avg_orientation = np.zeros(4)
    first_q = None
    for conn in all_data: # Should only be one
        for i, frame in enumerate(conn, start=1):
            # Construct quat
            q = np.array([
                frame["pose.orientation.x"],
                frame["pose.orientation.y"],
                frame["pose.orientation.z"],
                frame["pose.orientation.w"]
            ])

            # Accum positions
            pos_x_sum += frame["pose.position.x"]
            pos_y_sum += frame["pose.position.y"]
            pos_z_sum += frame["pose.position.z"]

            # First quat
            if first_q is None:
                first_q = q.copy()

            # Accum orientation
            avg_orientation = average_quaternion(avg_orientation, q, first_q, i)
        avg_position = np.array([pos_x_sum/len(conn), pos_y_sum/len(conn), pos_z_sum/len(conn)])
    return avg_position, avg_orientation

def load_trajectory(bag_path: str, topic:str):
    """
    Given a bag path, returns time samples and corresponding transforms as numpy arrays
    Args:
    bag_path: string path to bag
    topic: string, topic name (unused currently)

    Returns:
    times:np.array(N,1), transforms:np.array(N,4,4)
    """
    bag_path = Path(bag_path)
    all_data = []
    N = 0
    # Extract poses
    with AnyReader([bag_path]) as reader:
        for conn in reader.connections:
            if conn.topic != topic:
                # If topic is not the topic specified, skip
                continue

            rows = []
            desc = f"{bag_path.name}:{conn.topic}"
            N = conn.msgcount
            for _, ts, raw in tqdm(reader.messages(connections=[conn]),
                                   total=conn.msgcount, desc=desc, mininterval=1.0):
                msg = reader.deserialize(raw, conn.msgtype)
                row = {'stamp_ns': ts}
                row.update(pd.json_normalize(asdict(msg)).iloc[0].to_dict())
                rows.append(row)
            all_data.append(rows)

    # Convert to numpy matrices
    tfs = np.tile(np.eye(4), (N,1,1))
    times = np.zeros(N)

    print(f"Processing {N} transforms...")
    for conn in all_data:
        for i, frame in enumerate(tqdm(conn)):
            tfs[i,:3,:3] = R.from_quat([
                frame["pose.orientation.x"],
                frame["pose.orientation.y"],
                frame["pose.orientation.z"],
                frame["pose.orientation.w"]
            ]).as_matrix()

            tfs[i,:3,3] = np.r_[
                frame["pose.position.x"],
                frame["pose.position.y"],
                frame["pose.position.z"]
            ]

            times[i] = frame["stamp_ns"]

    # Return 4x4 transforms extracted from pose message stream
    return np.asarray(times), np.asarray(tfs)


def plot_trajectory(times, tfs, show_up=False, show_frames=False, up='Z'):
    if show_up:
        # Show UP vector, so don't show full ref frame
        show_frames = False

        # Now decide which way is up
        if up=='Z':
            up_vec = np.array([0,0,1])
        elif up=='Y':
            up_vec = np.array([0,1,0])
        elif up=='X':
            up_vec = np.array([1,0,0])
        elif isinstance(up, [np.array, list]) and len(up)==3 and np.isclose(np.linalg.norm(up), 1.):
            up_vec = np.asarray(up)
        else:
            raise ValueError(f"Unsupported specification for 'up' vector: {up}")

    # For now, let's just support position for viz
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')

    # Normalize to [0,1]
    t = (tmp:= (times - times.min())) / tmp.max()

    # This could be changed to a line plot given nans for "missing" tfs
    ax.scatter3D(tfs[:,0,3], tfs[:,1,3], tfs[:,2,3], c=t, cmap='plasma', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')

    plt.show()

def plot_pose_errors(times, transforms, sep=True):
    if sep:
        errs = np.zeros(2*(len(times)-1)).reshape(len(times)-1,2)
        for i,(p1,p2) in enumerate(zip(transforms[0:-1], transforms[1:])):
            errs[i,:] = pose_error(p1,p2,sep=True)

        # Plot on twin axes
        fig, axl = plt.subplots()

        axl.scatter(times[1:], errs[:,0], color='tab:red') # , label="Rotational Error") # err(R)
        axl.set_ylabel("Rotational Error")
        axl.tick_params(axis='y', labelcolor='tab:red')

        axr = axl.twinx()
        axr.scatter(times[1:], errs[:,1]) # , label="Translation Error") # err(t)
        axr.set_ylabel("Translational Error", color='tab:blue')
        axr.tick_params(axis='y', labelcolor='tab:blue')

        plt.xlabel("Time $[s]$")

    else:
        errs = np.zeros(len(times)-1)
        for i,(p1,p2) in enumerate(zip(transforms[0:-1], transforms[1:])):
            errs[i] = pose_error(p1,p2)
        plt.xlabel("Time $[s]$")
        plt.ylabel("Total Pose Error")
        plt.plot(times[1:], errs)

    plt.show()

#get_average_pose(Path("/home/ryan/lidarcalibrations/Trial_4cm_infradius_0.0slope_Trial3_07232025_10_37_30/mocap_bag"), "/CubeRover_V1/pose")

if __name__=="__main__":
    # Load pose bag
    import sys
    times, transforms = load_trajectory(sys.argv[1], sys.argv[2])

    # Visualize
    print(times[0::1000])
    print(transforms[0::1000])

    print("TFs shape: ", transforms.shape)

    plot_trajectory(times, transforms, up='Y')

    # Make these times relative to t0 and convert to seconds
    plot_pose_errors((times - times[0]) / 1e9, transforms, sep=True)


    # RDP Tester
    bag_path = sys.argv[1] # path to bag
    topic = sys.argv[2] # Odometry or Pose msg

    # Is this lazily loaded? Or all at once?
    pose_data = rdp.data.PoseData.from_bag(bag_path, topic=topic, time_tol=25.0, interp=True)
    print(pose_data)

    # Make a version of my plot pretty much
    pose_data.plot2d(dt=0.1, trajectory=True, pose=False)     # plots only position every second
    pose_data.plot2d(dt=20.0, trajectory=False, pose=True, axis_len=0.25)     # plots coordinate frames of the poses every 5 seconds

    plt.show()
