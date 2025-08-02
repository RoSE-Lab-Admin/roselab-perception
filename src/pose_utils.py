from pathlib import Path
from dataclasses import asdict
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from rosbags.highlevel import AnyReader

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
#get_average_pose(Path("/home/ryan/lidarcalibrations/Trial_4cm_infradius_0.0slope_Trial3_07232025_10_37_30/mocap_bag"), "/CubeRover_V1/pose")