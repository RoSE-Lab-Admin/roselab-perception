import numpy as np
from scipy.spatial.transform import Rotation as R
from rosbags.highlevel import AnyReader
from pathlib import Path

def get_average_gravity(bag_path, topic_name):
    accels = []
    with AnyReader([bag_path]) as reader:
        # Filter for the specific IMU topic
        connections = [x for x in reader.connections if x.topic == topic_name]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            accels.append([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

    # Return the normalized average vector
    avg = np.mean(accels, axis=0)
    return avg / np.linalg.norm(avg)

def align_vectors(source_vec, target_vec):
    """
    Computes the rotation matrix to align source_vec to target_vec
    using the Rodrigues' rotation formula logic via SciPy.
    """
    # Calculate the cross product (rotation axis) and dot product (angle)
    cross = np.cross(source_vec, target_vec)
    dot = np.dot(source_vec, target_vec)
    
    # Create the rotation object
    # If vectors are parallel, the cross product is zero
    if np.linalg.norm(cross) < 1e-6:
        return np.eye(3)
    
    # We find the rotation vector (axis * angle)
    # angle = arccos(dot / (norm_a * norm_b))
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    rotation_axis = cross / np.linalg.norm(cross)
    
    return R.from_rotvec(rotation_axis * angle)

if __name__=="__main__":
    import sys

    # --- Main Execution ---
    bag_file = Path(sys.argv[1])
    topic_center = sys.argv[2]
    topic_side = sys.argv[3]
    center_g = get_average_gravity(bag_file, topic_center)
    side_g = get_average_gravity(bag_file, topic_side)

    print("Center Lidar Gravity Vec: ", center_g)

    # Calculate the alignment rotation
    rotation_to_center = align_vectors(side_g, center_g)

    print("Rotation Magnitude: ", np.rad2deg(np.linalg.norm(rotation_to_center.as_rotvec())), " [deg]")
    print("Rotation Matrix (Side -> Center Alignment):")
    print(rotation_to_center.as_matrix())
    print(f"\nEuler angles (degrees): {rotation_to_center.as_euler('xyz', degrees=True)}")
