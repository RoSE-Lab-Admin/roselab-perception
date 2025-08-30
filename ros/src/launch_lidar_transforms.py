import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import subprocess

def load_tf_args_from_yaml(yaml_path, child_frame):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    T = np.array(data['Transform']['T']).flatten()
    R_matrix = np.array(data['Transform']['R'])
    rot = R.from_matrix(R_matrix)
    yaw, pitch, roll = rot.as_euler('xyz', degrees=False)

    return [*T, roll, pitch, yaw, 'base_link', child_frame]

def launch_static_tf(args):
    print("Launching static transform:")
    print("ros2 run tf2_ros static_transform_publisher", " ".join(map(str, args)))
    subprocess.Popen(['ros2', 'run', 'tf2_ros', 'static_transform_publisher', *map(str, args)])

def main():
    # Identity transform for center
    launch_static_tf([0, 0, 0, 0, 0, 0, 'base_link', 'l515_center_depth_optical_frame'])

    # Load east and west from YAML
    east_args = load_tf_args_from_yaml('L515_East_properties.yaml', 'l515_east_depth_optical_frame')
    west_args = load_tf_args_from_yaml('L515_West_properties.yaml', 'l515_west_depth_optical_frame')
    #center_args = load_tf_args_from_yaml('L515_Center_properties.yaml', 'l515_center_depth_optical_frame')

    launch_static_tf(east_args)
    launch_static_tf(west_args)
    #launch_static_tf(center_args)

if __name__ == '__main__':
    main()
#Colored ICP