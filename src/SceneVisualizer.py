import open3d as o3d
import numpy as np
import os
import copy
from scipy.spatial.transform import Rotation as R
from typing import Optional

def _parse_metadata(fname):
    """
    Parse a metadata CSV to extract camera positions and orientations.
    Args:
        fname (str): Path to the metadata file from motive
    Returns:
        tuple of np.ndarray: (positions, rotations)
            positions: N×3 array of [x, y, z] floats.
            rotations: N×4 array of [x, y, z, w] quaternions.
    """
    sensors_pos = []
    sensors_rot = []

    with open(fname, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] == 'Camera':
                # parse position directly as x, y, z
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                sensors_pos.append([x, y, z])

                # parse quaternion directly as [x, y, z, w]
                quat = [float(q) for q in parts[5:9]]
                sensors_rot.append(quat)

    return np.asarray(sensors_pos), np.asarray(sensors_rot)


class PoseObject:
    """
    Base class for objects with a 4×4 pose.
    """
    def __init__(self, pose: np.ndarray):
        if pose.shape != (4, 4):
            raise ValueError("Pose must be a 4×4 transformation matrix.")
        self.pose = pose

    def get_geometry(self) -> o3d.geometry.Geometry:
        """
        Return the Open3D geometry at the origin (untransformed).
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_transformed_geometry(self) -> o3d.geometry.Geometry:
        geom = self.get_geometry()
        geom = copy.deepcopy(geom)
        geom.transform(self.pose)
        return geom


class SimplePose(PoseObject):
    """
    A coordinate-frame axes at the given pose.
    """
    def __init__(self, pose: np.ndarray, size: float = 0.5):
        super().__init__(pose)
        self.size = size

    def get_geometry(self) -> o3d.geometry.TriangleMesh:
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.size)


class CameraPose(PoseObject):
    """
    A camera frustum represented as a LineSet, positioned by pose.
    """
    def __init__(self,
                 pose: np.ndarray,
                 scale: float = 1.0,
                 color: tuple = (1.0, 0.0, 0.0)):
        super().__init__(pose)
        self.scale = scale
        self.color = color

    def get_geometry(self) -> o3d.geometry.LineSet:
        # Define frustum: apex at origin, base square at z = scale
        apex = np.array([0.0, 0.0, 0.0])
        half = self.scale
        corners = np.array([
            [ half,  half,  half],
            [ half, -half,  half],
            [-half, -half,  half],
            [-half,  half,  half]
        ])
        pts = np.vstack([apex, corners])
        # Lines: apex to corners, and perimeter of base
        lines = [[0, i] for i in range(1, 5)] + [[1,2], [2,3], [3,4], [4,1]]
        colors = [self.color for _ in lines]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines)
        )
        ls.colors = o3d.utility.Vector3dVector(colors)
        return ls



class PointCloudPose(PoseObject):
    """
    A point cloud with an associated pose.
    """
    def __init__(self, file_path: str, pose: Optional[np.ndarray] = None):
        if pose is None:
            pose = np.eye(4)
        super().__init__(pose)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud not found: {file_path}")
        self.pcd = o3d.io.read_point_cloud(file_path)
        if self.pcd.is_empty():
            print(f"Warning: Loaded empty point cloud: {file_path}")

    def get_geometry(self) -> o3d.geometry.PointCloud:
        return self.pcd


class SceneVisualizer:
    """
    Collects PoseObjects and PointClouds, then visualizes them together.
    """
    def __init__(self):
        self.objects = []

    def add(self, obj):
        """
        Add a PoseObject or PointCloud to the scene.
        """
        self.objects.append(obj)

    def visualize(self,
                  window_name: str = 'Scene',
                  width: int = 800,
                  height: int = 800,
                  mesh_show_back_face: bool = True):
        geometries = []
        for obj in self.objects:
            if isinstance(obj, PoseObject):
                geometries.append(obj.get_transformed_geometry())
            elif isinstance(obj, PointCloud):
                geometries.append(obj.get_geometry())
            else:
                raise TypeError(f"Unsupported object type: {type(obj)}")

        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=width,
            height=height,
            mesh_show_back_face=mesh_show_back_face
        )

if __name__ == '__main__':
    # Instantiate scene visualizer
    scene = SceneVisualizer()

    # Center of the floor at origin
    floor_pose = np.eye(4)
    scene.add(SimplePose(floor_pose, size=0.5))
   
    # Parse OptiTrack metadata and add those cameras
    metadata_file = 'data/test.csv'
    positions, rotations = _parse_metadata(metadata_file)
    flip_z = np.diag([1,1,-1])
    for pos, quat in zip(positions, rotations):
        pose = np.eye(4)
        # position in meters
        pose[0:3, 3] = pos
        # original rotation
        rot = R.from_quat(quat).as_matrix()
        # apply flip to face frustum inward
        rot_fixed = rot.dot(flip_z)
        pose[0:3, 0:3] = rot_fixed
        scene.add(CameraPose(pose, scale=0.5, color=(0, 0, 1)))


    # Lidar
    lidar_pose = np.eye(4)
    # translate to (0,0,5)
    lidar_pose[0:3, 3] = np.array([0, 2.1, 0])

    # rotate +90° about X so that +Z maps to -Y
    lidar_rot = R.from_euler('x', 90, degrees=True).as_matrix()
    lidar_pose[0:3, 0:3] = lidar_rot
    scene.add(CameraPose(lidar_pose, scale=0.5, color=(1, 0, 0)))

    # Single point cloud with same transform as LiDAR
    pointcloud_pose = lidar_pose
    scene.add(PointCloudPose('data/out.ply', pose=pointcloud_pose))

    # Render the scene
    scene.visualize(window_name='MLSS Sensor Poses', width=1024, height=768)
