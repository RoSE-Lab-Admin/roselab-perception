import open3d as o3d
import numpy as np
import os
import copy
from scipy.spatial.transform import Rotation as R


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
    def __init__(self, pose: np.ndarray, scale: float = 0.1, color=(1.0, 0.0, 0.0)):
        super().__init__(pose)
        self.scale = scale
        self.color = color

    def get_geometry(self) -> o3d.geometry.LineSet:
        apex = np.array([0.0, 0.0, 0.0])
        corners = np.array([
            [ self.scale,  self.scale,  self.scale],
            [ self.scale, -self.scale,  self.scale],
            [-self.scale, -self.scale,  self.scale],
            [-self.scale,  self.scale,  self.scale]
        ])
        points = np.vstack([apex, corners])

        # Lines: camera center to each corner, plus base edges
        lines = [[0, i] for i in range(1, 5)] + [[1, 2], [2, 3], [3, 4], [4, 1]]
        colors = [list(self.color) for _ in lines]

        frustum = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        frustum.colors = o3d.utility.Vector3dVector(colors)
        return frustum


class PointCloud:
    """
    Wrapper for loading and storing a point cloud.
    """
    def __init__(self, file_path: str):
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

    # def visualize(self,
    #               window_name: str = 'Scene',
    #               width: int = 800,
    #               height: int = 600,
    #               mesh_show_back_face: bool = True):
    #     """
    #     Collect PoseObjects and PointClouds, add them to a custom Visualizer,
    #     then set up camera to view the combined scene bounding box.
    #     """
    #     # Collect all geometries
    #     geometries = []
    #     for obj in self.objects:
    #         if isinstance(obj, PoseObject):
    #             geometries.append(obj.get_transformed_geometry())
    #         elif isinstance(obj, PointCloud):
    #             geometries.append(obj.get_geometry())
    #         else:
    #             raise TypeError(f"Unsupported object type: {type(obj)}")

    #     # Initialize Visualizer
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window(window_name=window_name, width=width, height=height)

    #     # Add each geometry
    #     for geom in geometries:
    #         vis.add_geometry(geom)

    #     # Compute combined bounding box to center the view
    #     bboxes = [geom.get_axis_aligned_bounding_box() for geom in geometries]
    #     combined_bbox = bboxes[0]
    #     for bb in bboxes[1:]:
    #         combined_bbox += bb
    #     center = combined_bbox.get_center()

    #     # Configure view control
    #     ctr = vis.get_view_control()
    #     ctr.set_lookat(center)
    #     # Camera looks toward negative y by default
    #     ctr.set_front([0.0, -1.0, 0.0])
    #     ctr.set_up([0.0, 0.0, 1.0])
    #     # Adjust zoom: smaller value zooms out
    #     ctr.set_zoom(0.2)
    #     # Adjust clipping planes
    #     ctr.set_constant_z_near(0.00001)
    #     ctr.set_constant_z_far(100000.0)

    #     # Render options
    #     render_opt = vis.get_render_option()
    #     render_opt.mesh_show_back_face = mesh_show_back_face
    #     render_opt.line_width = 2.0
    #     render_opt.point_size = 2.0

    #     # Run
    #     vis.run()
    #     vis.destroy_window()

if __name__ == '__main__':
    # Instantiate scene visualizer
    scene = SceneVisualizer()

    # Center of the floor at origin
    floor_pose = np.eye(4)
    scene.add(SimplePose(floor_pose, size=1.0))

    # Cube half-size
    h = 5.0

    # Parse metadata for the four OptiTrack cameras
    metadata_file = 'test.csv'
    positions, rotations = _parse_metadata(metadata_file)
    # Add a CameraPose for each parsed sensor
    for pos, quat in zip(positions, rotations):
        pose_mat = np.eye(4)
        pose_mat[0:3, 3] = pos
        rot_mat = R.from_quat(quat).as_matrix()
        pose_mat[0:3, 0:3] = rot_mat
        scene.add(CameraPose(pose_mat, scale=1.5, color=(0, 0, 1)))


    # Lidar
    lidar_pose = np.eye(4)
    # translate to (0,0,5)
    lidar_pose[0:3, 3] = np.array([0, 0, 5])
    # rotate 180° about X to point down
    lidar_pose[1,1] = -1
    lidar_pose[2,2] = -1
    scene.add(CameraPose(lidar_pose, scale=1.5, color=(1, 0, 0)))

    # Add single point cloud
    #scene.add(PointCloud('scene_cloud.ply'))

    # Render the scene
    scene.visualize(window_name='MLSS Sensor Poses', width=1024, height=768)
