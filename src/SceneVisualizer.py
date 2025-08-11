import open3d as o3d
import numpy as np
import os
import copy
from scipy.spatial.transform import Rotation as R
from typing import Optional
import sys

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


def create_cylinder_between_points(p1, p2, radius=0.02, color=[1, 0, 0]):
    """
    Create a cylinder from p1 to p2 with specified radius and color.
    """
    cylinder_height = np.linalg.norm(p2 - p1)
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=cylinder_height, resolution=20, split=4)
    cylinder.paint_uniform_color(color)
    cylinder.compute_vertex_normals()

    # Align the cylinder along the vector (p2 - p1)
    direction = (p2 - p1) / cylinder_height
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction)
    angle = np.arccos(np.dot(z_axis, direction))
    if np.linalg.norm(axis) != 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle)
        cylinder.rotate(R, center=np.zeros(3))

    # Translate to midpoint
    midpoint = (p1 + p2) / 2
    cylinder.translate(midpoint)

    return cylinder

def create_dimension_cylinders(bbox, tf=None, thickness=0.02):
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    center = bbox.get_center()

    # Compute extents manually
    extent = max_bound - min_bound
    x_extent = extent[0]
    y_extent = extent[1]
    z_extent = extent[2]

    # Axis endpoints
    x0 = center - np.array([x_extent / 2, 0, 0])
    x1 = center + np.array([x_extent / 2, 0, 0])

    y0 = center - np.array([0, y_extent / 2, 0])
    y1 = center + np.array([0, y_extent / 2, 0])

    z0 = center - np.array([0, 0, z_extent / 2])
    z1 = center + np.array([0, 0, z_extent / 2])

    # Create cylinders for each axis
    x_cyl = create_cylinder_between_points(x0, x1, radius=thickness, color=[1, 0, 0])  # Red for X
    y_cyl = create_cylinder_between_points(y0, y1, radius=thickness, color=[0, 1, 0])  # Green for Y
    z_cyl = create_cylinder_between_points(z0, z1, radius=thickness, color=[0, 0, 1])  # Blue for Z

    if tf is not None:
        return [cyl.transform(tf) for cyl in [x_cyl, y_cyl, z_cyl]]

    return [x_cyl, y_cyl, z_cyl]


def create_dimension_lines(bbox):
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    center = bbox.get_center()

    # Define axis directions
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    # Lengths
    width = max_bound[0] - min_bound[0]
    length = max_bound[1] - min_bound[1]
    height = max_bound[2] - min_bound[2]

    # Starting point is the center of the bounding box
    origin = center

    # Compute endpoints for the lines
    x_end = origin + width / 2 * x_axis
    x_start = origin - width / 2 * x_axis

    y_end = origin + length / 2 * y_axis
    y_start = origin - length / 2 * y_axis

    z_end = origin + height / 2 * z_axis
    z_start = origin - height / 2 * z_axis

    # Combine into points and lines
    points = [
        x_start, x_end,
        y_start, y_end,
        z_start, z_end
    ]
    lines = [
        [0, 1],  # X axis (Width)
        [2, 3],  # Y axis (Length)
        [4, 5],  # Z axis (Height)
    ]
    colors = [
        [1, 0, 0],  # Red for width (x)
        [0, 1, 0],  # Green for length (y)
        [0, 0, 1],  # Blue for height (z)
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.size = o3d.utility.Scalar()

    return line_set

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

class ObjectPose(PoseObject):
    """
    Visualize an STL
    """
    def __init__(self, pose: np.ndarray, path: str, scale: float = 1.0):
        super().__init__(pose)
        self.path = path
        self.scale = scale

    def get_geometry(self) -> o3d.geometry.TriangleMesh:

        mesh = o3d.io.read_triangle_mesh(self.path)
        mesh.compute_vertex_normals()
        # 1) compute its centroid
        center = mesh.get_center()
        # 2) shift it so centroid is at (0,0,0)
        mesh.translate(-center)
        # 3) scale about that new local origin
        mesh.scale(self.scale, center=np.zeros(3))
        # 4) now apply your extrinsic pose
        mesh.transform(self.pose)
        return mesh

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

    def add(self, obj_list):
        """
        Add a PoseObject or PointCloud to the scene.
        """
        self.objects.extend(obj_list)

    def visualize(self,
                  window_name: str = 'Scene',
                  width: int = 800,
                  height: int = 800,
                  mesh_show_back_face: bool = True):
        geometries = []
        for obj in self.objects:
            if isinstance(obj, PoseObject):
                geometries.append(obj.get_transformed_geometry())
            elif isinstance(obj, PointCloudPose):
                geometries.append(obj.get_geometry())
            elif isinstance(obj, o3d.geometry.Geometry):
                geometries.append(obj)
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

    if(len(sys.argv) != 4):
        print("Usage: python SceneVisualizer.py <path to pointcloud ply> <path to STL> <path to optitrack metadata csv>")
        quit()
    pointcloud_path = sys.argv[1]
    stl_path = sys.argv[2]
    metadata_file = sys.argv[3]

    # Instantiate scene visualizer
    scene = SceneVisualizer()

    # Center of the floor at origin
    floor_pose = np.eye(4)

    # build a transform to place axes at corner of point cloud
    rx = np.deg2rad(-90)
    ry = np.deg2rad(45)
    rz = np.deg2rad(0)
    center_to_corner = np.array([-1.47,-0.65,-0.5])
    R_floor = R.from_euler('xyz', [rx,ry,rz])
    floor_pose[:3,:3] = R_floor.as_matrix()
    floor_pose[:3, 3] = center_to_corner
    scene.add([SimplePose(floor_pose, size=1.0)])

    # Parse OptiTrack metadata and add those cameras
#    metadata_file = 'data/test.csv'
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
        scene.add([CameraPose(pose, scale=0.5, color=(0, 0, 1))])
#        scene.add([SimplePose(pose, size=0.5)])

    # build a rotation about Y
    theta = np.deg2rad(-9)
    Ry = np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [            0, 1,            0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    # Lidar
    lidar_pose = np.eye(4)
    lidar_pose[0:3, 0] = np.array([0.80778813 , 0.01618329, 0.58925074 ])
    lidar_pose[0:3, 1] = np.array([-0.58921483, -0.00741073, 0.80794243 ])
    lidar_pose[0:3, 2] = np.array([ 0.01744194, -0.99984158 , 0.00354913  ])
    lidar_pose[0:3, 3] = np.array([0.6936598, 2.54371088, 0.33487012])
    lidar_pose[:3, :3] = Ry @ lidar_pose[:3, :3]

    lidar_rot =  R.from_euler('xyz', [0,np.pi,0]) * R.from_matrix(lidar_pose[:3,:3])
    lidar_pose[:3,:3] = lidar_rot.as_matrix()

    scene.add([CameraPose(lidar_pose, scale=0.5, color=(1, 0, 0))])
    scene.add([SimplePose(lidar_pose, size=1.0)])

    # Single point cloud with same transform as LiDAR
    pointcloud_pose = np.eye(4)
    pointcloud_pose[0:3, 3] = np.array([0.33487012, 0.0, 0.6936598])
    pointcloud_pose[:3, :3] = Ry @ pointcloud_pose[:3, :3]
    #pointcloud_pose[:3,  3 ] = Ry @ pointcloud_pose[:3,  3 ]
    scene.add([PointCloudPose(pointcloud_path, pose=pointcloud_pose)])

    # Rover reference frame
    rover_pose = np.eye(4)
    rx = np.deg2rad(-90)
    ry = np.deg2rad(45+90)
    rz = np.deg2rad(0)
    center_to_rover = np.array([1.29,-0.35,-0.12])
    R_rover = R.from_euler('xyz', [rx,ry,rz])
    rover_pose[:3,:3] = R_rover.as_matrix()
    rover_pose[:3, 3] = center_to_rover
    scene.add([SimplePose(rover_pose, size=1.0)])

    # STL of Gantry
    gantry_pose = np.eye(4)
    gantry_pose[0:3, 0] = np.array([ 0.9238795,  0.0,  -0.3826834])
    gantry_pose[0:3, 1] = np.array([0.0,  1.0,  0.0])
    gantry_pose[0:3, 2] = np.array([ 0.3826834,  0.0,  0.9238795])
    gantry_pose[0:3, 3] = np.array([0.3, 1.5, 0.2])
    
    gantry_geom = ObjectPose(gantry_pose, stl_path, 0.001)

#    bbox = gantry_geom.get_geometry().get_axis_aligned_bounding_box()
#    dims = create_dimension_cylinders(bbox, tf=floor_pose, thickness=0.05)

    # Hardcode gantry geometry since its bounding box is uhhhhhh weird
    # X dimension (same, width, East-West)
    rot = R.from_euler('y', np.pi/4).as_matrix()

    xdim = create_cylinder_between_points(rot @ np.array([-5.2,-1.5,6]), rot @ np.array([5.2,-1.5,6]), radius=0.02, color=[1, 0, 0])

    # Y dimension (former Z, length, North-South)
    ydim = create_cylinder_between_points(rot @ np.array([-6,-1.5,-5.2]), rot @ np.array([-6,-1.5,5.2]), radius=0.02, color=[0, 1, 0])

    # Z dimension (former Y, height)
    zdim = create_cylinder_between_points(np.array([-8.4,-1.5, 0]), np.array([-8.4,4.1, 0]), radius=0.02, color=[0, 0, 1])

    dims = [xdim, ydim, zdim]
    scene.add([gantry_geom] + dims)

    # Render the scene
    scene.visualize(window_name='MLSS Sensor Poses', width=1024, height=768)

#
#{
#	"class_name" : "ViewTrajectory",
#	"interval" : 29,
#	"is_loop" : false,
#	"trajectory" : 
#	[
#		{
#			"boundingbox_max" : [ 7.7568800406736402, 3.5627625872781534, 7.3948479728601413 ],
#			"boundingbox_min" : [ -7.3420793491526677, -1.2459973767111041, -7.8903603184415791 ],
#			"field_of_view" : 60.0,
#			"front" : [ -0.31522538429328806, 0.62961532773836182, 0.7100827389636114 ],
#			"lookat" : [ -0.29711469313404348, -0.56852788619118033, 0.77594232354844539 ],
#			"up" : [ 0.23290600086368904, 0.77667201987550472, -0.58526521193744874 ],
#			"zoom" : 0.47999999999999976
#		}
#	],
#	"version_major" : 1,
#	"version_minor" : 0
#}
