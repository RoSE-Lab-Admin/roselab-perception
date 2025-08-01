# Load Bag, and fuse with time probably
# Color match the plate (research color matching)
# Plane fit the plate  (research plane fitting)
# Grab center (or origin) of plate, store as origin
# Get normal of plane, store as Y
# Determine orientation of x, z
# Store all 
# Get transaltion based on camera frame (examples)
# Make translation matrix
# Rotation based on slope of plane
# Make rotation matrix
# Return both

# ROS
import rclpy
import rosbag2_py

# RH: Converting workflow to use rosbag2_py and cv_bridge
#from rosbags.highlevel import AnyReader
#from rosbags.image import message_to_cvimage

from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# Math and Vision
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# System/Misc
from pathlib import Path
import time
import threading
from tqdm import tqdm
import yaml

class OriginFusion():
    def __init__(self):
        self.color_images = []
        self.depth_images = []
        self.caminfo = None

    def LoadBag(self, bag_path, color_topic, depth_topic, caminfo_topic):
        bridge = CvBridge()

        # Initialize reader
        storage_options = StorageOptions(uri=bag_path, storage_id='mcap')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

        with open(Path(bag_path) / "metadata.yaml", 'r') as md:
            metadata = yaml.safe_load(md)
            print(metadata)

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        type_map = {}
        for topic_metadata in reader.get_all_topics_and_types():
            type_map[topic_metadata.name] = topic_metadata.type

        print(type_map)

        msg_max = max([int(x["message_count"]) for x in metadata["rosbag2_bagfile_information"]["topics_with_message_count"]])

        pbar = tqdm(total=msg_max, desc="Processing")

        # Read messages in loop and convert with cv_bridge
        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            pbar.update(1)

            if topic not in [color_topic, depth_topic, caminfo_topic]:
                continue

            # Deserialize message
            msg_type = Image if type_map[topic] == 'sensor_msgs/msg/Image' else CameraInfo
            if not msg_type:
                continue

            msg = deserialize_message(data, msg_type)

            # Convert to OpenCV image
            if topic == color_topic:
                self.color_images.append(bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))

            elif topic == depth_topic:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if(img.shape == (360,640)):
                    self.depth_images.append(img)

            elif topic == caminfo_topic:
                self.caminfo = msg

#        key = cv2.waitKey(1)
#        if key == ord('q'):
#            break

        pbar.close()

        self.stack_task()
        self.compute_pointcloud_from_rgbd()

    def stack_task(self):
        # Make stacked and median images
        stacked_color = np.stack(self.color_images, axis=0)
        self.median_color_img = np.median(stacked_color, axis=0).astype(np.uint8)
        stacked_depth = np.stack(self.depth_images, axis=0)
        self.median_depth_img = np.median(stacked_depth, axis=0).astype(np.uint16)
        #cv2.imwrite("/mnt/c/Users/ryan1/Downloads/img.tiff", self.median_depth_img)

    def compute_pointcloud_manually(self):
        # Do the tried and true point by point method where we can generate from a masked rgbd pair
        pass

    def compute_pointcloud_from_rgbd(self):
        # Use median images to generate points and associated colors (1:1 pixels) via Open3D
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(self.median_color_img), o3d.geometry.Image(self.median_depth_img), depth_scale=1000., depth_trunc=5.0)
        self.median_pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(self.median_color_img.shape[1], self.median_color_img.shape[0], np.asarray(self.caminfo.k).reshape(3,3)))

        # o3d.visualization.draw_geometries([self.median_pointcloud])
        try:
            o3d.io.write_point_cloud("out.pcd", self.median_pointcloud)
        except:
            print("Could not write out point cloud... Does a file called 'out.pcd' already exist?")
            pass

    def PlotImages(self):
        # Plot median images
        fig, axes=plt.subplots(1,2)
        axes[0].imshow(self.median_color_img)
        axes[1].imshow(self.median_depth_img, cmap="inferno") #, vmin=2300.0, vmax=2400.0)
        plt.show()

    def get_lidar_to_world_tf(self, reference="spot_charuco_board", visualize=True):
        if reference.lower() == 'emma_fusion_target':
            # Get Arcuo
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            params = cv2.aruco.DetectorParameters()

            # These parameters consistently find our aruco target, but perhaps my dictionary is incorrect???
            params.adaptiveThreshConstant = 6
            params.minMarkerPerimeterRate = 0.02
            params.maxMarkerPerimeterRate = 3.0
            params.polygonalApproxAccuracyRate = 0.07
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Improves accuracy

            # Improve contrast
            tmp = self.median_color_img.copy()
            tmp = cv2.resize(tmp, None, fx=2., fy=2.)

    #        gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    #        gray = cv2.GaussianBlur(gray, (3,3), 0)
    #        gray = cv2.equalizeHist(gray)
    #        gray = cv2.resize(gray, None, fx=2., fy=2.)

            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    #        corners, ids, rejected_candidates = detector.detectMarkers(gray) # use the contrast enhanced gray otherwise
            corners, ids, rejected_candidates = detector.detectMarkers(tmp) # use the contrast enhanced gray otherwise

    #        print(corners)
            print("Detected Aruco Target IDs: ", ids)
    #        print(rejected_candidates)

            if visualize:
    #            cv2.aruco.drawDetectedMarkers(tmp, np.asarray(rejected_candidates)/2, borderColor=(100, 0, 255))
    #            cv2.aruco.drawDetectedMarkers(tmp, np.asarray(corners)/2, ids)

                # Useful for debugging
    #            cv2.aruco.drawDetectedMarkers(tmp, rejected_candidates, borderColor=(100, 0, 255))
                cv2.aruco.drawDetectedMarkers(tmp, corners, ids)
                plt.imshow(tmp)
                plt.show()

            # Aruco coordinate system pose estimation in Lidar frame
            camera_matrix = np.asarray(self.caminfo.k).reshape(3,3)
            distortion_coeffs = np.zeros(5)
            mhl = 0.100/2 # marker half-length in meters
            object_points = np.array([
                                         [-mhl,mhl,0],
                                         [mhl,mhl,0],
                                         [mhl,-mhl,0],
                                         [-mhl,-mhl,0]
                                     ]) # These are the corner points in the aruco target frame
    #        rvecs, tvecs, _objPoints = cv2.aruco.legacy.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, distortion_coeffs)

            # Need to use PnP solver to get ref frame using the modern API, and cv2.SOLVEPNP_IPPE_SQUARE
            ret, rvec, tvec = cv2.solvePnP(object_points, corners[0], camera_matrix, distortion_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            print("R: ", ARUCO_ROT_IN_LIDAR:=cv2.Rodrigues(rvec)[0], "\nt: ", tvec)

    #        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(tmp, camera_matrix, distortion_coeffs, rvec, tvec, 0.03)
            plt.imshow(tmp)
            plt.show()

            # Find Static TF for aruco in world frame (frame cad)
            ARUCO_ROT_IN_WORLD = R.from_euler('xyz', [np.pi/2,0,0]) # 90 deg rotation about x, as x is colinear across frames

    # 	Measurements for Cal plate translation relative to origin (world)
    #	x: -70mm
    #	z: 140mm
    #	y: 8mm up
            ARUCO_POS_IN_WORLD = np.array([-0.070,0.008,0.140]) # Measured from Emma's CAD model, origin of Optitrack Square to center of Aruco target

            ARUCO_POSE_IN_WORLD = np.eye(4) # ARUCO -> WORLD

            ARUCO_POSE_IN_WORLD[:3,:3] = ARUCO_ROT_IN_WORLD.as_matrix()
            ARUCO_POSE_IN_WORLD[:3, 3] = ARUCO_POS_IN_WORLD
            print("ARUCO TARGET POSE IN WORLD FRAME: \n\n", ARUCO_POSE_IN_WORLD)

            # Plane fit to get normal vector (Ryan H)

            # Given aruco target corners, make vector of corner indices and then use opencv's fillpoly routine to generate a mask
            mask = np.zeros_like(self.median_depth_img, np.uint8)
            cv2.fillPoly(mask, pts=(np.asarray(corners)/2).astype(int), color=(255))
            kernel = np.ones((3,3))
            mask_eroded = cv2.erode(mask,kernel,iterations = 2)
            mask = mask_eroded.astype(bool)
            print("Mask Info: ", mask.dtype, mask.shape, mask.sum() / mask.size * 100, "% Masked")

            plt.imshow(mask)
            plt.show()

            # Convert depth data to xyz points and apply PCA eigen vector definition as with my normal calculation in the surface char module
            print("# of PCD Points  : ", len(np.asarray(self.median_pointcloud.points)))

            print(len(np.where(mask.flatten())[0]))

    #        aruco_pointcloud = self.median_pointcloud.select_by_index(np.where(mask.flatten())[0])
    #        print("# of Aruco Points: ", len(np.asarray(aruco_pointcloud.points)))
    #        plane_model, inliers = aruco_pointcloud.segment_plane(distance_threshold=0.05,
    #                                         ransac_n=15,
    #                                         num_iterations=1000)
    #        [a, b, c, d] = plane_model
    #        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    #        aruco_depth_normal = np.asarray([a,b,c]) / np.linalg.norm([a,b,c])

    #        aruco_pc = self.median_pointcloud.select_by_index(np.where((mask.T).flat)[0])

            # O3D Hack for Masking data while generating pointcloud from RGBD data
            depth_masked = self.median_depth_img.copy()
            depth_masked[~mask] = -1.
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(self.median_color_img), o3d.geometry.Image(depth_masked), depth_scale=1000., depth_trunc=5.0)
            aruco_pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(self.median_color_img.shape[1], self.median_color_img.shape[0], np.asarray(self.caminfo.k).reshape(3,3)))

            aruco_depth_points = np.asarray(aruco_pc.points)
            print("Aruco Target Centroid in Lidar Frame :", np.mean(aruco_depth_points, axis=0))

            o3d.visualization.draw_geometries([aruco_pc])

            aruco_cov = np.cov(aruco_depth_points.T)
            eigvals, eigvecs = np.linalg.eigh(aruco_cov)
            aruco_depth_normal = eigvecs[:, 0]  # eigenvector with smallest eigenvalue

            print("Dot-Product Similarity (Depth Fit Normal vs. Aruco +Z): ", np.dot(aruco_depth_normal, ARUCO_ROT_IN_LIDAR[:,2]))

            # Evaluate error between normal vector and +y pose vector for good measure (dot product)
            # Build transformation matrix for aruco in lidar frame, aka (lidar)^T_(aruco)
            ARUCO_POSE_IN_LIDAR = np.eye(4) # ARUCO -> LIDAR
            ARUCO_POS_IN_LIDAR = np.mean(aruco_depth_points, axis=0)

            ARUCO_POSE_IN_LIDAR[:3,:3] = ARUCO_ROT_IN_LIDAR
            ARUCO_POSE_IN_LIDAR[:3, 3] = ARUCO_POS_IN_LIDAR
            print("ARUCO TARGET POSE IN LIDAR FRAME: \n\n", ARUCO_POSE_IN_LIDAR)

            # Compute inverse analytically
            LIDAR_POSE_IN_ARUCO = np.eye(4)
            LIDAR_POSE_IN_ARUCO[:3,:3] = ARUCO_ROT_IN_LIDAR.T
            LIDAR_POSE_IN_ARUCO[:3, 3] = (-ARUCO_ROT_IN_LIDAR.T @ np.c_[ARUCO_POS_IN_LIDAR]).T[:]

            # Finally, find: world^T_lidar = world^T_aruco * (lidar^T_aruco)^-1
            LIDAR_POSE_IN_WORLD = ARUCO_POSE_IN_WORLD @ LIDAR_POSE_IN_ARUCO

            print("LIDAR POSE IN WORLD (Lidar -> World) : \n\n", LIDAR_POSE_IN_WORLD)

        elif reference.lower() == "spot_charuco_board":
            # Get Arcuo

            # Define board parameters
#            num_checkers_width = 5
#            num_checkers_height = 7
#            checker_dim = 0.2  # Length of chessboard square side in meters
#            marker_dim = 0.146   # Length of ArUco marker side in meters

            num_checkers_width=9
            num_checkers_height=4
            checker_dim=0.115
            marker_dim=0.09
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

            # Create the ChArUco board
            charuco_board = cv2.aruco_CharucoBoard(
                (num_checkers_width, num_checkers_height),
                checker_dim,
                marker_dim,
                aruco_dict,
            )

            # Set the legacy pattern
            charuco_board.setLegacyPattern(True)
#
#            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
            # params = cv2.aruco.DetectorParameters()

            detector_params = cv2.aruco.CharucoParameters()
            detector_params.minMarkers = 0
            detector_params.tryRefineMarkers = True

            charuco_detector = cv2.aruco.CharucoDetector(charuco_board, detector_params)
            charuco_detector.setBoard(charuco_board)

            # These parameters consistently find our aruco target, but perhaps my dictionary is incorrect???
#            params.adaptiveThreshConstant = 6
#            params.minMarkerPerimeterRate = 0.02
#            params.maxMarkerPerimeterRate = 3.0
#            params.polygonalApproxAccuracyRate = 0.07
#            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # Improves accuracy

            # Improve contrast
            tmp =  cv2.cvtColor(self.median_color_img.copy(), cv2.COLOR_BGR2GRAY)
            corners, ids, _, _ = charuco_detector.detectBoard(tmp)

            print("Detected Aruco Target IDs: ", ids)

            if visualize:
    #            cv2.aruco.drawDetectedMarkers(tmp, np.asarray(rejected_candidates)/2, borderColor=(100, 0, 255))
    #            cv2.aruco.drawDetectedMarkers(tmp, np.asarray(corners)/2, ids)

                # Useful for debugging
    #            cv2.aruco.drawDetectedMarkers(tmp, rejected_candidates, borderColor=(100, 0, 255))

                cv2.aruco.drawDetectedCornersCharuco(tmp, corners, ids)
                plt.imshow(tmp)
                plt.show()

            # Charuco Board coordinate system pose estimation in Lidar frame
            camera_matrix = np.asarray(self.caminfo.k).reshape(3,3)
            distortion_coeffs = np.zeros(5)

            obj_corners = charuco_board.getChessboardCorners()
            object_points = []
            for idx in ids:
                object_points.append(obj_corners[idx])

            object_points = np.array(object_points, dtype=np.float32)

            # Estimate charuco board pose
            ret, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, distortion_coeffs, flags=cv2.SOLVEPNP_IPPE)
            print("R: ", ARUCO_ROT_IN_LIDAR:=cv2.Rodrigues(rvec)[0], "\nt: ", tvec)

            disp = self.median_color_img.copy()
            cv2.drawFrameAxes(disp, camera_matrix, distortion_coeffs, rvec, tvec, 0.03)
            plt.imshow(disp)
            plt.show()

            # Construct intermediate and world transforms
            LIDAR_POSE_IN_WORLD = np.eye(4)

        return LIDAR_POSE_IN_WORLD

    def _detect_calibration_plate(self):
        # RH: optional, may be nice to have. Should find corners of plate and store these on the object
        return

    def _detect_aruco(self):
        return

    def _estimate_pose_aruco(self):
        return

    def _plot_aruco(self):
        return

    def _select_flat_regions(self, interactive=True):
        # RH: Best way to select these is still up in the air, but a manual method may do the trick...
        if interactive:
            # RH: Do manual polygon picking, which should give us all pixels belonging to poly (which are FULLY inside?)
            pass
        else:
            # RH: Non-interactive we may just use ARUCO target plane and orientation to find approximate
            pass

        return

    def _plane_fit(self, indices):
        # RH: Given a set of points, compute the planar fit via LSQ (or some other method...)
        return

if __name__ == "__main__":
    #bag_path = "/mnt/c/Users/ryan1/Downloads/07102025_15_07_51/07102025_15_07_51/192.168.2.4:8000"

    rclpy.init()

    # Get path to bag file
    import sys
    bag_path = sys.argv[1]
    color_topic = sys.argv[2]
    depth_topic = sys.argv[3]
    caminfo_topic = sys.argv[4]

    originFuser = OriginFusion()
    originFuser.LoadBag(bag_path, color_topic, depth_topic, caminfo_topic)
    originFuser.PlotImages()
    # Having the surface characterization run as its own node makes sense. Maybe i should just make this a service outright and skip the joining service.
    originFuser.get_lidar_to_world_tf()

    rclpy.shutdown()
