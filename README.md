# roselab-perception
Image and 3D data management, computer vision, and analysis routines for MLSS experiments

## Workflow 0: MLSS Calibration (Surface Prep Characterization)
Simply run: cd roselab-perception/src/ && ./surface_characterization.sh

## Workflow 1: Trial bag analysis
1) Convert each bag to point cloud via convert.py, output stored in bag folder
2) Run characterize_surface.py to generate tif files of various spatial statistics, voxel grid DEM, and "report card", output stored in bag folder
3) Run dem_compare.py for pre vs post analysis (specify folder for pre and post result tifs)

## Workflow 2: Single lidar scan analysis
1) Convert each bag to point cloud via convert.py, output stored in bag folder
2) Run characterize_surface.py to generate npz files of various spatial statistics andvarious spatial statistics, voxel grid DEM, and "report card", output stored in bag folder

# TODO - Capture

[ ] MVP! Add capture scripts in python / cpp for working with realsense D456 (Depth, Color, IMU, intrinsics, extrinsics)
[ ] MVP! Create ROS service similar to lidar gantry capture service which uses ROS service calls to trigger local (on Pi) bagging of MastCam topics on trial run (if MastCam namespace found with universal bagger? Launch parameter?)
[ ] Add functionality for automatically downloading capture
[ ] DEMO! Create foxglove UI layout for subscribing to relevant topics from MastCam (live feeds, or bags)

# TODO - Mapping

[ ] MVP! Add raw (Depth, Color, IMU, intrinsics/extrinsics) -> aligned RGBD images -> to PointCloud2 functionality
[ ] MVP! Add PointCloud2 aggregation via a) explicit pose stream reconstruction (MAST CAM ONLY) or b) estimated pose graph via KISS-ICP
[ ] Add support for saving transformed data to disk as PCD/PLY along with pose information if SLAM used
[ ] Add map filtering, downsample, voxelization, and DEM/mesh generation support
[ ] Add Open3D TSDFVolume integration for RGBD images with provided poses from either of the above methods
