# roselab-perception
Image and 3D data management, computer vision, and analysis routines for MLSS experiments

## Workflow 0: Surface Prep Characterization
This routine should be executed once every time you want to calculate 

Simply run: 

```bash
cd roselab-perception/src/ && ./surface_characterization.sh
```

## Workflow 1: MLSS Calibration Pipeline
Simply run: 

## Workflow 2: Trial Analyzer
Simply run: 

## Workflow 3: Health Checker
Simply run: 

## Workflow 4: MLSS Visualizer
Simply run: 

## Workflow 5: Realsense Capture Code + ROS2 Services


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
