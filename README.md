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
2) Run characterize_surface.py to generate tif files of various spatial statistics andvarious spatial statistics, voxel grid DEM, and "report card", output stored in bag folder
