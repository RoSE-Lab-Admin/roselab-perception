#!/usr/bin/env bash
# Finds the most recent calibration folder in ~/lidarcalibrations and runs:
#   1) gantry_services mlss_calibration
#   2) converter.py
#   3) characterize_surface.py


set -euo pipefail

# 1. Locate the most recent subdirectory in ~/lidarcalibrations
#    - ls -dt: sort by modification time, newest first
#    - head -1: take the top entry
recent_dir=$(ls -dt ~/lidarcalibrations/*/ | head -1)
# Remove trailing slash if present
calibration_path="${recent_dir%/}"

echo "Using calibration folder: $calibration_path"

# 2. Run the MLSS calibration for 10 seconds
echo "Running gantry_services mlss_calibration..."
ros2 run gantry_services mlss_calibration --duration 10.0

# 3. Convert images to point cloud
echo "Converting images to point cloud..."
python3 ~/roselab-perception/src/converter.py \
    "$calibration_path" \
    /l515_center/color/image_raw \
    /l515_center/aligned_depth_to_color/image_raw \
    ~/roselab-perception/clouds/out.ply

# 4. Characterize the surface from the generated point cloud
echo "Characterizing surface..."
python3 ~/roselab-perception/src/characterize_surface.py \
    ~/roselab-perception/clouds/out.ply

echo "Pipeline completed successfully, closing."
