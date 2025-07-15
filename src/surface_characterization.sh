#!/usr/bin/env bash
# Finds the most recent calibration folder in ~/lidarcalibrations and runs:
#   1) gantry_services mlss_calibration
#   2) converter.py
#   3) characterize_surface.py


set -euo pipefail

# 1) Most recent top‐level timestamp folder
top_ts_dir=$(ls -dt ~/lidarcalibrations/*/ | head -1)

# Dive two levels down (IP folder and its only subdir)
calibration_path=$(ls -d "${top_ts_dir}"*/*/ | head -1)
calibration_path="${calibration_path%/}"
calibration_path="${recent_dir%/}"

echo "Using calibration folder: $calibration_path"

# 2. Run the MLSS calibration for 10 seconds
echo "Running gantry_services mlss_calibration..."
ros2 run gantry_services mlss_calibration #--duration 10.0

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
    ~/roselab-perception/clouds/out.ply 0 1 --local

echo "Pipeline completed successfully, closing."
