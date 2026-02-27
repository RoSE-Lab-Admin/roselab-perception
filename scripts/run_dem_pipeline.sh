#!/usr/bin/env bash
# Finds the most recent calibration folder in ~/lidarcalibrations and runs:
#   1) Use provided bag
#   2) converter.py
#   3) characterize_surface.py


set -euo pipefail

# 1. Run the MLSS calibration for 10 seconds
BAG="${1:-/mnt/d/perception_data/}" # RH: This is how to set positional params with a default value if they aren't provided

echo "Using scan folder: $BAG"

# 3. Convert images to point cloud
echo "Converting images to point cloud..."
python3 ~/roselab-perception/src/converter.py \
    "$BAG" \
    /p_l515_center/color/image_raw \
    /p_l515_center/aligned_depth_to_color/image_raw \
    ../data/tf_summer_zup.npz \
    out.ply

# 4. Characterize the surface from the generated point cloud
echo "Characterizing surface..."
python3 ~/roselab-perception/src/characterize_surface.py \
    "$BAG"/out.ply 0 1 0.05 --local

echo "Pipeline completed successfully, closing."
