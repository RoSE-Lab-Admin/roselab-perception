#!/usr/bin/env bash
# Takes in trial folder, finds each lidar bag folder, orders by timestamp into pre and post
#   1) converter.py on each bag (pre, post)
#   2) characterize_surface.py (pre, post)
#   3) 

set -euo pipefail

source ~/roselab-perception/venv/bin/activate

# 2. Most recent top‐level timestamp folder
top_ts_dir=$(ls -dt ~/lidarcalibrations/* | head -1)

# Dive two levels down (IP folder and its only subdir)
ip_dir=$(ls -dt "${top_ts_dir}"/192.168.2.4:8000/ | head -n1)

scan_paths=$(ls -d "${ip_dir}"*/ | head -2)
scan_paths="${scan_paths%/}"

echo "Using lidar scan folders: $calibration_path"

for scan in "$scan_paths"; do

	# 3. Convert images to point cloud
	echo "Converting images to point cloud..."
	python ~/roselab-perception/src/converter.py \
	    "$scan" \
	    /l515_center/color/image_raw \
	    /l515_center/aligned_depth_to_color/image_raw \
	    out.ply

	# 4. Characterize the surface from the generated point cloud
	echo "Characterizing surface..."
	python ~/roselab-perception/src/characterize_surface.py \
	    "$scan"/out.ply 0 1 --local

done


scan_paths=$(ls -d "${ip_dir}"*/ | head -2)
scan_paths="${scan_paths%/}"

# 5. Now run diffing script against the tif files (namely DEM and slope, where diff=(post - pre))
python ~/roselab-perception/src/dem_compare.py --pre ${scan_paths[0]} --post ${scan_paths[1]}

echo "Pipeline completed successfully, closing."
