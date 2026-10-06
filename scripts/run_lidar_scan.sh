#!/usr/bin/env bash
# Run one LiDAR scan and wait until it has been downloaded. Wrapper around collect.sh.
#   run_lidar_scan.sh [SECONDS]   (default: the duration configured in ground control)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Perception Ground Control: LiDAR scan"
read -r -p "TURN OFF MOCAP CAMERAS. Press Enter to START the LiDAR scan..."
exec "${SCRIPT_DIR}/collect.sh" scan "$@"
