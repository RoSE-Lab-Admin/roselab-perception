#!/usr/bin/env bash
# Capture a single LiDAR scan (gantry) and download it. Wrapper around collect.sh.
#   capture_lidar_once.sh [SECONDS]   (default: the duration configured in ground control)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
read -r -p "TURN OFF MOCAP CAMERAS. Press Enter to START the LiDAR scan..."
exec "${SCRIPT_DIR}/collect.sh" scan "$@"
