#!/usr/bin/env bash
# Record MastCam + RoSEy telemetry ("trial" group) until Enter, then download.
# Wrapper around collect.sh.
#   capture_rosey_mastcam_once.sh [LABEL]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL_ARGS=()
[ -n "${1:-}" ] && LABEL_ARGS=(-l "$1")

read -r -p "TURN ON MOCAP CAMERAS. Press Enter to START MastCam + RoSEy recording..."
"${SCRIPT_DIR}/collect.sh" start trial "${LABEL_ARGS[@]}" || exit $?
read -r -p "Press Enter to STOP MastCam + RoSEy recording..."
exec "${SCRIPT_DIR}/collect.sh" stop trial
