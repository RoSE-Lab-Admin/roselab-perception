#!/usr/bin/env bash
# selftest_collection.sh - end-to-end self test of the data collection stack on ONE machine.
#
# Starts, in separate processes and on an isolated ROS_DOMAIN_ID:
#   * two test publishers,
#   * two remote_data_collector instances (real `ros2 bag record`), one with an open-ended
#     profile and one with a regex profile + timed default,
#   * a ground_control with its own fleet config,
# then drives everything through scripts/collect.sh exactly like an operator, and checks that
# bags with real messages land in the session directory, remote copies are deleted/kept as
# configured, and missing topics are reported. Nothing touches the real lab collectors.
#
# Usage:  source ros/install/setup.bash && scripts/selftest_collection.sh [WORK_DIR]
# Exit status 0 = all checks passed.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECT="${SCRIPT_DIR}/collect.sh"
W="${1:-$(mktemp -d /tmp/rdc_selftest.XXXXXX)}"
export ROS_DOMAIN_ID="${SELFTEST_DOMAIN_ID:-$((150 + RANDOM % 70))}"
export GROUND_CONTROL_NODE=/selftest_gc
PORT_BASE=$((20000 + RANDOM % 20000))
FAILS=0
PIDS=()

command -v ros2 > /dev/null || { echo "Source your ROS workspace first."; exit 2; }
EXE="$(ros2 pkg prefix remote_data_collector 2>/dev/null)/lib/remote_data_collector/remote_data_collector"
GC="$(ros2 pkg prefix perception_ground_control 2>/dev/null)/lib/perception_ground_control/ground_control"
[ -x "${EXE}" ] && [ -x "${GC}" ] || { echo "Build remote_data_collector + perception_ground_control first."; exit 2; }

cleanup() {
  [ ${#PIDS[@]} -gt 0 ] && kill -INT "${PIDS[@]}" 2> /dev/null
  sleep 3
  [ ${#PIDS[@]} -gt 0 ] && kill -9 "${PIDS[@]}" 2> /dev/null
}
trap cleanup EXIT
check() {  # check "description" command...
  local what="$1"; shift
  if "$@"; then echo "  PASS  ${what}"; else echo "  FAIL  ${what}"; FAILS=$((FAILS + 1)); fi
}

mkdir -p "${W}"
echo "Self test in ${W} (ROS_DOMAIN_ID=${ROS_DOMAIN_ID})"
cat > "${W}/cam.yaml" <<EOF
/**:
  ros__parameters:
    data_dir: ${W}/cam_data
    default_profiles: [cam]
    min_free_space_mb: 0
    stop_free_space_mb: 0
    http: {port: $((PORT_BASE + 1)), bind: 127.0.0.1, advertise_host: 127.0.0.1}
    profiles:
      cam: {topics: [/selftest/image, /selftest/never_published]}
EOF
cat > "${W}/lidar.yaml" <<EOF
/**:
  ros__parameters:
    data_dir: ${W}/lidar_data
    default_profiles: [scan]
    min_free_space_mb: 0
    stop_free_space_mb: 0
    http: {port: $((PORT_BASE + 2)), bind: 127.0.0.1, advertise_host: 127.0.0.1}
    profiles:
      scan: {regex: "^/selftest/lidar/.*"}
EOF
cat > "${W}/fleet.yaml" <<EOF
/**:
  ros__parameters:
    data_root: ${W}/ground
    default_group: trial
    remotes:
      cam: {node: /selftest_cam, bag_name: cam_bags, delete_after_download: true}
      gantry: {node: /selftest_lidar, bag_name: lidar_bags, duration: 4.0}
    groups:
      trial: [cam]
      lidar: [gantry]
EOF

ros2 topic pub -r 20 /selftest/image std_msgs/msg/String "{data: frame}" > /dev/null 2>&1 & PIDS+=($!)
ros2 topic pub -r 10 /selftest/lidar/points std_msgs/msg/String "{data: pts}" > /dev/null 2>&1 & PIDS+=($!)
"${EXE}" --ros-args -r __node:=selftest_cam --params-file "${W}/cam.yaml" > "${W}/cam.log" 2>&1 & PIDS+=($!)
"${EXE}" --ros-args -r __node:=selftest_lidar --params-file "${W}/lidar.yaml" > "${W}/lidar.log" 2>&1 & PIDS+=($!)
"${GC}" --ros-args -r __node:=selftest_gc --params-file "${W}/fleet.yaml" > "${W}/gc.log" 2>&1 & PIDS+=($!)
echo "Waiting for discovery..."
for _ in $(seq 1 30); do
  "${COLLECT}" status 2> /dev/null | grep -q "selftest_lidar .*online" && break
  sleep 1
done

echo "== status";     "${COLLECT}" status | tee "${W}/status.txt"
check "both collectors online" test "$(grep -c ' online ' "${W}/status.txt")" -eq 2
echo "== start";      "${COLLECT}" start -l selftest; START_RC=$?
check "start succeeded" test ${START_RC} -eq 0
sleep 4
echo "== stop";       "${COLLECT}" stop | tee "${W}/stop.txt"; STOP_RC=${PIPESTATUS[0]}
check "stop succeeded" test "${STOP_RC}" -eq 0
check "missing topic reported" grep -q "/selftest/never_published" "${W}/stop.txt"
echo "== scan";       "${COLLECT}" scan -l pre < /dev/null; SCAN_RC=$?
check "scan succeeded" test ${SCAN_RC} -eq 0

SESSION="$(dirname "$(find "${W}/ground" -name collection_log.jsonl | head -1)")"
CAM_BAG="$(find "${SESSION}" -maxdepth 1 -name 'selftest_cam_bags_*' | head -1)"
LIDAR_BAG="$(find "${SESSION}" -maxdepth 1 -name 'pre_lidar_bags_*' | head -1)"
count() { ros2 bag info "$1" 2> /dev/null | sed -n 's/^Messages: *//p'; }
check "camera bag downloaded with messages" test "$(count "${CAM_BAG}")" -gt 0
check "lidar bag downloaded with messages (regex profile)" test "$(count "${LIDAR_BAG}")" -gt 0
check "camera bag deleted on remote (delete_after_download)" \
  test -z "$(find "${W}/cam_data" -mindepth 1 -maxdepth 1 -not -name '.rdc')"
check "lidar bag kept on remote" test -n "$(find "${W}/lidar_data" -maxdepth 1 -name 'pre_lidar_bags_*')"
check "no .partial downloads left" test -z "$(find "${W}/ground" -name '*.partial')"

echo
if [ ${FAILS} -eq 0 ]; then echo "SELF TEST PASSED (${W})"; else echo "SELF TEST FAILED: ${FAILS} check(s) (logs in ${W})"; fi
exit $((FAILS > 0))
