#!/usr/bin/env bash
# collect.sh - start/stop perception data collection at runtime.
#
# Talks to the ground_control node (perception_ground_control), which drives every
# remote_data_collector (mastcam, gantry/lidar, ground station, ...) and downloads the bags into
# the current session directory. Targets are remote or group names from
# ros/src/perception_ground_control/config/ground_control.yaml ("all" = every remote).
#
# Usage:
#   collect.sh start  [TARGET...] [-l LABEL] [-d SECONDS]  start recording (default group: trial)
#   collect.sh stop   [TARGET...] [--no-wait]              stop, download (+ delete per config)
#   collect.sh record  TARGET...  -d SECONDS [-l LABEL]    timed recording, wait for download
#   collect.sh scan   [SECONDS]   [-l LABEL]               lidar scan (group "lidar"), wait for it
#   collect.sh status                                      remotes, recordings, downloads
#   collect.sh session                                     start a new local session directory
#   collect.sh trial  [LABEL_PREFIX] [--no-scan]           interactive scan/trial/scan loop
#   collect.sh rdc    ARGS...                              talk to one collector directly
#                                                          (no ground control), see `rdc -h`
# Examples:
#   collect.sh start                     # mastcam + ground station ("trial" group)
#   collect.sh start trial -l run07      # bags named run07_<bag_name>_<timestamp>
#   collect.sh stop                      # stop everything, wait for downloads
#   collect.sh scan 60                   # 60 s lidar scan, downloaded when done
#   collect.sh stop all                  # emergency stop of every remote
#   collect.sh rdc status /mastcam_front
#
# Environment:
#   GROUND_CONTROL_NODE  ground control node name (default /ground_control)
#   TRIAL_GROUP          group used by `trial` (default trial)
#   LIDAR_GROUP          group used by `scan` / `trial` (default lidar)
#   ROS_SETUP            ROS setup file (default /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash)
#   WS_SETUP             workspace setup file (default <repo>/ros/install/setup.bash)
#
# Exit status: 0 ok, 1 failure, 2 usage error, 3 ground control / collector unreachable.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRIAL_GROUP="${TRIAL_GROUP:-trial}"
LIDAR_GROUP="${LIDAR_GROUP:-lidar}"
export GROUND_CONTROL_NODE="${GROUND_CONTROL_NODE:-/ground_control}"

usage() { sed -n '2,/^# Exit status/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "collect.sh: $*" >&2; exit 2; }

# ---------------------------------------------------------------------------- environment
find_rdc() {
  local prefix
  local IFS=:
  for prefix in ${AMENT_PREFIX_PATH:-}; do
    if [ -x "${prefix}/lib/remote_data_collector/rdc" ]; then
      echo "${prefix}/lib/remote_data_collector/rdc"
      return 0
    fi
  done
  return 1
}

setup_env() {
  RDC="$(find_rdc)" && return 0
  local ros_setup="${ROS_SETUP:-/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash}"
  local ws_setup="${WS_SETUP:-${REPO_DIR}/ros/install/setup.bash}"
  set +u
  # shellcheck disable=SC1090
  [ -f "${ros_setup}" ] && source "${ros_setup}"
  # shellcheck disable=SC1090
  [ -f "${ws_setup}" ] && source "${ws_setup}"
  RDC="$(find_rdc)" || {
    echo "collect.sh: remote_data_collector is not built/sourced." >&2
    echo "  Build it:  cd ${REPO_DIR}/ros && colcon build --packages-up-to perception_ground_control" >&2
    echo "  or set WS_SETUP=/path/to/install/setup.bash" >&2
    exit 3
  }
}

rdc() { "${RDC}" "$@"; }
gc() { "${RDC}" gc "$@"; }

# ---------------------------------------------------------------------------- argument parsing
# Splits "$@" into TARGETS (positional), LABEL (-l), DURATION (-d), NO_WAIT (--no-wait).
parse_args() {
  TARGETS=(); LABEL=""; DURATION=""; NO_WAIT=0; NO_SCAN=0
  while [ $# -gt 0 ]; do
    case "$1" in
      -l|--label)    [ $# -ge 2 ] || die "$1 needs a value"; LABEL="$2"; shift 2 ;;
      -d|--duration) [ $# -ge 2 ] || die "$1 needs a value"; DURATION="$2"; shift 2 ;;
      --no-wait)     NO_WAIT=1; shift ;;
      --no-scan)     NO_SCAN=1; shift ;;
      -h|--help)     usage; exit 0 ;;
      -*)            die "unknown option $1 (see collect.sh --help)" ;;
      *)             TARGETS+=("$1"); shift ;;
    esac
  done
  if [ -n "${DURATION}" ] && ! [[ "${DURATION}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    die "duration must be a number of seconds, got '${DURATION}'"
  fi
}

# ---------------------------------------------------------------------------- commands
cmd_start() {
  local args=(start "${TARGETS[@]}")
  [ -n "${LABEL}" ] && args+=(--label "${LABEL}")
  [ -n "${DURATION}" ] && args+=(--duration "${DURATION}")
  gc "${args[@]}"
}

cmd_stop() {
  local args=(stop "${TARGETS[@]}")
  [ "${NO_WAIT}" = 1 ] && args+=(--no-wait)
  gc "${args[@]}"
}

# After a failed start, stop + collect the remotes that DID start (listed as "OK <remote>:" in
# ground control's reply), so a retry is not rejected as "busy" by half-started recordings.
cleanup_partial_start() {
  local started
  mapfile -t started < <(echo "$1" | sed -n 's/^  OK    \([^:]*\): recording.*/\1/p')
  [ ${#started[@]} -gt 0 ] || return 0
  echo "Stopping the remotes that did start (${started[*]}) so the attempt can be retried..."
  gc stop "${started[@]}"
}

# Start TARGETS, then wait for the recording(s) to end and collect them (downloads awaited).
# The wait time is taken from ground control's reply ("... (for 60s)"), so remotes with a
# configured default duration work without -d. Open-ended recordings wait for Enter.
start_and_collect() {
  local out rc secs left
  out="$(cmd_start)"; rc=$?
  echo "${out}"
  if [ ${rc} -ne 0 ]; then
    cleanup_partial_start "${out}"
    return ${rc}
  fi
  secs="$(echo "${out}" | grep -o '(for [0-9]*s)' | tr -dc '0-9\n' | sort -n | tail -1)"
  if [ -z "${secs}" ]; then
    read -r -p "Recording until stopped. Press Enter to STOP..."
  else
    for ((left = secs; left > 0; left--)); do
      printf '\r  recording... %4ss left ' "${left}"
      sleep 1
    done
    printf '\r  recording finished; collecting...      \n'
    sleep 2  # let the collector finalize on its own before asking for the results
  fi
  NO_WAIT=0
  cmd_stop
}

cmd_record() {
  [ ${#TARGETS[@]} -gt 0 ] || die "record needs at least one target"
  [ -n "${DURATION}" ] || die "record needs -d SECONDS"
  awk -v d="${DURATION}" 'BEGIN { exit !(d > 0) }' || die "record needs a positive duration"
  start_and_collect
}

cmd_scan() {
  local secs="${TARGETS[0]:-}"
  if [ -n "${secs}" ]; then
    [[ "${secs}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "scan takes a number of seconds, got '${secs}'"
    DURATION="${secs}"
  fi
  TARGETS=("${LIDAR_GROUP}")
  start_and_collect
}

cmd_trial() {
  local prefix="${TARGETS[0]:-trial}" n=1
  echo "Interactive trial loop (Ctrl-C to quit). $(gc status | head -1)"
  if [ "${NO_SCAN}" = 0 ]; then
    read -r -p "TURN OFF MOCAP CAMERAS, then press Enter to START the LiDAR scan..."
    TARGETS=(); LABEL="${prefix}$(printf '%02d' 0)_pre"; DURATION=""; cmd_scan || true
  fi
  while true; do
    echo
    read -r -p "TURN ON MOCAP CAMERAS, then press Enter to START ${prefix}$(printf '%02d' ${n}) (${TRIAL_GROUP})..."
    TARGETS=("${TRIAL_GROUP}"); LABEL="${prefix}$(printf '%02d' ${n})"; DURATION=""
    local out
    if ! out="$(cmd_start)"; then
      echo "${out}"
      cleanup_partial_start "${out}"
      echo "Start failed (see above). Fix the problem and press Enter to retry."
      continue
    fi
    echo "${out}"
    read -r -p "Press Enter to STOP ${prefix}$(printf '%02d' ${n})..."
    TARGETS=("${TRIAL_GROUP}"); NO_WAIT=0; cmd_stop || true
    if [ "${NO_SCAN}" = 0 ]; then
      read -r -p "TURN OFF MOCAP CAMERAS, then press Enter to START the LiDAR scan..."
      TARGETS=(); LABEL="${prefix}$(printf '%02d' ${n})_post"; DURATION=""; cmd_scan || true
    fi
    n=$((n + 1))
  done
}

# ---------------------------------------------------------------------------- main
main() {
  local cmd="${1:-}"
  [ -n "${cmd}" ] || { usage; exit 2; }
  shift
  case "${cmd}" in
    -h|--help|help) usage; exit 0 ;;
    start|stop|record|scan|status|session|trial|rdc) ;;
    *) die "unknown command '${cmd}' (see collect.sh --help)" ;;
  esac
  setup_env
  case "${cmd}" in
    start)   parse_args "$@"; cmd_start ;;
    stop)    parse_args "$@"; cmd_stop ;;
    record)  parse_args "$@"; cmd_record ;;
    scan)    parse_args "$@"; cmd_scan ;;
    status)  gc status ;;
    session) gc new-session ;;
    trial)   parse_args "$@"; cmd_trial ;;
    rdc)     rdc "$@" ;;
    *)       die "unknown command '${cmd}' (see collect.sh --help)" ;;
  esac
}

main "$@"
