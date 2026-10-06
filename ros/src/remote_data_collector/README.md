# remote_data_collector

One configurable ROS 2 node that **records rosbags on the machine it runs on**, controlled through
a ROS 2 **action**, with a sidecar **`python3 -m http.server`** for downloading the results. Run
one instance per data-producing computer (mast cam Pi, gantry/lidar box, rover avionics, wheel
cam Pi, ground station, ...). Data only crosses the network once, after the trial, over HTTP.

It replaces the per-system capture services (`mastcam_service/mastcam_capture_service`,
`gantry_services/gantry_capture_service`) and their interface packages.

```
             ground control (Slade)                          remote host (e.g. mast cam Pi)
 ┌───────────────────────────────────────┐      ┌───────────────────────────────────────────┐
 │ scripts/collect.sh  ──►  ground_control│ goal │ remote_data_collector  (/mastcam_front)   │
 │      or `rdc ...`        (master client)├─────►│   ~/record  (RecordBag action)            │
 │                                        │◄─────┤     └─ spawns `ros2 bag record` locally   │
 │                                        │feedbk│   ~/status ~/list_bags ~/delete_bag       │
 │      downloads <session>/<bag>/  ◄─────┼──────┤   http.server :8000  (serves data_dir)    │
 └───────────────────────────────────────┘ HTTP └───────────────────────────────────────────┘
```

Contents: [Quick start](#quick-start) · [Interface](#ros-interface) ·
[Recording lifecycle](#recording-lifecycle) · [Configuration](#configuration) ·
[rdc CLI](#rdc-command-line-client) · [Python client](#python-client-library) ·
[Migration](#migrating-from-the-legacy-capture-services) · [Tests](#tests) ·
[Troubleshooting](#troubleshooting)

---

## Quick start

Build (on every host that runs a collector **and** on the ground station; all hosts must build the
same `remote_data_collector_interfaces`):

```bash
cd ~/roselab-perception/ros
colcon build --packages-up-to remote_data_collector
source install/setup.bash
```

Run a collector. The node **name** is its identity (`/<name>/record`, ...). If a shipped config
has the same name it is used automatically:

```bash
ros2 launch remote_data_collector remote_data_collector.launch.py name:=mastcam_front   # mast cam Pi
ros2 launch remote_data_collector remote_data_collector.launch.py name:=gantry          # gantry box
ros2 launch remote_data_collector remote_data_collector.launch.py name:=ground_station  # Slade
# Any other machine: copy config/example.yaml, edit, then
ros2 launch remote_data_collector remote_data_collector.launch.py name:=my_box config:=/path/my_box.yaml
```

Talk to it from anywhere on the network:

```bash
ros2 run remote_data_collector rdc discover                     # which collectors are up?
ros2 run remote_data_collector rdc status /mastcam_front
ros2 run remote_data_collector rdc start  /mastcam_front -n trial01
ros2 run remote_data_collector rdc stop   /mastcam_front --download ~/data --delete
```

For multi-system trials use **ground control** (`perception_ground_control`) and
`scripts/collect.sh` — see the top-level README.

## ROS interface

All endpoints are relative to the node: `/<namespace>/<node_name>/...`.

| Endpoint | Type | Purpose |
|---|---|---|
| `~/record` | action `remote_data_collector_interfaces/action/RecordBag` | Record a bag. Cancel = stop. |
| `~/status` | srv `remote_data_collector_interfaces/srv/GetStatus` | Health, disk, HTTP URL, profiles, active recordings. |
| `~/list_bags` | srv `remote_data_collector_interfaces/srv/ListBags` | Bags in `data_dir` with download URL + file list. |
| `~/delete_bag` | srv `remote_data_collector_interfaces/srv/DeleteBag` | Delete exactly one (non-active) bag. |

`RecordBag` goal: `bag_name` (prefix; `_<YYYY-mm-ddTHH-MM-SS>` is appended), `profiles[]`,
`topics[]`, `regex`, `duration` (`<= 0` = until canceled). Feedback: a `RecordingStatus`
(state, elapsed, size, subscribed topics) every `feedback_period_sec`. Result: `success`,
`message`, `bag` (`BagInfo`: name, path, `download_url`, `files[]`, `file_sizes[]`, ...),
`recorded_sec`, `subscribed_topics[]`, `missing_topics[]`, `log_url`.

Raw CLI equivalents (no `rdc` needed):

```bash
# timed 30 s recording, prints feedback and the result (Ctrl-C cancels = stops early)
ros2 action send_goal --feedback /gantry/record remote_data_collector_interfaces/action/RecordBag \
  "{profiles: [gantry_state, p_l515_center], duration: 30.0}"
# stop EVERY recording on a collector (zero goal id + zero stamp = cancel all goals)
ros2 service call /mastcam_front/record/_action/cancel_goal action_msgs/srv/CancelGoal "{}"
ros2 service call /mastcam_front/status remote_data_collector_interfaces/srv/GetStatus
ros2 service call /mastcam_front/list_bags remote_data_collector_interfaces/srv/ListBags "{pattern: 'trial*'}"
ros2 service call /mastcam_front/delete_bag remote_data_collector_interfaces/srv/DeleteBag "{name: 'trial01_2026-10-06T12-00-00'}"
```

## Recording lifecycle

```
goal ─► validate (profiles, topics, bag name, free disk, max_concurrent) ──fail──► ABORTED (+reason)
          │
          ▼
     STARTING: spawn `ros2 bag record -o <data_dir>/<bag> ...`  ──never starts──► ABORTED (+log tail)
          │  (waits until the recorder reports "Recording..." and the bag folder exists)
          ▼
     RECORDING: feedback every feedback_period_sec
          │
          ├─ goal canceled ("stop") ─────────────┐
          ├─ duration reached ───────────────────┤
          ├─ max_duration_sec safety net ────────┤
          ├─ disk below stop_free_space_mb ──────┤
          ├─ collector shutting down (SIGINT/TERM)┤
          └─ recorder died ──────────────────────┤
                                                 ▼
     STOPPING: SIGINT ─(stop_timeout_sec)─► SIGTERM ─► SIGKILL; if metadata.yaml is missing,
               `ros2 bag reindex` (reindex_unfinished_bags)
          │
          ▼
     result: CANCELED (stopped) | SUCCEEDED (duration) | ABORTED (anything else)
             result.success == "a finalized bag is on disk" — true even for CANCELED.
```

Robustness features worth knowing about:

* **Stateless stop.** Stopping uses the standard action cancel protocol, so *any* client can stop
  and collect a recording that another process started (`rdc stop`, ground control after a
  restart, `ros2 service call .../_action/cancel_goal`). Results are kept for
  `result_timeout_sec` and can be fetched by goal id.
* **Graceful finalization.** The recorder is launched through a trampoline that restores the
  default SIGINT disposition. Without it, a collector started from a context that ignores SIGINT
  (bash background jobs, some service managers) leaves recorders that silently ignore the stop
  request and never write `metadata.yaml`. The legacy capture services had this problem.
* **Crash recovery.** Each running recording leaves a marker in `<data_dir>/.rdc/active/`. On
  startup the collector stops recorders orphaned by a crashed predecessor and reindexes their bags.
* **Diagnostics.** Each recorder's stdout/stderr goes to `<data_dir>/.rdc/logs/<bag>.log`
  (downloadable via `log_url`). Results list `missing_topics`: requested topics that never
  appeared, usually a publisher that is down or a typo.
* **Unique recorder node names** (`rdc_<collector>_<goal>`), so recorders on different hosts
  never collide as the old `rosbag2_recorder` nodes could.
* **Safe deletes.** Only exact names of direct children of `data_dir`. Wildcards, paths, and
  active bags are rejected. The legacy services passed user input to `glob()`, so a `*` could
  delete every bag.
* **Supervised HTTP sidecar.** Restarted automatically if it dies; `status` reports `http_alive`.
* **No orphans.** The sidecar and recorders are started with `PR_SET_PDEATHSIG`. If the collector
  dies hard (crash, SIGKILL), the kernel SIGTERMs the sidecar, so it can't keep holding the port,
  and SIGINTs each recorder, so it finalizes its bag.
* **Shutdown.** SIGINT/SIGTERM finalize all recordings before exiting (budget: `stop_timeout_sec`
  + escalation + `reindex_timeout_sec`; the launch file waits accordingly). A second Ctrl-C
  forces exit after telling recorders to finalize on their own. Under systemd (the ansible
  units), `systemctl stop` signals every process in the unit and bags are finalized; Jazzy's
  `ros2 bag record` finalizes on SIGTERM too.

## Configuration

Parameters (see the fully commented [`config/example.yaml`](config/example.yaml)):

| Parameter | Default | Meaning |
|---|---|---|
| `data_dir` | `~/remote_data_collector` | Where bags/logs go (`~` expanded). |
| `default_bag_name` | node name | Prefix when a goal leaves `bag_name` empty. |
| `default_profiles` | — | Profiles used when a goal names none. |
| `max_concurrent_recordings` | `1` | Extra goals are aborted with a "busy" reason. |
| `max_duration_sec` | `0` | Safety net for forgotten recordings (0 = off). |
| `min_free_space_mb` / `stop_free_space_mb` | `1024` / `256` | Refuse to start / stop recording below these. |
| `feedback_period_sec` | `1.0` | Feedback and stop-polling period. |
| `startup_timeout_sec` | `20` | Recorder must start writing within this. |
| `stop_timeout_sec` | `60` | SIGINT grace before escalating. |
| `reindex_unfinished_bags` / `reindex_timeout_sec` | `true` / `300` | Run `ros2 bag reindex` on unfinalized bags. |
| `result_timeout_sec` | `3600` | How long finished results stay fetchable. |
| `record_args` | — | Extra `ros2 bag record` flags for every recording (e.g. storage/compression). |
| `ros2_executable` | `ros2` | Recorder executable (tests point this at a fake). |
| `http.enabled` / `http.port` / `http.bind` | `true` / `8000` / `0.0.0.0` | Sidecar server. |
| `http.advertise_host` | auto | Host/IP used in download URLs (auto = LAN IP of default route). |
| `profiles.<name>.*` | — | Topic profiles, below. |

**Profiles** are named topic selections; a goal records the union of the profiles it names plus
its own `topics`/`regex`. Keys: `description`, `topics`, `regex`, `exclude_regex`,
`exclude_topics`, `include` (compose other profiles), `all_topics`, `extra_args`.

```yaml
profiles:
  gantry_state: {topics: [/tf, /tf_static, /gantry/gantry_status/gantry_state]}
  p_l515_center: {topics: [/p_l515_center/color/image_raw, ...]}
  lidar_scan: {include: [gantry_state, p_l515_center, p_l515_west, p_l515_east]}
  rosey_telemetry:
    topics: [/cmd_vel, /joint_states, ...]
    regex: "^/(bno055|joy|roseybot_base_controller)/.*"   # topics + regex are a union
```

Shipped configs (in `config/`), mirroring the legacy setup:

| Config / node name | Host | Replaces | Default profile |
|---|---|---|---|
| `mastcam_front` | dev@192.168.2.104 | `mastcam_capture_service` | `mastcam` (compressed, unaligned) |
| `gantry` | gantry@192.168.2.99 | `gantry_capture_service` | `lidar_scan` (3× L515 + gantry state) |
| `ground_station` | Slade | ground control's own `roseybag` | `rosey_telemetry` (+ mocap) |
| `rosey_avionics` | rosey@192.168.2.50 | — (new: record rover topics on the rover) | `rover_telemetry` |
| `wheelcam` | picam@192.168.2.51 | — (new) | `wheelcam` |

> ROS parameter files cannot express an empty list (`[]`): omit list keys instead.

## rdc command line client

```
rdc discover                                   list collectors on the ROS graph
rdc status   REMOTE                            health, disk, active recordings
rdc start    REMOTE [-n NAME] [-p PROFILE]... [-t TOPIC]... [-e REGEX] [-d SECONDS]
                    [--wait [--download DIR [--delete]]]
rdc stop     REMOTE [--bag NAME]... [--download DIR [--delete]]
rdc list     REMOTE [PATTERN]
rdc download REMOTE BAG... [--dest DIR] [--delete]
rdc delete   REMOTE BAG...
rdc gc start [TARGET...] [-l LABEL] [-d SECONDS]   # ground control (used by collect.sh)
rdc gc stop  [TARGET...] [--no-wait]
rdc gc status | rdc gc new-session
```

`start` returns once the collector confirms it is recording (so a successful exit means data
is being written). `start --wait` stays attached with live progress; Ctrl-C stops the recording.
Exit codes: `0` ok, `1` failure, `2` usage, `3` unreachable.

## Python client library

```python
from remote_data_collector.client import RemoteCollectorClient, spin_in_background

executor = spin_in_background(node)              # the client needs the node to be spun
mast = RemoteCollectorClient(node, '/mastcam_front')
started = mast.start(bag_name='trial01', profiles=['mastcam'],
                     done_callback=lambda outcome: print(outcome.result.message))
...
for outcome in mast.stop():                      # cancels, waits for finalization
    if outcome.ok:
        path = mast.download(outcome.bag, '/mnt/e/perception-data/today')
        mast.delete_bag(outcome.bag.name)
```

All calls block and are thread-safe. Call them from a thread that is *not* the only executor
thread, e.g. a `ReentrantCallbackGroup` callback under a `MultiThreadedExecutor`, or a script
that uses `spin_in_background`.

## Migrating from the legacy capture services

| Legacy | New |
|---|---|
| `ros2 run mastcam_service mastcam_capture_service` | `ros2 launch remote_data_collector remote_data_collector.launch.py name:=mastcam_front` |
| `ros2 run gantry_services gantry_capture_service` | `... name:=gantry` |
| `mastcam_capture_service/start` + `/stop` (services) | `/mastcam_front/record` goal; cancel to stop |
| `gantry_capture_service/capture` (blocking `duration`) | `/gantry/record` goal with `duration` (non-blocking, with feedback) |
| `.../download/name` → JSON with `url` | `result.bag.download_url` + `files[]` (or `~/list_bags`) |
| `.../delete/name` (glob!) | `~/delete_bag` (exact name only) |
| `.../info` | `~/status` |
| `wget -r -nH -P <dir> <url>` | built-in verified download (`rdc download`, ground control) |
| Mastcam params `compress` / `align` | profiles `mastcam`, `mastcam_aligned`, `mastcam_raw` |
| Gantry `sensors: [p_l515_center, ...]` | profiles `p_l515_center`, ... (`lidar_scan` = all) |

Data layout is unchanged: `<data_dir>/<prefix>_<YYYY-mm-ddTHH-MM-SS>/` on the remote and
`<data_root>/<MMDDYYYY>/<HH-MM-SS>/<bag>/` on the ground station. The legacy packages are left
in the tree untouched, so a rollback only means running the old nodes again.

## Tests

```bash
cd ~/roselab-perception/ros
colcon build --packages-up-to perception_ground_control
colcon test --packages-select remote_data_collector perception_ground_control && colcon test-result --verbose
# or, for quick iteration (from ros/src/remote_data_collector):
python3 -m pytest -q test/
```

| File | What it covers |
|---|---|
| `test_naming_profiles.py` | Name/topic/regex validation, profile composition, `ros2 bag record` argv. |
| `test_recorder_store_http.py` | Recorder start/stop/escalation, the SIGINT-inheritance regression, crash/reindex, orphan reaping, safe deletes, HTTP sidecar, verified downloads. |
| `test_collector_integration.py` | The real action server + client in one process: start/stop/download/delete, timed goals, rejections with reasons, concurrency, stop-by-name from another client, recorder crash/hang, shutdown, `max_duration_sec`, crash recovery, bad config. Also one run with the real `ros2 bag record`. |
| `test_cli.py` | Every `rdc` command, exit codes. |
| `scripts/selftest_collection.sh` | Multi-process end-to-end on one machine: real `ros2 bag record`, publishers, two collectors, ground control, driven via `collect.sh`. |
| `test_flake8.py`, `test_pep257.py` | ROS 2 lint. |

Most tests use a **fake `ros2`** (`remote_data_collector/testing/fake_ros2.py`) that implements
`ros2 bag record`/`reindex` without DDS and can simulate failures
(`FAKE_ROS2_MODE=fail_start|slow_start|crash|ignore_sigint`). It is handy for manual debugging
too: set `ros2_executable` to the path printed by
`python3 -c "from remote_data_collector.testing import install_fake_ros2; print(install_fake_ros2('/tmp/fake'))"`.

## Troubleshooting

* **`rdc status` says "not reachable"**: is the node running (`ros2 node list`)? Same
  `ROS_DOMAIN_ID`, RMW, and FastDDS profile as the caller? Were both sides built from the same
  `remote_data_collector_interfaces`? Mismatched interface definitions prevent matching.
  `ros2 daemon stop; ros2 daemon start` refreshes a stale CLI graph cache.
* **Goal aborted with "busy"**: a recording is still active (`rdc status`); stop it or raise
  `max_concurrent_recordings`.
* **`missing_topics` in the result**: nothing was published on those topics during the
  recording. Check the publisher (`ros2 topic info -v`). For realsense, check
  `enable_*`/`align_depth`; for compressed topics, check image_transport plugins.
* **Download URL uses the wrong IP**: set `http.advertise_host` (shipped configs pin the lab IPs).
* **"HTTP sidecar failed to start ... already in use"**: something else (often an orphaned
  legacy `python3 -m http.server`) holds port 8000. `ss -ltnp | grep 8000`, kill it, or change
  `http.port`.
* **Recorder failed to start**: the goal's message includes the tail of the recorder log; the
  full log is at `log_url` / `<data_dir>/.rdc/logs/<bag>.log`.
