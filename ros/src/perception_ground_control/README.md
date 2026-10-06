# perception_ground_control

The **master client** for perception data collection. `ground_control` runs on the ground
station (Slade) and drives any number of
[`remote_data_collector`](../remote_data_collector/README.md) instances (mast cam, gantry/lidar,
ground station telemetry, rover avionics, wheel cam, ...). It:

1. starts/stops recordings on remotes or **groups** of remotes (RecordBag action goals),
2. downloads finished bags over each remote's HTTP sidecar into a local **session directory**
   (`<data_root>/<MMDDYYYY>/<HH-MM-SS>/<bag>/`, the same layout as before),
3. optionally deletes the remote copy after a verified download,
4. logs every recording/download to `<session>/collection_log.jsonl`.

Ground control records nothing itself. Even the rover telemetry bag that the old node recorded
locally now comes from a `ground_station` collector, so every recorder is the same component.

Operators normally use [`scripts/collect.sh`](../../../scripts/collect.sh) rather than calling
these services directly.

## Run

```bash
cd ~/roselab-perception/ros && source install/setup.bash
ros2 launch perception_ground_control launch_ground.py                 # config/ground_control.yaml
ros2 launch perception_ground_control launch_ground.py duration:=60.0  # legacy args still work
ros2 launch perception_ground_control launch_ground.py config:=/path/to/my_fleet.yaml
```

Legacy launch arguments are mapped onto the new parameters: `slade_root` → `data_root`,
`duration` → `remotes.gantry.duration`, `pi_file` → `remotes.mastcam_front.bag_name`,
`panda_file` → `remotes.gantry.bag_name`.

## Services

| Service | Type | |
|---|---|---|
| `/ground_control/start` | `remote_data_collector_interfaces/srv/StartCollection` | `targets[]`, `label`, `duration` (0 = remote default, >0 seconds, <0 = until stopped) |
| `/ground_control/stop` | `remote_data_collector_interfaces/srv/StopCollection` | `targets[]` (empty = all), `wait_for_download` |
| `/ground_control/status` | `std_srvs/srv/Trigger` | multi-line summary in `message` |
| `/ground_control/new_session` | `std_srvs/srv/Trigger` | new session directory (returned in `message`) |

```bash
ros2 run remote_data_collector rdc gc start trial --label run03     # pretty CLI
ros2 run remote_data_collector rdc gc stop
ros2 service call /ground_control/start remote_data_collector_interfaces/srv/StartCollection \
  "{targets: [lidar], duration: 30.0}"                                  # raw
```

Behavior:

* **start** fans out to all targets in parallel. It returns once each remote *confirms* it is
  recording, or reports why it isn't (unreachable, busy, unknown profile, recorder failed...).
  Remotes that fail don't prevent the others from starting; `started[]`/`failed[]` list both.
* **Timed** recordings (duration > 0, e.g. the lidar scan) are downloaded automatically as soon
  as they finish.
* **stop** cancels the recordings on the targets (stateless: it also stops recordings started
  by `rdc` or by a previous ground control instance), then downloads. With
  `wait_for_download: true` (the default in `rdc gc stop`/`collect.sh stop`) it returns once the
  data is on disk. A bare stop skips remotes that are offline and have nothing of ours
  recording; naming an offline remote explicitly is an error.
* A failed download leaves the bag on the remote and logs the `rdc download ...` command to
  retry it.
* Ground control can be restarted at any time. Remotes keep recording, and the next `stop`
  collects their bags.

## Configuration (`config/ground_control.yaml`)

```yaml
ground_control:
  ros__parameters:
    data_root: /mnt/e/perception-data
    session_format: "%m%d%Y/%H-%M-%S"     # session subdirectory (strftime)
    default_group: trial                  # used when start has no targets
    start_timeout_sec: 30.0
    stop_timeout_sec: 180.0
    download_socket_timeout_sec: 60.0
    download_retries: 3
    max_parallel_downloads: 3
    remotes:
      mastcam_front:
        node: /mastcam_front              # collector node name (default "/<remote name>")
        bag_name: mastcam_bags            # bag prefix ("<label>_" is prepended if given)
        profiles: [mastcam]               # optional; default = the collector's default_profiles
        topics: [/extra/topic]            # optional extra topics
        regex: ""                         # optional extra regex
        duration: 0.0                     # default duration, 0 = until stopped
        download: true
        delete_after_download: true
    groups:
      trial: [mastcam_front, ground_station]
      lidar: [gantry]
```

Adding a new system is: run a `remote_data_collector` there with a unique name, then add a
`remotes.<name>` entry (and put it in a group). No new code or interfaces are needed.

## Tests

```bash
colcon test --packages-select perception_ground_control && colcon test-result --verbose
```

* `test_fleet.py`: config parsing, duration/label rules, target and group resolution.
* `test_ground_control_integration.py`: ground control plus two real collectors (fake
  recorder) in one process. Covers start/stop/download/delete, auto-download of timed scans,
  per-remote error reporting, offline remotes, stopping recordings started elsewhere, session
  directories, and the `rdc gc` CLI.
* `test_collect_sh.py`: `scripts/collect.sh` argument handling and wiring, using a stub `rdc`.
