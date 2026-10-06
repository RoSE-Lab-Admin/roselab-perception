# remote_data_collector_interfaces

Interfaces for [`remote_data_collector`](../remote_data_collector/README.md) and
[`perception_ground_control`](../perception_ground_control/README.md).

| Interface | Used by | Purpose |
|---|---|---|
| `action/RecordBag` | collector `~/record` | Record a bag; cancel to stop; result has the download info |
| `msg/BagInfo` | results, `ListBags` | Bag folder: name, path, `download_url`, `files[]`, `file_sizes[]`, ... |
| `msg/RecordingStatus` | feedback, `GetStatus` | Live state of one recording |
| `srv/GetStatus` | collector `~/status` | Health, disk, HTTP, profiles, active recordings |
| `srv/ListBags` | collector `~/list_bags` | Stored bags (newest first) |
| `srv/DeleteBag` | collector `~/delete_bag` | Delete exactly one bag |
| `srv/StartCollection` | ground control `~/start` | Start recordings on remotes/groups |
| `srv/StopCollection` | ground control `~/stop` | Stop, download (and delete) |

Every machine that talks to a collector must be built from the **same** version of this
package. ROS 2 Jazzy matches endpoints by type hash, so mismatched definitions just look like
"not reachable".

This package replaces `mastcam_interfaces` and `gantry_lidar_interfaces` for remote recording.
Those are kept only for the legacy nodes and `scan_automation`.
