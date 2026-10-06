# A bag folder stored in a remote_data_collector's data directory.

# Folder name, e.g. "mastcam_bags_2026-10-06T12-34-56".
string name
# Absolute path on the collector host.
string path
# Base URL of the folder on the collector's HTTP sidecar, ending in "/" ("" if HTTP disabled).
# File i is downloadable at download_url + files[i].
string download_url
# Paths of every file in the folder, relative to the folder (POSIX separators).
string[] files
# Size in bytes of each entry of `files` (same order).
uint64[] file_sizes
# Sum of file_sizes.
uint64 size_bytes
# True when the bag has been finalized (metadata.yaml exists and it is not being recorded).
bool complete
# True while a recorder is writing to this bag.
bool active
# Last modification time of the folder contents.
builtin_interfaces/Time modified
