# Live status of one recording on a remote_data_collector.
# Used as RecordBag feedback and in GetStatus responses.

string STATE_STARTING=STARTING
string STATE_RECORDING=RECORDING
string STATE_STOPPING=STOPPING

# Action goal UUID as 32 lowercase hex characters.
string goal_id
# Final bag folder name (prefix + timestamp).
string bag_name
# One of the STATE_* constants.
string state
# Seconds since the recorder was started.
float64 elapsed_sec
# Requested duration in seconds; <= 0 means "until canceled".
float64 duration_sec
# Current on-disk size of the bag folder.
uint64 size_bytes
# Topics the recorder has subscribed to so far (best effort).
string[] subscribed_topics
