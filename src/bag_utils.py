# This module contains various routines for visualizing things like message density in bags
import seaborn as sns
from pose_utils import load_trajectory
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path
from typing import Union, List, Any
import numpy as np
from tqdm import tqdm

def load_topics_times(bag_file: Union[str, Path], topics: Union[List[str], str]):
    # Use similar strategy as pose_utils or the RGBD bag reading stuff
    topics_times_map = {}
    CHECK_TOPIC = True

    if isinstance(topics, str):
        topics = [topics]
    elif topics is None:
        # If None, do it for all topics
        CHECK_TOPIC = False

    bag_path = Path(bag_file)
    N = 0

    # Extract poses
    with AnyReader([bag_path]) as reader:
        print(f"Unpacking bag message times for topics {topics}")
        for conn in reader.connections:
            if CHECK_TOPIC:
                if conn.topic not in topics:
                    # If topic is not the topic specified, skip
                    continue

            rows = []
            desc = f"{bag_path.name}:{conn.topic}"
            N = conn.msgcount
            times = np.zeros(N)

            # Just care about unpacking message times, not raw data
            for i, (_, ts, _) in tqdm(enumerate(reader.messages(connections=[conn])),
                                    total=N, desc=desc):
                times[i] = ts / 1.e9

            topics_times_map[conn.topic] = (N, times)

    return topics_times_map

def plot_message_density(times_map):
    # Extract only the table object
    # Adjust to seconds from nanoseconds
    tmp = {k:(v[1]) for k,v in times_map.items()}
    earliest_time = np.inf
    for v in tmp.values():
        earliest_time = min(v[0], earliest_time)

    print(f"{earliest_time=}")
    for v in tmp.values():
        v -= earliest_time

    sns.histplot(tmp, kde=True, title="Message Density", alpha=0.5) # Times map should be a dictionary of time vectors for which a kde will be constructed for each

if __name__=="__main__":
    # Load trajectory
    import sys

    # Requires bag and topic name for pose
#    times, transforms = load_trajectory(sys.argv[1], sys.argv[2])

    # Calculate message density via kde
    # Setup a seaborn plot axes object
#    plot_message_density({'Pose Message Density': times})

    plot_message_density(load_topics_times(sys.argv[1], sys.argv[2:]))

    plt.show()
