# This module contains various routines for visualizing things like message density in bags
import argparse
import seaborn as sns
from pose_utils import load_trajectory
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from pathlib import Path
from typing import Union, List, Any
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

def _convert_filename_to_path(f: Union[List[str], str, Path]) -> List[Path]:
    if isinstance(f, list):
        return [Path(fi) for fi in f]
    else:
        return [Path(f)]

def print_bag_info(bag_files: Union[List[str], str, Path]):
    bag_paths = _convert_filename_to_path(bag_files)
    with AnyReader(bag_paths) as reader:
        # Times are in nanoseconds, so we divide by 1e9 to get seconds
        start_sec = reader.start_time / 1e9
        end_sec = reader.end_time / 1e9
        duration = end_sec - start_sec

        # Format timestamps into human-readable strings
        start_str = datetime.fromtimestamp(start_sec).strftime('%b %d %Y %H:%M:%S.%f')[:-3]
        end_str = datetime.fromtimestamp(end_sec).strftime('%b %d %Y %H:%M:%S.%f')[:-3]

        # Print overall bag statistics
        print(f"Start:         {start_str}")
        print(f"End:           {end_str}")
        print(f"Duration:      {duration:.3f} seconds")
        print(f"Total Msgs:    {reader.message_count}")
        print("-" * 60)

        # Print a cleanly aligned table of topics
        print("Topics:")
        # reader.topics is a dictionary mapping topic names to TopicInfo objects
        for topic, info in sorted(reader.topics.items()):
            # Left-align topic name to 35 chars, right-align count to 8 chars
            print(f"  {topic:<35} | {info.msgcount:>8} msgs | {info.msgtype}")

def load_topics_times(bag_files: Union[List[str], str, Path], topics: Union[List[str], str]):
    # Use similar strategy as pose_utils or the RGBD bag reading stuff
    CHECK_TOPIC = True

    if isinstance(topics, str):
        topics = [topics]
    if len(topics) == 0:
        # If empty list, do it for all topics
        CHECK_TOPIC = False

    bag_paths = _convert_filename_to_path(bag_files)

    N = 0

    topics_times_map = defaultdict(list)

    # Extract poses
    with AnyReader(bag_paths) as reader:
        print(f"Unpacking bag message times for topics {topics if topics else '*'}")
        START_TIME = reader.start_time / 1e9
        for conn in reader.connections:
            if CHECK_TOPIC:
                if conn.topic not in topics:
                    # If topic is not the topic specified, skip
                    continue

            rows = []
            desc = f"{conn.topic}"
            N = conn.msgcount
            times = [np.nan]*N

            # Just care about unpacking message times, not raw data
            for i, (_, ts, _) in tqdm(enumerate(reader.messages(connections=[conn])),
                                    total=N, desc=desc):
                times[i] = ts / 1.e9

            topics_times_map[conn.topic].extend(times)

    # Convert to numpy arrays
    flagged = []
    for k,v in topics_times_map.items():
        if len(v)==0:
            print(f"[INFO] Removing topic {k} due to 0 messages received.")
            flagged.append(k)
            continue
        topics_times_map[k] = np.array(v)

    for k in flagged:
        del topics_times_map[k]

    return topics_times_map, START_TIME

def plot_message_density(times_map, START_TIME):
    # Extract only the table object
    # Adjust to seconds from nanoseconds
    earliest_time = START_TIME
    # for v in times_map.values():
    #     earliest_time = min(v[0], earliest_time)

    print(f"{earliest_time=}")
    for v in times_map.values():
        v -= earliest_time

    sns.histplot(times_map, bins=100, kde=True, element="step", fill=False, alpha=0.5, linestyle='--') # Times map should be a dictionary of time vectors for which a kde will be constructed for each
    plt.title("Message Mass and Density (KDE) over Time")

def _setup_parser():
    parser = argparse.ArgumentParser(
        description="Process multiple ROS bags and filter by specific topics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --bags: Required, accepts multiple arguments
    parser.add_argument(
        "--bags", 
        type=str, 
        nargs='+', 
        required=True, 
        help="Path(s) to one or more ROS bag files or directories."
    )

    # --topics: Optional, accepts multiple arguments
    parser.add_argument(
        "--topics", 
        type=str, 
        nargs='+', 
        default=[], # Defaults to an empty list if the user omits it
        help="Specific topic(s) to extract. If omitted, processes all topics."
    )

    return parser

if __name__=="__main__":
    import sys

    parser = _setup_parser()
    args = parser.parse_args()

    # Update with argparse for topic filtering, wildcard, and multi-bag support
    print_bag_info(args.bags)

    plot_message_density(*load_topics_times(args.bags, args.topics))

    plt.show()
