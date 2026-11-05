from pathlib import Path
from dataclasses import asdict
from typing import List, Tuple

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib as mpl
import matplotlib.pyplot as plt

import cv2
from tqdm import tqdm
from rosbags.highlevel import AnyReader

# Useful lib for abstracting loading and manipulating pose streams!!!
import robotdatapy as rdp

def rotation_magnitude(rot):
    return np.linalg.norm(R.from_matrix(rot).as_rotvec())

def translation_magnitude(trans):
    return np.linalg.norm(trans)

# SEP arg controls whether this func returns combined or R,t error components
def pose_error(p1, p2, alpha=0.5, beta=0.5, sep=False):
    dp = np.linalg.inv(p1) @ p2
    if sep:
        return rotation_magnitude(dp[:3,:3]), translation_magnitude(dp[:3,3])

    else:
        normed = np.array([alpha, beta])
        normed /= np.linalg.norm(normed)

        return normed[0]*rotation_magnitude(dp[:3,:3]) + normed[1]*translation_magnitude(dp[:3,3])

def are_quaternions_close(q1: np.ndarray, q2: np.ndarray) -> bool:
    """Check if two quaternions represent the same rotation direction."""
    return np.dot(q1, q2) >= 0.0

def inverse_sign_quaternion(q: np.ndarray) -> np.ndarray:
    return -q

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)

def average_quaternion(cumulative: np.ndarray, new_q: np.ndarray, first_q: np.ndarray, count: int) -> np.ndarray:
    """
    Incrementally averages quaternions using a cumulative sum and normalizing the result.

    Args:
        cumulative: np.ndarray of shape (4,), running sum of quaternions.
        new_q: np.ndarray of shape (4,), new quaternion to add.
        first_q: np.ndarray of shape (4,), first quaternion used for sign reference.
        count: int, number of quaternions added including this one.

    Returns:
        np.ndarray: normalized average quaternion of shape (4,)
    """
    # Since we expect them to be close, any non-close ones are due to the double-cover problem
    if not are_quaternions_close(new_q, first_q):
        new_q = inverse_sign_quaternion(new_q)

    cumulative += new_q
    average = cumulative / count
    return normalize_quaternion(average)


def get_average_pose(bag_path: str, topic:str):
    """
    Given a bag path, returns the average postiion and orientation
    Args:
    bag_path: string path to bag
    topic: string, topic name (unused currently)

    Returns:
    avg_position:np.array(3,1), avg_orientation:np.array(4,1) 
    """
    bag_path = Path(bag_path)
    all_data = []
    # Extract poses
    with AnyReader([bag_path]) as reader:
        for conn in reader.connections:
            rows = []
            desc = f"{bag_path.name}:{conn.topic}"
            for _, ts, raw in tqdm(reader.messages(connections=[conn]),
                                   total=conn.msgcount, desc=desc):
                msg = reader.deserialize(raw, conn.msgtype)
                row = {'stamp_ns': ts}
                row.update(pd.json_normalize(asdict(msg)).iloc[0].to_dict())
                rows.append(row)
            all_data.append(rows)

    # Average pose
    pos_x_sum=0
    pos_y_sum=0
    pos_z_sum=0
    avg_orientation = np.zeros(4)
    first_q = None
    for conn in all_data: # Should only be one
        for i, frame in enumerate(conn, start=1):
            # Construct quat
            q = np.array([
                frame["pose.orientation.x"],
                frame["pose.orientation.y"],
                frame["pose.orientation.z"],
                frame["pose.orientation.w"]
            ])

            # Accum positions
            pos_x_sum += frame["pose.position.x"]
            pos_y_sum += frame["pose.position.y"]
            pos_z_sum += frame["pose.position.z"]

            # First quat
            if first_q is None:
                first_q = q.copy()

            # Accum orientation
            avg_orientation = average_quaternion(avg_orientation, q, first_q, i)
        avg_position = np.array([pos_x_sum/len(conn), pos_y_sum/len(conn), pos_z_sum/len(conn)])
    return avg_position, avg_orientation

def load_trajectory(bag_path: str, topic:str, return_msgs:bool = False):
    """
    Given a bag path, returns time samples and corresponding transforms as numpy arrays
    Args:
    bag_path: string path to bag
    topic: string, topic name (unused currently)

    Returns:
    times:np.array(N,1), transforms:np.array(N,4,4)
    """
    bag_path = Path(bag_path)
    all_data = []
    N = 0

    msgtype = None

    # Extract poses
    with AnyReader([bag_path]) as reader:
        for conn in reader.connections:
            if conn.topic != topic:
                # If topic is not the topic specified, skip
                continue

            rows = []
            desc = f"{bag_path.name}:{conn.topic}"
            N = conn.msgcount
            msgtype = conn.msgtype
            for _, ts, raw in tqdm(reader.messages(connections=[conn]),
                                   total=conn.msgcount, desc=desc, mininterval=1.0):
                msg = reader.deserialize(raw, conn.msgtype)
                row = {'stamp_ns': ts}
                row.update(pd.json_normalize(asdict(msg)).iloc[0].to_dict())
                rows.append(row)
            all_data.append(rows)

    # Convert to numpy matrices
    tfs = np.tile(np.eye(4), (N,1,1))
    times = np.zeros(N)

    print(f"Processing {N} transforms...")

    # For now, just do this...
    print(f"Using {msgtype=}")
    if msgtype == 'gantry_interfaces/msg/GantryState':
        for conn in all_data:
            for i, frame in enumerate(tqdm(conn)):
                # For now, assume 2D plane, orientation doesn't change
                tfs[i,:3,:3] = np.eye(3)

                tfs[i,:3,3] = np.r_[
                    frame["encoder_c"], # Carriage is x dir
                    np.nanmean([frame["encoder_e"], frame["encoder_w"]]), # Take average for y dir
                    0. # This would normally be set to the actual height (hopefully constant...) of the cart w.r.t MLSS origin
                ]

                times[i] = frame["stamp_ns"]

    else: # Assume regular posestamped msgtype
        for conn in all_data:
            for i, frame in enumerate(tqdm(conn)):
                tfs[i,:3,:3] = R.from_quat([
                    frame["pose.orientation.x"],
                    frame["pose.orientation.y"],
                    frame["pose.orientation.z"],
                    frame["pose.orientation.w"]
                ]).as_matrix()

                tfs[i,:3,3] = np.r_[
                    frame["pose.position.x"],
                    frame["pose.position.y"],
                    frame["pose.position.z"]
                ]

                times[i] = frame["stamp_ns"]

    # Return 4x4 transforms extracted from pose message stream
    return np.asarray(times), np.asarray(tfs)


def plot_trajectory(times, tfs, show_up=False, show_frames=False, up='Z'):
    if show_up:
        # Show UP vector, so don't show full ref frame
        show_frames = False

        # Now decide which way is up
        if up=='Z':
            up_vec = np.array([0,0,1])
        elif up=='Y':
            up_vec = np.array([0,1,0])
        elif up=='X':
            up_vec = np.array([1,0,0])
        elif isinstance(up, [np.array, list]) and len(up)==3 and np.isclose(np.linalg.norm(up), 1.):
            up_vec = np.asarray(up)
        else:
            raise ValueError(f"Unsupported specification for 'up' vector: {up}")

    # For now, let's just support position for viz
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')

    # Normalize to [0,1]
    t = (tmp:= (times - times.min())) / tmp.max()

    # This could be changed to a line plot given nans for "missing" tfs
    ax.scatter3D(tfs[:,0,3], tfs[:,1,3], tfs[:,2,3], c=t, cmap='plasma', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')

    plt.show()

def plot_pose_errors(times, transforms, sep=True):
    if sep:
        errs = np.zeros(2*(len(times)-1)).reshape(len(times)-1,2)
        for i,(p1,p2) in enumerate(zip(transforms[0:-1], transforms[1:])):
            errs[i,:] = pose_error(p1,p2,sep=True)

        # Plot on twin axes
        fig, axl = plt.subplots()

        axl.scatter(times[1:], errs[:,0], color='tab:red') # , label="Rotational Error") # err(R)
        axl.set_ylabel("Rotational Error")
        axl.tick_params(axis='y', labelcolor='tab:red')

        axr = axl.twinx()
        axr.scatter(times[1:], errs[:,1]) # , label="Translation Error") # err(t)
        axr.set_ylabel("Translational Error", color='tab:blue')
        axr.tick_params(axis='y', labelcolor='tab:blue')

        plt.xlabel("Time $[s]$")

    else:
        errs = np.zeros(len(times)-1)
        for i,(p1,p2) in enumerate(zip(transforms[0:-1], transforms[1:])):
            errs[i] = pose_error(p1,p2)
        plt.xlabel("Time $[s]$")
        plt.ylabel("Total Pose Error")
        plt.plot(times[1:], errs)

    plt.show()

def valid_pose_mask(tfs, ortho_tol=1e-3):
    """
    Boolean mask of poses that are finite and have a proper rotation (orthonormal, det=+1).
    """
    finite = np.all(np.isfinite(tfs.reshape(len(tfs), -1)), axis=1)
    rot_ok = np.zeros(len(tfs), dtype=bool)
    rots = tfs[finite, :3, :3]
    if len(rots):
        ortho_err = np.linalg.norm(rots @ np.transpose(rots, (0,2,1)) - np.eye(3), axis=(1,2))
        rot_ok[finite] = (ortho_err < ortho_tol) & (np.linalg.det(rots) > 0)
    return finite & rot_ok

def sliding_window_pose_error(times, tfs, window=5.0, order=2, stride=120, min_samples=5):
    """
    Local pose noise estimate over a centered sliding time window.

    For each window, all poses are expressed relative to the center pose
    (inv(T_c) @ T_j -> rotvec, translation), then a polynomial in time of
    degree `order` is fit to each of the 6 components. The RMS residual of
    that fit is the error for the window. order=0 measures deviation from the
    local mean pose (good for static data), order=1 removes constant
    velocity motion (good for a moving platform), order=2 also removes
    constant acceleration.

    Unlike consecutive differencing, this is insensitive to the sampling rate
    and does not count real motion as error.

    Args:
    times: np.array(N,) sample times in SECONDS
    tfs: np.array(N,4,4) transforms
    window: float, window width in seconds (centered on each evaluated sample)
    order: int, polynomial degree of the local motion model
    stride: int, evaluate every `stride`-th sample (speeds up long streams)
    min_samples: int, windows with fewer valid poses than this are NaN

    Returns:
    centers:np.array(M,) window center times, rot_err:np.array(M,) [rad], trans_err:np.array(M,) [tf units]
    """
    times = np.asarray(times, dtype=float)
    valid = valid_pose_mask(tfs)
    t_v, tf_v = times[valid], tfs[valid]
    min_samples = max(min_samples, order + 2) # need dof > 0 for a residual

    centers = t_v[::stride]
    rot_err = np.full(len(centers), np.nan)
    trans_err = np.full(len(centers), np.nan)

    lo = np.searchsorted(t_v, centers - window/2, side='left')
    hi = np.searchsorted(t_v, centers + window/2, side='right')

    for k, c in enumerate(tqdm(range(0, len(t_v), stride), desc="Sliding window error", mininterval=1.0)):
        if hi[k] - lo[k] < min_samples:
            continue

        rel = np.linalg.inv(tf_v[c]) @ tf_v[lo[k]:hi[k]]
        y = np.hstack([R.from_matrix(rel[:,:3,:3]).as_rotvec(), rel[:,:3,3]]) # (n,6)

        dt = t_v[lo[k]:hi[k]] - t_v[c]
        A = np.vander(dt, order + 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef

        dof = len(dt) - (order + 1)
        rot_err[k] = np.sqrt(np.sum(resid[:,:3]**2) / dof)
        trans_err[k] = np.sqrt(np.sum(resid[:,3:]**2) / dof)

    return centers, rot_err, trans_err

def plot_sliding_window_pose_error(times, tfs, window=0.5, order=1, stride=1, anomalies=None, show=True):
    """
    Plot the sliding window pose error (see sliding_window_pose_error) for
    rotation and translation on stacked axes. If an anomalies DataFrame from
    detect_pose_anomalies is given, gaps are shaded and other anomalies are
    marked with vertical lines.
    """
    centers, rot_err, trans_err = sliding_window_pose_error(times, tfs, window, order, stride)

    fig, (axr, axt) = plt.subplots(2, 1, sharex=True)
    axr.plot(centers, np.rad2deg(rot_err), color='tab:red', lw=1)
    axr.set_ylabel("Rot. RMS error [deg]")
    axt.plot(centers, trans_err, color='tab:blue', lw=1)
    axt.set_ylabel("Trans. RMS error [m]")
    axt.set_xlabel("Time $[s]$")
    fig.suptitle(f"Sliding window pose error (window={window}s, order={order}, stride={stride})")

    if anomalies is not None and len(anomalies):
        _mark_anomalies([axr, axt], anomalies)

    fig.tight_layout()
    if show:
        plt.show()
    return fig

def _robust_threshold(x, k):
    """median + k * (MAD scaled to std) of the finite values of x"""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.inf
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    return med + k * max(mad, 1e-12)

def detect_pose_anomalies(times, tfs, gap_factor=5.0, max_gap=None, jump_k=10.0,
                          max_lin_vel=None, max_ang_vel=None, verbose=True):
    """
    Find problems in a pose stream:
      - 'invalid':   NaN/inf entries or a non-rotation matrix
      - 'time':      duplicate or backwards timestamps
      - 'gap':       dt > gap_factor * median(dt), or dt > max_gap if given
      - 'jump_trans' / 'jump_rot': step speed between consecutive valid poses
                     above median + jump_k * MAD (robust outlier), or above
                     max_lin_vel / max_ang_vel [units/s, rad/s] if given

    Jumps are measured as speed (step / dt) so a gap alone does not register
    as a jump, but a pose that teleports across a gap still will.

    Args:
    times: np.array(N,) sample times in SECONDS
    tfs: np.array(N,4,4) transforms

    Returns:
    pd.DataFrame with columns [type, index, time, value, detail], sorted by time.
    'index' is the index into the original arrays (the later sample for step-based checks).
    """
    times = np.asarray(times, dtype=float)
    events = []

    def add(kind, idx, value, detail):
        for i, v in zip(np.atleast_1d(idx), np.atleast_1d(value)):
            events.append({'type': kind, 'index': int(i), 'time': times[i], 'value': float(v), 'detail': detail})

    # Invalid poses
    valid = valid_pose_mask(tfs)
    bad = np.flatnonzero(~valid)
    add('invalid', bad, np.zeros(len(bad)), "non-finite or non-rotation transform")

    # Timestamps
    dt_all = np.diff(times)
    nonmono = np.flatnonzero(dt_all <= 0) + 1
    add('time', nonmono, dt_all[nonmono-1], "duplicate/backwards timestamp [dt s]")

    pos_dt = dt_all[dt_all > 0]
    med_dt = np.median(pos_dt) if len(pos_dt) else np.nan
    gap_thresh = gap_factor * med_dt if max_gap is None else max_gap
    gaps = np.flatnonzero(dt_all > gap_thresh) + 1
    add('gap', gaps, dt_all[gaps-1], f"gap [s] (threshold {gap_thresh:.4g}s)")

    # Jumps between consecutive valid poses
    vidx = np.flatnonzero(valid)
    if len(vidx) > 1:
        rel = np.linalg.inv(tfs[vidx[:-1]]) @ tfs[vidx[1:]]
        step_t = np.linalg.norm(rel[:,:3,3], axis=1)
        step_r = np.linalg.norm(R.from_matrix(rel[:,:3,:3]).as_rotvec(), axis=1)
        dt = times[vidx[1:]] - times[vidx[:-1]]
        with np.errstate(divide='ignore', invalid='ignore'):
            vel_t = np.where(dt > 0, step_t / dt, np.nan)
            vel_r = np.where(dt > 0, step_r / dt, np.nan)

        for kind, vel, vmax, unit in [('jump_trans', vel_t, max_lin_vel, "units/s"),
                                      ('jump_rot', vel_r, max_ang_vel, "rad/s")]:
            thresh = _robust_threshold(vel, jump_k) if vmax is None else vmax
            hits = np.flatnonzero(vel > thresh)
            add(kind, vidx[1:][hits], vel[hits], f"step speed [{unit}] (threshold {thresh:.4g})")

    df = pd.DataFrame(events, columns=['type', 'index', 'time', 'value', 'detail'])
    df = df.sort_values(['time', 'type'], kind='stable').reset_index(drop=True)

    if verbose:
        print(f"Pose stream: {len(times)} samples, median dt {med_dt:.4g}s, {len(bad)} invalid")
        if len(df):
            print(df.groupby('type').agg(count=('index', 'size'), max_value=('value', 'max')))
        else:
            print("No anomalies found.")
    return df

def _mark_anomalies(axes, anomalies):
    colors = {'invalid': 'k', 'time': 'tab:purple', 'jump_trans': 'tab:orange', 'jump_rot': 'tab:green'}
    for ax in axes:
        for _, ev in anomalies.iterrows():
            if ev['type'] == 'gap':
                ax.axvspan(ev['time'] - ev['value'], ev['time'], color='0.6', alpha=0.4, lw=0)
            else:
                ax.axvline(ev['time'], color=colors.get(ev['type'], 'k'), alpha=0.6, lw=0.8, ls='--')
    # Legend proxies, only for types present
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Patch(color='0.6', alpha=0.4, label='gap')] if (anomalies['type'] == 'gap').any() else []
    handles += [Line2D([], [], color=c, ls='--', label=k) for k, c in colors.items() if (anomalies['type'] == k).any()]
    axes[0].legend(handles=handles, loc='upper right', fontsize='small')

def plot_pose_anomalies(times, tfs, anomalies=None, show=True, **detect_kwargs):
    """
    Plot per-step linear/angular speed between consecutive valid poses with
    gaps shaded and flagged anomalies marked. Runs detect_pose_anomalies if
    `anomalies` is not given. Returns (fig, anomalies).
    """
    times = np.asarray(times, dtype=float)
    if anomalies is None:
        anomalies = detect_pose_anomalies(times, tfs, **detect_kwargs)

    vidx = np.flatnonzero(valid_pose_mask(tfs))
    rel = np.linalg.inv(tfs[vidx[:-1]]) @ tfs[vidx[1:]]
    dt = np.diff(times[vidx])
    with np.errstate(divide='ignore', invalid='ignore'):
        vel_t = np.linalg.norm(rel[:,:3,3], axis=1) / dt
        vel_r = np.linalg.norm(R.from_matrix(rel[:,:3,:3]).as_rotvec(), axis=1) / dt

    fig, (axr, axt) = plt.subplots(2, 1, sharex=True)
    axr.plot(times[vidx[1:]], np.rad2deg(vel_r), color='tab:red', lw=0.8)
    axr.set_ylabel("Ang. speed [deg/s]")
    axt.plot(times[vidx[1:]], vel_t, color='tab:blue', lw=0.8)
    axt.set_ylabel("Lin. speed [m/s]")
    axt.set_xlabel("Time $[s]$")
    for ax in (axr, axt):
        ax.set_yscale('symlog', linthresh=1e-3) # jumps tend to dwarf everything else
    fig.suptitle(f"Pose stream anomalies ({len(anomalies)} flagged)")

    if len(anomalies):
        _mark_anomalies([axr, axt], anomalies)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, anomalies

#get_average_pose(Path("/home/ryan/lidarcalibrations/Trial_4cm_infradius_0.0slope_Trial3_07232025_10_37_30/mocap_bag"), "/CubeRover_V1/pose")

if __name__=="__main__":
    # Load pose bag
    import sys
    times, transforms = load_trajectory(sys.argv[1], sys.argv[2])

    # Visualize
    print(times[0::1000])
    print(transforms[0::1000])

    print("TFs shape: ", transforms.shape)

    plot_trajectory(times, transforms, up='Z')

    # Make these times relative to t0 and convert to seconds
    plot_pose_errors((times - times[0]) / 1e9, transforms, sep=True)

    # Stream health + local noise estimate
    t_s = (times - times[0]) / 1e9

    # For now ignore anomalies
    #  _, anomalies = plot_pose_anomalies(t_s, transforms, show=False)
    _ = plot_pose_anomalies(t_s, transforms, show=False)


    plot_sliding_window_pose_error(t_s, transforms)


    #    # RDP Tester
    #    bag_path = sys.argv[1] # path to bag
    #    topic = sys.argv[2] # Odometry or Pose msg
    #
    #    # Is this lazily loaded? Or all at once?
    #    pose_data = rdp.data.PoseData.from_bag(bag_path, topic=topic, time_tol=25.0, interp=True)
    #    print(pose_data)
    #
    #    # Make a version of my plot pretty much
    #    pose_data.plot2d(dt=0.1, trajectory=True, pose=False)     # plots only position every second
    #    pose_data.plot2d(dt=20.0, trajectory=False, pose=True, axis_len=0.25)     # plots coordinate frames of the poses every 5 seconds
    #
    #    plt.show()
