"""
RoSE Lab Motive Data parsing and visualization for pose error evaluation.

Author: Ryan Hartzell, ryan_hartzell@mines.edu
"""

import itertools
import numpy as np
from scipy.spatial.transform import Rotation as R
import rerun as rr
import sys
import matplotlib.pyplot as plt

# Set up inferno colormap
import matplotlib as mpl
cmap = mpl.colormaps["inferno"]

def create_heatmap(times, rots, trans, errors, sensor_locs=None):
        mask = np.isclose(errors, 0.0)

        # Get error summary stats
        print("Summary statistics on Mean Marker Error of each solution: ")
        print("Average MME:         ", np.mean(errors[~mask]))
        print("+/- Sigma MME:       ", np.std(errors[~mask]))
        print("Max MME:             ", np.max(errors[~mask]))

        # Using all this data, plot absolute position of rover over time with corresponding error
        # First set up world frame

        # Next transforms are already world to rover, so we can just use translation directly here)

        # Create the hexbin plot
        fig, ax = plt.subplots()

        hm = plt.hexbin(trans[:,0], trans[:,1], C=errors, reduce_C_function=np.mean, gridsize=100, cmap='inferno', vmin=0.0)

        # Add a colorbar for reference
        cbar = fig.colorbar(hm)

        # Add a label to the colorbar
        cbar.set_label('Maximum MME [mm]')

        plt.scatter(trans[mask][:,0], trans[mask][:,1], color='r', label='Rover Lost Event \n(Outside of Tracking Volume)')

        if sensor_locs is not None:
                plt.scatter(sensor_locs[:,0], sensor_locs[:,1], 2**10, marker='*', color='b', label='Sensor Locations')

        plt.legend()

        plt.title("Mean Marker Error Along X-Y Plane of Rover Trajectory")
        plt.xlabel("X [mm]")
        plt.ylabel("Y [mm]")

        # Show the plot
        plt.show()

# This should add some static frustums with reference frames same as 
def plot_camera_poses(camera_trans, camera_rots):
        for i,(pos,rot) in enumerate(zip(camera_trans, camera_rots)):
                # Make a camera body in the world frame
                cam_body = rr.Arrows3D(
                        vectors=[[1,0,0],[0,1,0],[0,0,1]],
                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                )

#                rr.log(
#                        f"world/cam{i}",
#                        cam_body,
#                        static=True
#                )

                rr.log(
                        f"world/cam{i}",
                        cam_body,
                        rr.Transform3D(
                                rotation=rr.Quaternion(xyzw=rot),
                                relation=rr.TransformRelation.ChildFromParent,
                                translation=pos,
                                scale=1,
                        ),
                        static=True
                )

                # Could also add in a viewing frustum and technically (eventually) the full data streams!!!

def pose_chain_viz(times, rots, trans, errors, sensors_pos, sensors_rot):
        # Should change this to use https://github.com/rerun-io/rerun/tree/docs-latest/examples/python/structure_from_motion as an example

        rr.init("optitrack_pose_solution_error", spawn=True)

        # Init the world frame and rover coordinate system vectors
#        rr.log("world", rr.ViewCoordinates.RFU, static=True)  # +X-right, +Y-forward, +Z-up
        # rr.log("world", rr.ViewCoordinates.LUF, static=True)  # +X-left, +Y-up, +Z-forward <= Motive native default ref frame
        rr.log("world", rr.ViewCoordinates.FLU, static=True)

        rr.log("plot/mme", rr.SeriesLines(colors=[240, 45, 58]), static=True)

#       # Make a rover body in the world frame
#       rover_body = rr.Arrows3D(
#               vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
#               colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
#       )
#
#       rr.log(
#               "world/rover",
#               rover_body,
#               static=True
#       )

        errmax = errors.max()

        # Correct from mm -> m
        trans = trans / 1000.
        rover_body = rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        )

        rr.log(
                "world/rover",
                rover_body,
                static=True
        )
        # For some reason not seeing the transforms take effect....
        for i in range(len(times)):
                t = times[i]
                rot = rots[i]
                pos = trans[i]
                err = errors[i]

                rr.set_time("elapsed", duration=t)

                trans_i = rr.Transform3D(
                        rotation=rr.Quaternion(xyzw=rot),
                        relation=rr.TransformRelation.ChildFromParent,
                        translation=pos,
                        scale=1,
                )

                # Update transform for world -> rover (not sure if absolute or relative..... need to double check that)
                rr.log(
                        "world/rover",
                        trans_i,
                )


                # Change the color of locations by error value? Probably need to do this all at once with send_columns and the latest set of points and times
#                rr.log(
#                        "world/rover/location_history",
#                        rr.Points3D([[0,0,0]], radii=0.1, colors=cmap(err/errmax)),
#                        trans_i,
#                )

        # Plot all cameras as static entities over all times
        plot_camera_poses(sensors_pos / 1000., sensors_rot)

        # Finally update ALL OF OUR ERRORS as a trace at once
        rr.send_columns(
                "plot/mme",
                indexes=[rr.TimeColumn("elapsed", duration=times)],
                columns=rr.Scalars.columns(scalars=errors)
        )

def plot_lidar_system_approximate(all=True):
        # Use the approximate translation and rotation matrices for this so it's symmetric
        # Define center position and orientation
        # Define the positions relative to center
        # Define the orientations relative to center

        return

def plot_point_cloud():
        # Requires open3d
        return

def _parse_metadata(fname):
        # Grab locations of all sensors and return
        sensors_rot = []
        sensors_pos = []

        with open(fname, 'r') as f:
                for l in f:
                        s = l.split(',')
                        if s[0] == "Camera":
                                # If we match the start of the line tag, grab data and push back into sensors list
                                # APPARENTLY motive doesn't respect our units nor our axis conventions, so these need to be reordered and converted to mm from mm
                                # Currently looks like ordering is Y-up, Z-forward, we want Z-up, Y-forward, which means technically some of these may need to be rotated?
                                sensors_pos.append([float(s[4]), float(s[2]), float(s[3])])
                                # Need to rotate quaternion into new reference frame
                                old_quat = R.from_quat([float(si) for si in s[5:9]])
                                permute = np.matrix(np.c_[[0,0,1],[1,0,0],[0,1,0]]) # This is effectively the matrix we used to correct the translation axes
                                sensors_rot.append(R.from_matrix(permute @ old_quat.as_matrix() @ permute).as_quat()) # Trick here so we still end up with a rh'd system

        return (np.asarray(sensors_pos) * 1000., np.asarray(sensors_rot))

def _parse_pose_line(l):
        # Split line by commas
        s = l.split(',')

        # Grab time entry as float [s]
        t = float(s[1])

        # Grab quat entries, as list of floats [rad?]
        quat = [float(x) for x in s[2:6]]

        # Grab pos entries, as list of floats [mm]
        pos = [float(x) for x in s[6:9]]

        # Grab error column as float [mm]
        error = float(s[9])

        return t, quat, pos, error

def _parse_pose(fname):
        # Create master lists: rotations, positions, errors
        times = []
        rots = []
        trans = []
        errors = []

        # Open file
        with open(fname, 'r') as f:
                # Seek to start of data (line 8) then start reading data
                for l in itertools.islice(f, 8, None):
                        # For each line in file, parse each line
                        try:
                                t, q, p, e = _parse_pose_line(l)
                        except:
                                continue

                        # For line data, append to master lists as appropriate objects
                        times.append(t)
                        rots.append(q)
                        trans.append(p)
                        errors.append(e)

                        # Get next line
                        l = f.readline()

        # Create numpy arrays of data and return
        times = np.asarray(times)
        rots = np.asarray(rots)
        # rots = R.from_quat(q) # Scipy defaults to [x,y,z,w] convention :)
        # rots = [R.from_quat(q) for q in rots] # Scipy defaults to [x,y,z,w] convention :)
        trans = np.asarray(trans)
        errors = np.asarray(errors)

        return times, rots, trans, errors

if __name__=="__main__":
        if len(sys.argv) != 4 or ((mode := sys.argv[3]) not in ("--heatmap", "--posegraph", "--all", "-hm", "-pg", "-a")):
                raise ValueError("""
Usage: WAYLAND_DISPLAY= eval.py <filename 1> <filename 2>  <mode> \n\n 
        filename 1 = pose data from Motive in CSV format \n
        filename 2 = metadata from Motive in CSV format \n
        mode       = controls program flow. May be one of: \n
                        --heatmap [-hm]   : displays matplotlib heatmap of max MME over rover trajectory
                        --posegraph [-pg] : displays pose over time in rerun application, and the MME trace
                        --all [-a]        : displays visuals from both modes
"""
)

        pfname = sys.argv[1]
        sfname = sys.argv[2]

        print(f"Parsing pose file {pfname}...")
        data = _parse_pose(pfname)
        print("Done!")

        print(f"Parsing sensor locations from metadata file {sfname}...")
        sensors = _parse_metadata(sfname)
        print("Done!")

        # Report errors via x-y plane projection
        if mode in ("--heatmap", "-hm", "--all", "-a"):
                create_heatmap(*data, sensor_locs=sensors[0])

        # Now run our pose chain to transform this body over time
        if mode in ("--posegraph", "-pg", "--all", "-a"):
                pose_chain_viz(*(data + sensors))
