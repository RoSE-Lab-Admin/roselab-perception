# Perception Campaign

# START UP & INITIALIZATION

These startup routines result in all data streams, payloads, controls, and avionics to be initialized and published.

### NOTE: RUN THE FOLLOWING IN EVERY TERMINAL IF DOING ROS STUFF, AFTER SOURCING YOUR ROS ENVIRONMENT

	```bash
	ros2 daemon start
	```

## ROSEY - state interfaces, control interfaces, state topics

### Start up rover hardware
1. Open terminal on Slade
2. Run: 

	```bash
 	ssh rosey@192.168.2.50 -i ./.ssh/id_rsa_ansible
    ```

		# If issues happen with ssh:
		ssh-keygen -R 192.168.2.50 (this needs to be done every time sd card swap)

    ```bash
	source CubeRover/install/setup.bash
	ros2 launch roseybot_control hardware_startup.launch.py 
    ```

### Enable controller-based teleop of Rosey (Or follow Nav2 instructions in the CubeRover repo for waypoint following)
3. Open terminal on NUC
4. Run: 
    ```bash
        source CubeRover/install/setup.bash
        ros2 launch roseybot_control joystick.launch.py
    ```

## MAST CAM - RGBD Forward

1. Open terminal on Slade
2. Run: 

    ```bash
        ssh dev@192.168.2.104 -i .ssh/id_rsa_ansible
        cd roselab-perception
        ./scripts/launch_realsense_d456_latest.sh
    ```
3. Open another tab on the slade and run the mast cam's data collector (records on the Pi, serves bags over HTTP :8000):
    ```bash
        ssh dev@192.168.2.104 -i .ssh/id_rsa_ansible
        cd roselab-perception
        source venv/bin/activate
        source ros/install/setup.bash
        ros2 launch remote_data_collector remote_data_collector.launch.py name:=mastcam_front
        # (legacy, deprecated: ros2 run mastcam_service mastcam_capture_service)
    ```


## WHEEL CAM - x4 RGB Cameras

1. Open WSL terminal on Slade
2. Run: 
    ```bash
        ssh picam@192.168.2.51 -i .ssh/id_rsa_ansible
        ./boot.sh
    ```
3. Open WSL terminal on the slade
4. In the home directory run:
	```bash
		./perception_boot.sh
	```

## LIDAR & GANTRY SYSTEM

You can either use NoMachine on the NUC for remote access, like so:

1. NoMachine -> Gantry Computer PW: ********
2. Open NoMachine application and select the Gantry Computer, wait for password prompt and desktop to show.

OR via SSH:

Then:
- Open first tab (start up the lidars):

    ```bash
	    ssh gantry@192.168.2.99 -i .ssh/id_rsa_ansible
	    cd ~/gantry_control
        ./run_roselab_perception.sh
    ```
- Open second tab (run the gantry/lidar data collector):
    ```bash
	    ssh gantry@192.168.2.99 -i .ssh/id_rsa_ansible
		cd ~/roselab-perception
        source /opt/ros/jazzy/setup.bash
        source ros/install/setup.bash
        ros2 launch remote_data_collector remote_data_collector.launch.py name:=gantry
        # (legacy, deprecated: ros2 run gantry_services gantry_capture_service)
    ```
Finally, ssh into the lattepanda from WSL on Slade (or run command on Gantry Computer Miniforum)
- Open tab:
    ```bash
	    ssh gantry_lattepanda@192.168.2.4 -i .ssh/id_rsa_ansible
        cd /m3_robotics/gantry_control
        ./runSystem --no-gui
    ```

## OPTITRACK - Pose
1. Open Motive software on Slade and select CubeRover_V1 (or your specific rigid body) from "Assets" tab
2. MAKE SURE TO TURN ON OPTITRACK CAMERA LEDS IF THEY ARE OFF!
3. Open terminal on WSL
4. Run: ./optitrack.sh
5. Make sure it reads "Activated!" If not, restart. It should also list whichever rigid body is found and streaming. 

## FOXGLOVE - HUD

1. Open WSL terminal on Slade
2. Run: ./foxglove_boot.sh
3. Open foxglove desktop app on Slade, select address "ws://localhost:8765" url to open perception layout

## GROUND CONTROL - Session data collection and bagging

Every computer that produces data runs one **`remote_data_collector`** (a ROS 2 action server that
records bags locally and serves them over HTTP). **Ground control** on Slade starts/stops them and
downloads the bags into `/mnt/e/perception-data/<MMDDYYYY>/<HH-MM-SS>/`. You drive it all with
**`scripts/collect.sh`**. Details: [`remote_data_collector`](ros/src/remote_data_collector/README.md),
[`perception_ground_control`](ros/src/perception_ground_control/README.md).

1. Open terminal on slade and start the ground station collector (rover telemetry + mocap) and ground control:
    ```bash
        cd roselab-perception/ros
        source install/setup.bash
        ros2 launch remote_data_collector remote_data_collector.launch.py name:=ground_station &
        ros2 launch perception_ground_control launch_ground.py duration:=60.0 # lidar scan length (s)
    ```
2. Check that every collector is up and all needed topics are being published:
    ```bash
        ./scripts/collect.sh status            # remotes online/offline, disk space, HTTP
        ros2 run remote_data_collector rdc discover
    ```
3. Record testbed on Reolink
4. Open new tab on slade (do data capture):
    ```bash
        cd roselab-perception
        ./scripts/collect.sh trial              # guided loop: lidar scan -> trial -> lidar scan -> ...
        # or step by step:
        ./scripts/collect.sh scan               # one lidar scan (configured duration), downloaded when done
        ./scripts/collect.sh start -l run07     # mastcam + rosey telemetry ("trial" group)
        ./scripts/collect.sh stop               # stop everything, download, clean up remotes
        ./scripts/collect.sh session            # start a new session folder
        ./scripts/collect.sh stop all           # emergency stop of every remote
    ```
    `capture_lidar_once.sh`, `capture_rosey_mastcam_once.sh`, `stop_capture.sh` and `run_lidar_scan.sh`
    still work. They are now thin wrappers around `collect.sh`.

Which remotes exist and how they are grouped (`trial`, `lidar`, ...) lives in
`ros/src/perception_ground_control/config/ground_control.yaml`. Topic sets live in each collector's
config in `ros/src/remote_data_collector/config/`. To add a system, run a collector there and add
it to `ground_control.yaml`; no new code is needed.

To check that the whole stack works on a machine (no lab hardware needed: it spins up test
collectors, a test ground control and publishers on an isolated ROS domain), run
`scripts/selftest_collection.sh`. It should end with `SELF TEST PASSED`.

Build/update on every host after pulling:
```bash
cd ~/roselab-perception/ros
colcon build --packages-up-to perception_ground_control   # Slade
colcon build --packages-up-to remote_data_collector       # remote hosts (mastcam Pi, gantry, ...)
```
------------------------------------------------------------------------

# DATA INVENTORY
This is the full list of topics which should show up during data collection when queried from the Slade:

![topics list](topics "Topics List")

------------------------------------------------------------------------
# DATA UTILITIES

I've put together a few utilities for visualizing data, doing basic processing, and performing health checks on captured bags.

TODO: RH - NEED TO OUTLINE DIFFERENT CLI APPS HERE AND PROVIDE USER GUIDE FOR DEM RECONSTRUCTION AND CALIBRATION SCRIPTS.

------------------------------------------------------------------------
# DEBUGGING
If ROS2 topic list is not working or services that should be available are not showing, do 
```bash
	ros2 daemon stop
	ros2 daemon start
```

Data collection:
- `./scripts/collect.sh status` shows each remote as online/OFFLINE, with disk space, HTTP health and active recordings.
- `ros2 run remote_data_collector rdc status /mastcam_front` talks to one collector directly, bypassing ground control.
- A recording's result lists `missing_topics`: requested topics that never published. The recorder log
  for each bag is at `<data_dir>/.rdc/logs/<bag>.log` on the remote (also downloadable over HTTP).
- If a download fails, the bag stays on the remote. Fetch it with `rdc download <remote> <bag> --dest <dir>`
  (`rdc list <remote>` shows what's there).
- More: the Troubleshooting section of `ros/src/remote_data_collector/README.md`.
