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
 	OR
    ssh rosey@192.168.2.50 -> PW: roseyrover
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
		*OR*
		ssh dev@192.168.2.104 -> PW: regolith 		
        cd roselab-perception
        ./scripts/launch_realsense_d456_latest.sh
    ```
3. Open another tab on the slade and run:
    ```bash
        ssh dev@192.168.2.104 -i .ssh/id_rsa_ansible
		*OR*
		ssh dev@192.168.2.104 -> PW: regolith 		
        cd roselab-perception
        source venv/bin/activate
        source ros/install/setup.bash
        ros2 run mastcam_service mastcam_capture_service  
    ```


## WHEEL CAM - x4 RGB Cameras

1. Open WSL terminal on Slade
2. Run: 
    ```bash
        ssh picam@192.168.2.51 -> PW: roseycam
        ./boot.sh
    ```
3. Open WSL terminal on the slade
4. In the home directory run:
	```bash
		./perception_boot.sh
	```

## LIDAR & GANTRY SYSTEM

You can either use NoMachine on the NUC for remote access, like so:

1. NoMachine -> Gantry Computer PW: M3Robotics 
2. Open NoMachine application and select the Gantry Computer, wait for password prompt and desktop to show.

OR via SSH:

1. ```bash
   ssh gantry@192.168.2.99 -i .ssh/id_rsa_ansible
   ```

Then:
- Open first tab (start up the lidars):

    ```bash
        cd ~/gantry_control
        ./run_roselab_perception.sh
    ```
- Open second tab (run gantry capture service):
    ```bash
        cd ~/roselab-perception
        source /opt/ros/jazzy/setup.bash
        source ros/install/setup.bash
        ros2 run gantry_services gantry_capture_service
    ```
Finally, ssh into the lattepanda from either NUC or WSL on Slade -> ssh gantry_lattepanda@192.168.2.4 -> PW: M3Robotics
- Open tab:
    ```bash
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
1. Open terminal on slade (start service for data capture):
    ```bash
        cd roselab-perception/ros
        source install/setup.bash
        ros2 launch perception_ground_control launch_ground.py duration:='60.0' # The duration is in seconds, and should be formatted like shown as a float
    ```
2. Check all needed topics are being published with ros2 topic list
3. Record testbed on Reolink
4. Open new tab on slade (do data capture):
    ```bash
        cd roselab-perception/ros
        source install/setup.bash
        cd ../scripts
        ./capture_lidar_once.sh # To capture a single lidar bag
        *OR*
        ./capture_rosey_mastcam_once.sh # To capture rosey and mastcam bags until [ENTER] is pressed
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

------------------------------------------------------------------------


# PROCESS FLOW


![process flow](process-flow.jpg "Process Flow Diagram")


