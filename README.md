# Perception Campaign Fall 2025

# START UP & INITIALIZATION

These startup routines result in all data streams, payloads, controls, and avionics to be initialized and published.

## ROSEY - state interfaces, control interfaces, state topics

### Swap teensy and SD card for perception
1.	Unplug teensy from breadboard in Rosey
2.	Replace with OUR teensy flashed with ros2 control stack
3.	Replace SD card in avionics Pi

### Start up rover hardware
1. Open terminal on Slade
2. Run: 

	```bash
    ssh rosey@192.168.2.50 -> PW: roseyrover
    ```

		# If issues happen with ssh:
		ssh-keygen -R 192.168.2.50 (this needs to be done every time sd card swap)

    ```bash
	source CubeRover/install/setup.bash
	ros2 launch roseybot_control ros2_control.launch.py # Ryan check syntax here
	ros2 launch roseybot_control hardware_startup.launch.py # Ryan: not sure we need this and the IMU launch... should just have one launch for all hardware (avionics, IMU) and ros2 control
    ```

### Enable controller-based teleop of Rosey
3. Open terminal on NUC
4. Run: 
    ```bash
        source CubeRover/install/setup.bash
        ros2 launch roseybot_control joystick.launch.py # Ryan check syntax
    ```

## MAST CAM - RGBD Forward

1. Open terminal on Slade
2. Run: 

    ```bash
        ssh dev@192.168.2.104 -> PW: regolith 		# Ryan check this IP address
        cd roselab-perception
        source venv/bin/activate
        source install/setup.bash
    ```

## WHEEL CAM - x4 RGB Cameras

1. Open WSL terminal on Slade
2. Run: 
    ```bash
        ssh picam@19.168.2.51 -> PW: roseycam
        ./boot.sh
    ```

## LIDAR & GANTRY SYSTEM

1. NoMachine -> PW: M3Robotics 
2. Open NoMachine application and select the gantry computer (Latte Panda)
3. Once window opens showing LattePanda desktop, open three terminal tabs:
        
- In first tab:

    ```bash
        cd ~/gantry_control
        ./run_roselab_perception.sh
    ```
- In second tab:
    ```bash
        cd ~/gantry_control
        ./runSystem --no-gui
    ```
- In third tab:
    ```bash
        cd ~/roselab-perception/ros
        source /opt/ros/jazzy/setup.bash
        source install/setup.bash
        ros2 run gantry_services gantry_capture_service
    ```

## OPTITRACK - Pose

1. Open terminal on NUC
2. Run: ./optitrack.sh

## FOXGLOVE - HUD

1. Open WSL terminal on Slade
2. Run: ./foxglove_boot.sh
3. Open foxglove desktop app on NUC, select Slade address ws://... url to open perception layout

## GROUND CONTROL - Session data collection and bagging

<RYAN M - FILL THIS IN BASED ON SESSION RUNNING MANUAL OR AUTOMATED WORKFLOWS!>

------------------------------------------------------------------------


# DATA INVENTORY
The ground control must bag the following topics during each *trial*. Note that the wild card (asterisk) operator is all topics underneath that topic namespace:

### ROSEY
/CubeRover_V1/pose

/bno055/*

/cmd_vel

/dynamic_joint_states

/initialpose

/joint_states

/joy

/joy/*

/robot_description

/roseybot_base_controller/*

/rosout

/tf

/tf_static

And any others that y'all deem important to working with the data during playback.

### WheelCams
RH: TODO - these are same topics from mobility, Cameron should know their names

### MastCam
MastCam Pi should bag the following topics during each *trial*:

RH: TODO - list all topics we need here but basically already set up in the launch script within roselab-perception/scripts

Color, Aligned-depth-to-color, tfs, camera info topics, extrinsics, etc

------------------------------------------------------------------------


# PROCESS FLOW


![process flow](process-flow.jpg "Process Flow Diagram")
