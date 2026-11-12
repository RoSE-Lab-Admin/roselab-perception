"""Default topic selection for the Rosey telemetry bag ("roseybag").

Kept free of ROS imports so launch files can import it cheaply and the
node and launch arguments share a single source of truth.
"""

# Topics always recorded (when present) in the roseybag.
DEFAULT_ROSEY_TOPICS = [
    '/FitRosey_passive/pose',
    '/cmd_vel',
    '/dynamic_joint_states',
    '/initialpose',
    '/joint_states',
    '/robot_description',
    '/rosout',
    '/tf',
    '/tf_static',
    '/plan',
    '/optimal_trajectory', # Should merge between Nav2, Rosey avionics, and Rosey wheelcam topics
    '/imu/data',
    '/wheelcam/cam_0/synced/camera_info',
    '/wheelcam/cam_0/synced/image_raw/compressed',
    '/wheelcam/cam_1/synced/camera_info',
    '/wheelcam/cam_1/synced/image_raw/compressed',
    '/wheelcam/cam_2/synced/camera_info',
    '/wheelcam/cam_2/synced/image_raw/compressed',
    '/wheelcam/cam_3/synced/camera_info',
    '/wheelcam/cam_3/synced/image_raw/compressed',
]

# Namespaces whose currently-advertised topics are auto-discovered and
# appended to the roseybag topic list at record time.
DEFAULT_ROSEY_NAMESPACES = [
    '/joy',
    '/roseybot_base_controller',
    '/optitrack', # metadata and statistics
]


def parse_name_list(value):
    """Normalize a parameter value into a list of names.

    Accepts a list/tuple of strings, or a single string of names separated
    by commas and/or whitespace (handy from the launch command line).
    Empty entries are dropped.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = value.replace(',', ' ').split()
    return [str(v).strip() for v in value if str(v).strip()]


def normalize_topic(name):
    """Ensure a topic name is absolute (leading '/', no trailing '/')."""
    name = name.strip()
    if not name.startswith('/'):
        name = '/' + name
    return name.rstrip('/') or '/'


def normalize_namespace(name):
    """Ensure a namespace is absolute and ends with '/' for prefix matching."""
    return normalize_topic(name).rstrip('/') + '/'
