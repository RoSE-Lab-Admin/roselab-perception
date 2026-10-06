from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'remote_data_collector'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ryan Hartzell',
    maintainer_email='ryan_hartzell@mines.edu',
    description='Remote rosbag recording via a ROS 2 action server with an HTTP download '
                'sidecar, plus the `rdc` command line client.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'remote_data_collector = remote_data_collector.collector_node:main',
            'rdc = remote_data_collector.cli:entry_point',
        ],
    },
)
