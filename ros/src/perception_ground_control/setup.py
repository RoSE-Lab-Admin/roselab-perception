from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'perception_ground_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lexij',
    maintainer_email='lexi.j.hanlon@gmail.com',
    description='Ground control for perception data collection: drives any number of '
                'remote_data_collector action servers and downloads their bags.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ground_control=perception_ground_control.ground_control:main',
        ],
    },
)
