from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception_ground_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lexij',
    maintainer_email='lexi.j.hanlon@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ground_control=perception_ground_control.ground_control:main',
            'ground_control_service=perception_ground_control.ground_control_service:main',
            'run_session=perception_ground_control.session_runner:main',
        ],
    },
)
