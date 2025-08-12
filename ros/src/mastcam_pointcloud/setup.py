from setuptools import find_packages, setup

package_name = 'mastcam_pointcloud'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rhart',
    maintainer_email='ryan_hartzell@mines.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mastcam_pointcloud_publisher = mastcam_pointcloud.mastcam_pointcloud_node:main'
        ],
    },
)
