from setuptools import find_packages, setup

package_name = 'aggregated_pointcloud'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/aggregated_pointcloud.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ryan Hartzell',
    maintainer_email='ryan_hartzell@mines.edu',
    description='Aggregates and downsamples PointCloud2s with pose transforms.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aggregated_pointcloud_node = aggregated_pointcloud.aggregated_pointcloud_node:main'
        ],
    },
)
