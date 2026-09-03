from setuptools import find_packages, setup

package_name = 'mastcam_service'

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
    maintainer='ryan',
    maintainer_email='ryan123rudder@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mastcam_client=mastcam_service.mastcam_client:main',
            'mastcam_capture_service=mastcam_service.mastcam_capture_service:main',
        ],
    },
)
