from setuptools import setup
import os
from glob import glob

package_name = 'detection_3d'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt') + glob('models/*.onnx')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leon',
    maintainer_email='leon@todo.com',
    description='3D object detection using YOLOv8 and Orbbec Gemini Pro depth camera',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yolo_detector = detection_3d.yolo_detector_node:main',
            'detection_3d_calculator = detection_3d.detection_3d_calculator_node:main',
            'arm_serial_bridge = detection_3d.arm_serial_bridge_node:main',
        ],
    },
)
