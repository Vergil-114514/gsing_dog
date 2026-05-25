"""Launch YOLO detector and 3D detection calculator nodes."""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('detection_3d')
    params_file = os.path.join(pkg_dir, 'config', 'detection_params.yaml')

    return LaunchDescription([
        Node(
            package='detection_3d',
            executable='yolo_detector',
            name='yolo_detector',
            parameters=[params_file],
            output='screen',
        ),
        Node(
            package='detection_3d',
            executable='detection_3d_calculator',
            name='detection_3d_calculator',
            parameters=[params_file],
            output='screen',
        ),
        Node(
            package='detection_3d',
            executable='arm_serial_bridge',
            name='arm_serial_bridge',
            parameters=[params_file],
            output='screen',
        ),
    ])
