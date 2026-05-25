"""Launch the Orbbec Gemini Pro camera driver with depth registration enabled."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    astra_camera_dir = get_package_share_directory('astra_camera')
    gemini_launch = os.path.join(astra_camera_dir, 'launch', 'gemini.launch.xml')

    return LaunchDescription([
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(gemini_launch),
            launch_arguments={
                'depth_registration': 'true',
                'enable_colored_point_cloud': 'false',
                'color_width': '640',
                'color_height': '480',
                'color_fps': '30',
                'depth_width': '640',
                'depth_height': '400',
                'depth_fps': '30',
            }.items(),
        ),
    ])
