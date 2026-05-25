"""Launch YOLO detector and 3D detection calculator nodes (no camera, no RViz)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('detection_3d')
    params_file = os.path.join(pkg_dir, 'config', 'detection_params.yaml')

    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(pkg_dir, 'models', 'best.pt'),
    )
    input_topic_arg = DeclareLaunchArgument(
        'input_topic', default_value='/camera/color/image_raw',
    )
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', default_value='/camera/depth/image_raw',
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic', default_value='/camera/depth/camera_info',
    )
    use_arm_bridge_arg = DeclareLaunchArgument(
        'use_arm_bridge', default_value='false',
        description='Launch arm serial bridge to STM32',
    )
    source_w_arg = DeclareLaunchArgument(
        'source_image_width', default_value='640',
        description='Source RGB image width used for YOLO inference',
    )
    source_h_arg = DeclareLaunchArgument(
        'source_image_height', default_value='480',
        description='Source RGB image height used for YOLO inference',
    )
    target_frame_arg = DeclareLaunchArgument(
        'target_frame', default_value='arm_base',
        description='Target frame for grasp/place coordinates',
    )
    camera_to_arm_xyz_arg = DeclareLaunchArgument(
        'camera_to_arm_xyz', default_value='[0.0, 0.0, 0.0]',
        description='Base offset: camera -> arm_base translation [x, y, z] in meters',
    )
    camera_to_arm_rpy_arg = DeclareLaunchArgument(
        'camera_to_arm_rpy', default_value='[0.0, 0.0, 0.0]',
        description='Base offset: camera -> arm_base rotation [roll, pitch, yaw] in radians',
    )

    return LaunchDescription([
        model_path_arg,
        input_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        use_arm_bridge_arg,
        source_w_arg,
        source_h_arg,
        target_frame_arg,
        camera_to_arm_xyz_arg,
        camera_to_arm_rpy_arg,
        Node(
            package='detection_3d',
            executable='yolo_detector',
            name='yolo_detector',
            parameters=[params_file, {
                'model_path': LaunchConfiguration('model_path'),
                'input_topic': LaunchConfiguration('input_topic'),
            }],
            output='screen',
        ),
        Node(
            package='detection_3d',
            executable='detection_3d_calculator',
            name='detection_3d_calculator',
            parameters=[params_file, {
                'depth_topic': LaunchConfiguration('depth_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'source_image_width': LaunchConfiguration('source_image_width'),
                'source_image_height': LaunchConfiguration('source_image_height'),
            }],
            output='screen',
        ),
        Node(
            package='detection_3d',
            executable='arm_serial_bridge',
            name='arm_serial_bridge',
            parameters=[params_file, {
                'target_frame': LaunchConfiguration('target_frame'),
                'camera_to_arm_xyz': LaunchConfiguration('camera_to_arm_xyz'),
                'camera_to_arm_rpy': LaunchConfiguration('camera_to_arm_rpy'),
            }],
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_arm_bridge')),
        ),
    ])
