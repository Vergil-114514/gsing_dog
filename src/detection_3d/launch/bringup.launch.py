"""Full system bringup: camera + detection + optional RViz + optional arm bridge."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('detection_3d')
    params_file = os.path.join(pkg_dir, 'config', 'detection_params.yaml')

    # ---- Launch arguments ----
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(pkg_dir, 'models', 'best.pt'),
        description='Path to YOLO model (.pt or .onnx)',
    )
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='auto',
        description='STM32 USB CDC port (auto, /dev/ttyACM0, etc.)',
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description='Launch RViz2 with detection config',
    )
    use_arm_bridge_arg = DeclareLaunchArgument(
        'use_arm_bridge', default_value='false',
        description='Launch arm serial bridge to STM32',
    )
    input_topic_arg = DeclareLaunchArgument(
        'input_topic', default_value='/camera/color/image_raw',
        description='Color image topic for YOLO',
    )
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic', default_value='/camera/depth/image_raw',
        description='Depth image topic for 3D calculator',
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic', default_value='/camera/depth/camera_info',
        description='Camera info topic for intrinsics',
    )
    target_class_arg = DeclareLaunchArgument(
        'target_class', default_value='',
        description='Filter detections by class name (empty = any)',
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

    # ---- Camera launch ----
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'camera.launch.py')
        ),
    )

    # ---- Detection nodes ----
    yolo_node = Node(
        package='detection_3d',
        executable='yolo_detector',
        name='yolo_detector',
        parameters=[params_file, {
            'model_path': LaunchConfiguration('model_path'),
            'input_topic': LaunchConfiguration('input_topic'),
        }],
        output='screen',
    )
    calc_node = Node(
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
    )

    # ---- Optional arm serial bridge ----
    arm_bridge_node = Node(
        package='detection_3d',
        executable='arm_serial_bridge',
        name='arm_serial_bridge',
        parameters=[params_file, {
            'serial_port': LaunchConfiguration('serial_port'),
            'target_class': LaunchConfiguration('target_class'),
            'target_frame': LaunchConfiguration('target_frame'),
            'camera_to_arm_xyz': LaunchConfiguration('camera_to_arm_xyz'),
            'camera_to_arm_rpy': LaunchConfiguration('camera_to_arm_rpy'),
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_arm_bridge')),
    )

    # ---- Optional RViz ----
    rviz_config = os.path.join(pkg_dir, 'rviz', 'detection.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription([
        model_path_arg,
        serial_port_arg,
        use_rviz_arg,
        use_arm_bridge_arg,
        input_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        target_class_arg,
        source_w_arg,
        source_h_arg,
        target_frame_arg,
        camera_to_arm_xyz_arg,
        camera_to_arm_rpy_arg,
        camera_launch,
        yolo_node,
        calc_node,
        arm_bridge_node,
        rviz_node,
    ])
