"""Full system bringup: detection + optional RViz + optional arm bridge."""

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

    # ---- Launch arguments ----
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(pkg_dir, 'models', 'best_int8.xml'),
        description='Path to OpenVINO INT8 model (.xml)',
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
    use_sample_logger_arg = DeclareLaunchArgument(
        'use_sample_logger', default_value='false',
        description='Record stable 3D detection samples to CSV',
    )
    sample_output_path_arg = DeclareLaunchArgument(
        'sample_output_path', default_value='detection_samples.csv',
        description='CSV output path for coordinate samples',
    )
    use_estimator_comparison_logger_arg = DeclareLaunchArgument(
        'use_estimator_comparison_logger', default_value='false',
        description='Record same-frame center_median/cluster_centroid estimates to CSV',
    )
    estimator_comparison_output_path_arg = DeclareLaunchArgument(
        'estimator_comparison_output_path',
        default_value='estimator_comparison_samples.csv',
        description='CSV output path for same-frame estimator comparison samples',
    )
    depth_estimator_mode_arg = DeclareLaunchArgument(
        'depth_estimator_mode', default_value='cluster_centroid',
        description='Depth estimator mode: cluster_centroid or center_median',
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
            'depth_estimator_mode': LaunchConfiguration('depth_estimator_mode'),
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
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_arm_bridge')),
    )

    # ---- Optional coordinate sample logger ----
    sample_logger_node = Node(
        package='detection_3d',
        executable='coordinate_sample_logger',
        name='coordinate_sample_logger',
        parameters=[params_file, {
            'output_path': LaunchConfiguration('sample_output_path'),
            'target_class': LaunchConfiguration('target_class'),
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_sample_logger')),
    )

    # ---- Optional same-frame estimator comparison logger ----
    estimator_comparison_logger_node = Node(
        package='detection_3d',
        executable='estimator_comparison_logger',
        name='estimator_comparison_logger',
        parameters=[params_file, {
            'output_path': LaunchConfiguration('estimator_comparison_output_path'),
            'target_class': LaunchConfiguration('target_class'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'source_image_width': LaunchConfiguration('source_image_width'),
            'source_image_height': LaunchConfiguration('source_image_height'),
        }],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_estimator_comparison_logger')),
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
        use_sample_logger_arg,
        sample_output_path_arg,
        use_estimator_comparison_logger_arg,
        estimator_comparison_output_path_arg,
        depth_estimator_mode_arg,
        input_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        target_class_arg,
        source_w_arg,
        source_h_arg,
        yolo_node,
        calc_node,
        arm_bridge_node,
        sample_logger_node,
        estimator_comparison_logger_node,
        rviz_node,
    ])
