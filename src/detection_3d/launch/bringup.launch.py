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
    depth_pixel_offset_x_arg = DeclareLaunchArgument(
        'depth_pixel_offset_x_px', default_value='0.0',
        description='Extra x offset from RGB detector pixel to depth pixel',
    )
    depth_pixel_offset_y_arg = DeclareLaunchArgument(
        'depth_pixel_offset_y_px', default_value='0.0',
        description='Extra y offset from RGB detector pixel to depth pixel',
    )
    debug_projection_log_arg = DeclareLaunchArgument(
        'debug_projection_log', default_value='false',
        description='Print per-detection projection details',
    )
    serial_tx_log_arg = DeclareLaunchArgument(
        'serial_tx_log', default_value='true',
        description='Print serial target/pump commands sent to MCU',
    )
    serial_tx_log_hex_arg = DeclareLaunchArgument(
        'serial_tx_log_hex', default_value='false',
        description='Print complete serial TX frame bytes in hex',
    )
    serial_rx_log_arg = DeclareLaunchArgument(
        'serial_rx_log', default_value='true',
        description='Print parsed MCU feedback frames received from CDC',
    )
    serial_rx_log_hex_arg = DeclareLaunchArgument(
        'serial_rx_log_hex', default_value='true',
        description='Print complete serial RX frame bytes in hex',
    )
    camera_tilt_forward_arg = DeclareLaunchArgument(
        'camera_tilt_forward_deg', default_value='45.0',
        description='Camera optical axis tilt from vertical down toward arm +x_e',
    )
    command_offset_x_arg = DeclareLaunchArgument(
        'command_offset_x_m', default_value='0.0',
        description='Final x command offset applied before serial TX',
    )
    command_offset_y_arg = DeclareLaunchArgument(
        'command_offset_y_m', default_value='0.0',
        description='Final y command offset applied before serial TX',
    )
    command_offset_z_arg = DeclareLaunchArgument(
        'command_offset_z_m', default_value='0.15',
        description='Final z command offset applied before serial TX',
    )
    command_abs_y_offset_arg = DeclareLaunchArgument(
        'command_abs_y_offset_m', default_value='-0.01',
        description='Adjust absolute y command magnitude before serial TX; negative shrinks toward zero',
    )
    place_target_index_arg = DeclareLaunchArgument(
        'place_target_index', default_value='0',
        description='Index of fixed place target: 0=right-rear, 1=left-rear, 2=left-front, 3=right-front',
    )
    grasp_occlusion_hold_arg = DeclareLaunchArgument(
        'grasp_occlusion_hold_enabled', default_value='true',
        description='Keep sending filtered grasp command when vision is occluded',
    )
    grasp_command_filter_window_arg = DeclareLaunchArgument(
        'grasp_command_filter_window', default_value='5',
        description='Recent successful grasp commands used for occlusion median',
    )
    grasp_occlusion_timeout_arg = DeclareLaunchArgument(
        'grasp_occlusion_timeout_sec', default_value='8.0',
        description='Maximum seconds to hold grasp target after vision timeout',
    )
    arrival_stall_enabled_arg = DeclareLaunchArgument(
        'arrival_stall_enabled', default_value='true',
        description='Treat near-target stopped end-effector as arrival',
    )
    arrival_stall_epsilon_arg = DeclareLaunchArgument(
        'arrival_stall_epsilon_m', default_value='0.015',
        description='Maximum per-frame end-effector motion for stall arrival',
    )
    arrival_stall_frames_arg = DeclareLaunchArgument(
        'arrival_stall_frames', default_value='5',
        description='Consecutive stall frames required for arrival',
    )
    arrival_stall_max_distance_arg = DeclareLaunchArgument(
        'arrival_stall_max_distance_m', default_value='0.08',
        description='Maximum target distance accepted for stall arrival',
    )
    feedback_loss_abort_arg = DeclareLaunchArgument(
        'feedback_loss_abort_sec', default_value='3.0',
        description='Reset state and request pump off after this long without MCU feedback',
    )
    mcu_reached_enabled_arg = DeclareLaunchArgument(
        'mcu_reached_enabled', default_value='true',
        description='Allow constrained MCU reached state as one arrival condition',
    )
    mcu_reached_stable_frames_arg = DeclareLaunchArgument(
        'mcu_reached_stable_frames', default_value='2',
        description='Consecutive MCU reached feedback frames required for arrival',
    )
    mcu_reached_max_distance_arg = DeclareLaunchArgument(
        'mcu_reached_max_distance_m', default_value='0.10',
        description='Maximum end-target distance accepted for MCU reached arrival',
    )
    mcu_reached_min_motion_arg = DeclareLaunchArgument(
        'mcu_reached_min_motion_m', default_value='0.01',
        description='Minimum end-effector motion after target send before MCU reached is trusted',
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
            'depth_pixel_offset_x_px': LaunchConfiguration('depth_pixel_offset_x_px'),
            'depth_pixel_offset_y_px': LaunchConfiguration('depth_pixel_offset_y_px'),
            'debug_projection_log': LaunchConfiguration('debug_projection_log'),
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
            'serial_tx_log': LaunchConfiguration('serial_tx_log'),
            'serial_tx_log_hex': LaunchConfiguration('serial_tx_log_hex'),
            'serial_rx_log': LaunchConfiguration('serial_rx_log'),
            'serial_rx_log_hex': LaunchConfiguration('serial_rx_log_hex'),
            'camera_tilt_forward_deg': LaunchConfiguration('camera_tilt_forward_deg'),
            'command_offset_x_m': LaunchConfiguration('command_offset_x_m'),
            'command_offset_y_m': LaunchConfiguration('command_offset_y_m'),
            'command_offset_z_m': LaunchConfiguration('command_offset_z_m'),
            'command_abs_y_offset_m': LaunchConfiguration('command_abs_y_offset_m'),
            'place_target_index': LaunchConfiguration('place_target_index'),
            'grasp_occlusion_hold_enabled': LaunchConfiguration(
                'grasp_occlusion_hold_enabled'
            ),
            'grasp_command_filter_window': LaunchConfiguration(
                'grasp_command_filter_window'
            ),
            'grasp_occlusion_timeout_sec': LaunchConfiguration(
                'grasp_occlusion_timeout_sec'
            ),
            'arrival_stall_enabled': LaunchConfiguration('arrival_stall_enabled'),
            'arrival_stall_epsilon_m': LaunchConfiguration(
                'arrival_stall_epsilon_m'
            ),
            'arrival_stall_frames': LaunchConfiguration('arrival_stall_frames'),
            'arrival_stall_max_distance_m': LaunchConfiguration(
                'arrival_stall_max_distance_m'
            ),
            'feedback_loss_abort_sec': LaunchConfiguration('feedback_loss_abort_sec'),
            'mcu_reached_enabled': LaunchConfiguration('mcu_reached_enabled'),
            'mcu_reached_stable_frames': LaunchConfiguration(
                'mcu_reached_stable_frames'
            ),
            'mcu_reached_max_distance_m': LaunchConfiguration(
                'mcu_reached_max_distance_m'
            ),
            'mcu_reached_min_motion_m': LaunchConfiguration(
                'mcu_reached_min_motion_m'
            ),
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
            'depth_pixel_offset_x_px': LaunchConfiguration('depth_pixel_offset_x_px'),
            'depth_pixel_offset_y_px': LaunchConfiguration('depth_pixel_offset_y_px'),
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
        depth_pixel_offset_x_arg,
        depth_pixel_offset_y_arg,
        debug_projection_log_arg,
        serial_tx_log_arg,
        serial_tx_log_hex_arg,
        serial_rx_log_arg,
        serial_rx_log_hex_arg,
        camera_tilt_forward_arg,
        command_offset_x_arg,
        command_offset_y_arg,
        command_offset_z_arg,
        command_abs_y_offset_arg,
        place_target_index_arg,
        grasp_occlusion_hold_arg,
        grasp_command_filter_window_arg,
        grasp_occlusion_timeout_arg,
        arrival_stall_enabled_arg,
        arrival_stall_epsilon_arg,
        arrival_stall_frames_arg,
        arrival_stall_max_distance_arg,
        feedback_loss_abort_arg,
        mcu_reached_enabled_arg,
        mcu_reached_stable_frames_arg,
        mcu_reached_max_distance_arg,
        mcu_reached_min_motion_arg,
        yolo_node,
        calc_node,
        arm_bridge_node,
        sample_logger_node,
        estimator_comparison_logger_node,
        rviz_node,
    ])
