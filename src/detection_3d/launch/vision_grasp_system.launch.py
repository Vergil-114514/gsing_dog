"""One-command Linux bringup for Astra camera, YOLO 3D detection, and STM32 bridge."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
import os


def generate_launch_description():
    """Launch the full vision-grasp stack with deployment-friendly defaults."""
    detection_pkg = get_package_share_directory('detection_3d')
    astra_pkg = get_package_share_directory('astra_camera')

    serial_port_arg = DeclareLaunchArgument(
        'serial_port',
        default_value=(
            '/dev/serial/by-id/'
            'usb-STMicroelectronics_STM32_Virtual_ComPort_3542354B3333-if00'
        ),
        description='STM32 USB CDC serial path',
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description='Launch RViz2 from detection bringup',
    )
    target_class_arg = DeclareLaunchArgument(
        'target_class', default_value='',
        description='Filter detections by class name (empty = any)',
    )
    depth_registration_arg = DeclareLaunchArgument(
        'depth_registration', default_value='true',
        description='Enable Astra hardware depth-to-color registration',
    )
    color_depth_sync_arg = DeclareLaunchArgument(
        'color_depth_synchronization', default_value='true',
        description='Enable Astra depth/color hardware synchronization',
    )
    depth_pixel_offset_x_arg = DeclareLaunchArgument(
        'depth_pixel_offset_x_px', default_value='0.0',
        description='Field-calibrated RGB-to-depth x pixel offset',
    )
    depth_pixel_offset_y_arg = DeclareLaunchArgument(
        'depth_pixel_offset_y_px', default_value='0.0',
        description='Field-calibrated RGB-to-depth y pixel offset',
    )
    debug_projection_log_arg = DeclareLaunchArgument(
        'debug_projection_log', default_value='false',
        description='Print projection details from 2D pixels to 3D coordinates',
    )
    serial_tx_log_arg = DeclareLaunchArgument(
        'serial_tx_log', default_value='true',
        description='Print serial target/pump commands sent to MCU',
    )
    serial_tx_log_hex_arg = DeclareLaunchArgument(
        'serial_tx_log_hex', default_value='false',
        description='Print complete serial TX frames in hex',
    )
    serial_rx_log_arg = DeclareLaunchArgument(
        'serial_rx_log', default_value='true',
        description='Print parsed MCU feedback frames received from CDC',
    )
    serial_rx_log_hex_arg = DeclareLaunchArgument(
        'serial_rx_log_hex', default_value='true',
        description='Print complete serial RX frames in hex',
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
        'command_offset_z_m', default_value='0.11',
        description='Final z command offset applied before serial TX',
    )
    command_abs_y_offset_arg = DeclareLaunchArgument(
        'command_abs_y_offset_m', default_value='0.03',
        description='Adjust absolute y command magnitude before serial TX; negative shrinks toward zero',
    )
    place_target_index_arg = DeclareLaunchArgument(
        'place_target_index',
        default_value='0',
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

    camera_launch = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(astra_pkg, 'launch', 'astra_pro.launch.xml')
        ),
        launch_arguments={
            'uvc_vendor_id': '0x2bc5',
            'uvc_product_id': '0x0511',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_ir': 'false',
            'color_width': '640',
            'color_height': '480',
            'color_fps': '30',
            'depth_width': '640',
            'depth_height': '400',
            'depth_fps': '30',
            'depth_registration': LaunchConfiguration('depth_registration'),
            'color_depth_synchronization': LaunchConfiguration('color_depth_synchronization'),
        }.items(),
    )

    detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(detection_pkg, 'launch', 'bringup.launch.py')
        ),
        launch_arguments={
            'use_arm_bridge': 'true',
            'serial_port': LaunchConfiguration('serial_port'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'target_class': LaunchConfiguration('target_class'),
            'input_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/depth/image_raw',
            'camera_info_topic': '/camera/depth/camera_info',
            'source_image_width': '640',
            'source_image_height': '480',
            'depth_pixel_offset_x_px': LaunchConfiguration('depth_pixel_offset_x_px'),
            'depth_pixel_offset_y_px': LaunchConfiguration('depth_pixel_offset_y_px'),
            'debug_projection_log': LaunchConfiguration('debug_projection_log'),
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
        }.items(),
    )

    return LaunchDescription([
        serial_port_arg,
        use_rviz_arg,
        target_class_arg,
        depth_registration_arg,
        color_depth_sync_arg,
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
        camera_launch,
        detection_launch,
    ])
