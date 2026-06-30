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
    command_offset_x_arg = DeclareLaunchArgument(
        'command_offset_x_m', default_value='0.0',
        description='Final x command offset applied before serial TX',
    )
    command_offset_y_arg = DeclareLaunchArgument(
        'command_offset_y_m', default_value='0.0',
        description='Final y command offset applied before serial TX',
    )
    command_offset_z_arg = DeclareLaunchArgument(
        'command_offset_z_m', default_value='0.09',
        description='Final z command offset applied before serial TX',
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
            'command_offset_x_m': LaunchConfiguration('command_offset_x_m'),
            'command_offset_y_m': LaunchConfiguration('command_offset_y_m'),
            'command_offset_z_m': LaunchConfiguration('command_offset_z_m'),
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
        command_offset_x_arg,
        command_offset_y_arg,
        command_offset_z_arg,
        camera_launch,
        detection_launch,
    ])
