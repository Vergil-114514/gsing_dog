"""
robot.launch.py — Bringup the full quadruped robot stack.

Launches:
  1. serial_bridge_node   (MCU ↔ ROS2 bridge)
  2. strategy_manager_node (high-level coordination)
  3. (placeholder) LiDAR driver + Nav2
  4. (placeholder) vision_node

Usage:
  ros2 launch quadruped_bringup robot.launch.py port:=/dev/ttyACM0 baudrate:=115200
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ---- Arguments ----
    port_arg = DeclareLaunchArgument(
        'port', default_value='/dev/ttyACM0',
        description='Serial device path for MCU communication'
    )
    baudrate_arg = DeclareLaunchArgument(
        'baudrate', default_value='115200',
        description='Serial baudrate'
    )
    use_lidar_arg = DeclareLaunchArgument(
        'use_lidar', default_value='false',
        description='Set to true to include LiDAR driver launch'
    )
    use_nav2_arg = DeclareLaunchArgument(
        'use_nav2', default_value='false',
        description='Set to true to include Nav2 launch'
    )
    use_vision_arg = DeclareLaunchArgument(
        'use_vision', default_value='false',
        description='Set to true to include vision_node'
    )

    # ---- Nodes ----
    serial_bridge_node = Node(
        package='quadruped_control',
        executable='serial_bridge_node',
        name='serial_bridge_node',
        output='screen',
        parameters=[PathJoinSubstitution([
            FindPackageShare('quadruped_bringup'), 'config', 'params.yaml'
        ])],
        arguments=['--ros-args', '--log-level', 'INFO'],
    )

    strategy_manager_node = Node(
        package='quadruped_control',
        executable='strategy_manager_node',
        name='strategy_manager_node',
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO'],
    )

    vision_node = Node(
        package='quadruped_vision',
        executable='vision_node',
        name='vision_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_vision')),
    )

    # ---- LiDAR driver (placeholder) ----
    # lidar_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         PathJoinSubstitution([FindPackageShare('sllidar_ros2'), 'launch', 'sllidar_launch.py'])
    #     ]),
    #     condition=IfCondition(LaunchConfiguration('use_lidar')),
    # )

    # ---- Nav2 (placeholder) ----
    # nav2_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         PathJoinSubstitution([FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'])
    #     ]),
    #     condition=IfCondition(LaunchConfiguration('use_nav2')),
    #     launch_arguments={
    #         'params_file': PathJoinSubstitution([
    #             FindPackageShare('quadruped_bringup'), 'config', 'nav2_params.yaml'
    #         ]),
    #     }.items(),
    # )

    return LaunchDescription([
        port_arg,
        baudrate_arg,
        use_lidar_arg,
        use_nav2_arg,
        use_vision_arg,
        serial_bridge_node,
        strategy_manager_node,
        vision_node,
        # lidar_launch,
        # nav2_launch,
    ])
