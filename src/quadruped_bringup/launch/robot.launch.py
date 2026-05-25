"""
robot.launch.py — Bringup the full quadruped robot stack.

Launches:
  1. strategy_manager_node (high-level coordination)
  2. (placeholder) LiDAR driver + Nav2
  3. (placeholder) vision_node

Note: serial_bridge_node has been superseded by detection_3d/arm_serial_bridge.
      All serial I/O now goes through the Python-side unified serial bridge
      using the 0x55 0xAA protocol. See detection_3d/launch/bringup.launch.py.

Usage:
  ros2 launch quadruped_bringup robot.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ---- Arguments ----
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
        use_lidar_arg,
        use_nav2_arg,
        use_vision_arg,
        strategy_manager_node,
        vision_node,
        # lidar_launch,
        # nav2_launch,
    ])
