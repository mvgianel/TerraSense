#!/usr/bin/env python3
"""
rtab_nav2.launch.py

Launch sequence for ROS 2 Humble:
 1. Launch RTAB-Map with specified parameters.
 2. Wait for a user-defined RTAB-Map topic to appear.
 3. Launch Nav2 bringup and EKF under a lifecycle manager only after that topic is available.

Usage examples:
  ros2 launch your_pkg terrasense_laptop.launch.py \
    rtabmap_ready_topic:=/rtabmap/map \
    rtabmap_wait_timeout:=30.0 \
    use_sim_time:=true
"""
import time
from typing import List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.conditions import IfCondition


def _wait_for_topics(topics: List[str], timeout: float, label: str, poll_period: float = 0.5):
    """Block until all topics exist or timeout."""
    import rclpy
    from rclpy.node import Node as RclpyNode

    rclpy.init(args=None)
    node = RclpyNode(f'{label}_wait_node')
    start = time.time()

    while True:
        existing = {t for (t, _) in node.get_topic_names_and_types()}
        missing = [t for t in topics if t not in existing]
        if not missing:
            print(f"[INFO] [{label}] Required topics present: {topics}")
            break
        if timeout > 0 and (time.time() - start) > timeout:
            print(f"[WARN] [{label}] Timeout {timeout:.1f}s waiting for topics. Missing: {missing}")
            break
        rclpy.spin_once(node, timeout_sec=poll_period)

    node.destroy_node()
    rclpy.shutdown()


def generate_launch_description():
    # --------------------
    # Launch Arguments
    # --------------------
    text_image_arg = DeclareLaunchArgument(
        'demo', default_value='false',
        description='Put class string on image for demo?'
    )

    rtab_ready_arg = DeclareLaunchArgument(
        'rtabmap_ready_topic', default_value='/rtabmap/map',
        description='Topic to wait for before launching Nav2 and EKF.'
    )

    rtab_timeout_arg = DeclareLaunchArgument(
        'rtabmap_wait_timeout', default_value='30.0',
        description='Seconds to wait for RTAB-Map readiness topic (<=0 means wait forever).'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (ROS) clock.'
    )

    # --------------------
    # RTAB-Map Launch
    # --------------------
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('rtabmap_launch'), 'launch', 'rtabmap.launch.py'
            ])
        ),
        launch_arguments={
            'rtabmap_args': '--delete_bd_on_start',
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/depth/image_rect_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'imu_topic': '/rtabmap/imu',
            # 'imu_topic': '/odometry/filtered',
            'frame_id': 'base_footprint',
            'odom_frame_id': 'odom',
            #'use_action_for_goal': 'true',
            'subscribe_depth': 'true',
            'approx_sync': 'true',
            # 'approx_sync_max_interval': '0.2',  # Max time diff between messages (seconds)
            # 'wait_for_transform': '2.0',        # Increase transform wait time
            'qos': '2',
            'rtabmap_viz': 'false',
            'rviz': 'false',
            #'publish_tf': 'false',
            'publish_tf': 'true',
            'Reg/Force3DoF': 'true',
            'Mem/STMSize': '30',
            # ''
            'decimation': '2',
            'approx_sync_max_interval': '0.05',
            # 'topic_queue_size': '30',    # Increased from 10
            # 'sync_queue_size': '30',      # Increased from 10
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items(),
        # remappings=[
        # ('/odom', '/odometry/filtered')
        # ]
    )

    # --------------------
    # Nav2 Bringup
    # --------------------
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'), 'launch', 'bringup_launch.py'
            ])
        ),
        launch_arguments={
            'params_file': PathJoinSubstitution([
                '/home/kivi/amd_ws/src/terra_sense', 'config', 'nav2_params_noclass.yaml'
            ]),
            'slam':'False',
            'map': ' ',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'autostart': 'true'
        }.items()
    )

    # --------------------
    # EKF Filter Node
    # --------------------
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                '/home/kivi/amd_ws/src/terra_sense', 'config', 'ekf.yaml'
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time'),
             'publish_tf': True,}
        ]
    )

    # Lifecycle Manager for localization (EKF + RTAB-Map)
    lifecycle_manager_localization = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'autostart': True,
            'node_names': ['ekf_filter_node', 'rtabmap']
        }]
    )


    # --------------------
    # Deferred Launch of Nav2, EKF, and Lifecycle Manager until RTAB-Map ready
    # --------------------
    def _deferred_launch(context, *args, **kwargs):
        topic = LaunchConfiguration('rtabmap_ready_topic').perform(context)
        timeout = float(LaunchConfiguration('rtabmap_wait_timeout').perform(context))
        _wait_for_topics([topic], timeout, 'rtabmap_ready')
        return [lifecycle_manager_localization, ekf_node, nav2_launch]

    deferred = OpaqueFunction(function=_deferred_launch)

    # --------------------
    # Optional Text Overlay Node
    # --------------------
    text_image_node = Node(
        package='terra_sense',
        executable='image_text_overlay.py',
        name='terrain_img_overlay',
        output='screen',
        condition=IfCondition(LaunchConfiguration('demo'))
    )

    # --------------------
    # Assemble Launch Description
    # --------------------
    ld = LaunchDescription([
        rtab_ready_arg,
        rtab_timeout_arg,
        use_sim_time_arg,
        text_image_arg,
        rtabmap_launch,
        ekf_node,
        # deferred,
        text_image_node,
    ])

    return ld
