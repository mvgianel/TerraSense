#!/usr/bin/env python3
"""
- RealSense + TurtleBot start immediately.
- On RealSense process start: wait for IMU topics, then launch imu_filter.
- On imu_filter process start: wait for image topics, then launch terra_sense (if enabled).
- Timeouts: launch anyway with a warning.
"""

import time
from typing import List

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
)
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart, OnShutdown
from launch_ros.actions import Node


def _wait_for_topics_blocking(topics: List[str], timeout: float, label: str, poll_period: float = 0.5):
    """
    Blocking wait: returns after all topics appear or timeout.
    """
    import rclpy
    from rclpy.node import Node as RclpyNode

    rclpy.init(args=None)
    node = RclpyNode(f'{label}_wait_node')
    start = time.time()

    def have_all():
        existing = {t for (t, _types) in node.get_topic_names_and_types()}
        return all(t in existing for t in topics), existing

    while True:
        ok, existing = have_all()
        if ok:
            print(f'[INFO] [{label}] Found all topics: {topics}')
            break
        if timeout > 0 and (time.time() - start) > timeout:
            missing = [t for t in topics if t not in existing]
            print(f'[WARN] [{label}] Timeout ({timeout:.1f}s). Missing topics: {missing}')
            break
        rclpy.spin_once(node, timeout_sec=poll_period)

    node.destroy_node()
    rclpy.shutdown()


def generate_launch_description():
    # ---------------- Launch Arguments ----------------
    launch_terra = DeclareLaunchArgument(
        'launch_terra_sense', default_value='true',
        description='Whether to launch terra_sense terrain_publisher node.'
    )
    imu_topics_arg = DeclareLaunchArgument(
        'imu_ready_topics', default_value='/camera/camera/imu',
        description='Comma-separated required topics before launching imu_filter.'
    )
    terra_topics_arg = DeclareLaunchArgument(
        'terra_ready_topics',
        default_value='/camera/camera/imu,/camera/camera/color/image_raw,/camera/camera/depth/image_rect_raw',
        description='Comma-separated required topics before launching terra_sense.'
    )
    imu_timeout_arg = DeclareLaunchArgument(
        'imu_wait_timeout', default_value='30.0',
        description='Seconds to wait for IMU topics (<=0 wait forever).'
    )
    terra_timeout_arg = DeclareLaunchArgument(
        'terra_wait_timeout', default_value='40.0',
        description='Seconds to wait for terra topics (<=0 wait forever).'
    )

    # ---------------- Core Processes ----------------
    realsense_proc = ExecuteProcess(
        cmd=[
            'ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
            'device_type:=d455',
            'initial_reset:=true',
            'enable_color:=true',
            'enable_depth:=true',
            'pointcloud.enable:=false',
            'enable_gyro:=true',
            'enable_accel:=true',
            'unite_imu_method:=2',
            'rgb_camera.color_profile:=640x480x15',
            'depth_module.depth_profile:=640x480x15'
        ],
        output='screen'
    )

    turtlebot_proc = ExecuteProcess(
        cmd=['ros2', 'launch', 'turtlebot3_bringup', 'robot.launch.py'],
        output='screen'
    )

    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter_madgwick',
        output='screen',
        remappings=[
            ('/imu/data_raw', '/camera/camera/imu'),
            ('/imu/data', '/rtabmap/imu'),
        ],
        parameters=[{
            'use_mag': False,
            'fixed_frame': 'camera_link',
            '_publish_tf': False,
            '_world_frame': 'enu',
        }]
    )

    terra_node = Node(
        package='terra_sense',
        executable='terrain_publisher.py',
        name='terrain_publisher',
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_terra_sense'))
    )

    # ------------- Event Handlers (Chaining) -------------
    def on_realsense_started(event, context):
        topics = [t.strip() for t in LaunchConfiguration('imu_ready_topics').perform(context).split(',') if t.strip()]
        timeout = float(LaunchConfiguration('imu_wait_timeout').perform(context))
        print(f'[INFO] [imu_filter] Waiting for topics after RealSense start: {topics} (timeout={timeout}s)')
        _wait_for_topics_blocking(topics, timeout, 'imu_filter')
        # Return the imu_filter node so launch system spawns it now.
        return [imu_filter_node]

    imu_start_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=realsense_proc,
            on_start=on_realsense_started
        )
    )

    def on_imu_filter_started(event, context):
        # Only proceed if terra is enabled
        if LaunchConfiguration('launch_terra_sense').perform(context).lower() != 'true':
            print('[INFO] [terra_sense] Disabled; skipping wait.')
            return []
        topics = [t.strip() for t in LaunchConfiguration('terra_ready_topics').perform(context).split(',') if t.strip()]
        timeout = float(LaunchConfiguration('terra_wait_timeout').perform(context))
        print(f'[INFO] [terra_sense] Waiting for topics after IMU filter start: {topics} (timeout={timeout}s)')
        _wait_for_topics_blocking(topics, timeout, 'terra_sense')
        return [terra_node]

    terra_chain_handler = RegisterEventHandler(
        OnProcessStart(
            target_action=imu_filter_node,
            on_start=on_imu_filter_started
        )
    )

    # ------------- Assemble -------------
    ld = LaunchDescription([
        launch_terra,
        imu_topics_arg,
        terra_topics_arg,
        imu_timeout_arg,
        terra_timeout_arg,

        realsense_proc,
        turtlebot_proc,

        imu_start_handler,
        terra_chain_handler,
    ])

    return ld
