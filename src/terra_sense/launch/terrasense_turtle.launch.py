from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    run_in_robot = LaunchConfiguration('run_in_robot')

    bringup_dir = get_package_share_directory('nav2_bringup')
    turtlebot_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    turtlebot_dir = get_package_share_directory('turtlebot3_bringup')
    terrasense_layer_dir = get_package_share_directory('terra_sense')
    rviz_config_dir = os.path.join(terrasense_layer_dir, 'config', 'rviz_nav2.rviz')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use simulation (Gazebo) clock if true')

    declare_run_in_robot_cmd = DeclareLaunchArgument(
        'run_in_robot',
        default_value='False',
        description='Running launch in robot?')

    realsense = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
    ),
    launch_arguments={
        'device_type': 'd455',
        'enable_color': 'true',
        'enable_depth': 'true',
        'pointcloud.enable':'false',
        'enable_gyro': 'true',
        'enable_accel':'true',
        'unite_imu_method':2,
        'rgb_camera.color_profile': '640x480x15',
        'depth_module.depth_profile': '640x480x15',
    }.items(),
    )
    
    classification = Node(
        package='terra_sense',
        executable='terrain_classifier',
    )
        
    turtlebot_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot_gazebo_dir, 'launch', 'turtlebot3_house.launch.py')
        ),
        launch_arguments={
            'use_sim_time' : use_sim_time,
            'x_pose' : '5',
            'y_pose' : '1',
        }.items(),
    )

    turtlebot_real = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot_gazebo_dir, 'launch', 'turtlebot3_house.launch.py')
        ),
    )

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            'rtabmap_args': '--delete_bd_on_start',
            'rgb_topic':'/camera/camera/color/image_raw depth',
            'topic': 'camera/camera/depth/image_rect_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'imu_topic': '/rtabmap/imu',
            'frame_id': 'base_footprint',
            'subscribe_depth': 'true',
            'approx_sync': 'true',
            'qos': '2',
            'sync_queue_size': '30',
            'rtabmap_viz': 'false',
            'rviz': 'false',
        }.items(),
    )

    imu = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        remappings=[
            ('/imu/data_raw', '/camera/camera/imu'),
            ('/imu/data', '/rtabmap/imu'),
            ('_publish_tf', 'false'),
            ('_world_frame', 'enu'),
        ],
        parameters=[
            ('use_mag', 'false'),
            ('fixed_frame', 'camera_link'),
        ]
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'params_file': os.path.join(terrasense_layer_dir, 'config', 'nav2_params.yaml'),
            'slam': 'False',
            'map': '',
            'use_sim_time' : use_sim_time,
            'autostart': 'True',
            'use_composition': 'False',
            'use_respawn': 'False',
        }.items(),
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_dir],
    )

    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_run_in_robot_cmd)

    # launch ekf to combine imu and wheel odometry
    # ld.add_action(realsense)
    # ld.add_action(classification)
    # ld.add_action(turtlebot_real)
    # ld.add_action(turtlebot_sim)
    # ld.add_action(imu)
    # ld.add_action(TimerAction(period=10.0, actions=[LogInfo(msg='Starting RTAB-Map')]))
    # ld.add_action(slam)
    # ld.add_action(TimerAction(period=15.0, actions=[LogInfo(msg='Starting Nav2')]))
    ld.add_action(nav2)
    # ld.add_action(TimerAction(period=20.0, actions=[LogInfo(msg='Starting Rviz')]))
    # ld.add_action(rviz)

    return ld