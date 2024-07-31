from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    turtlebot_dir = get_package_share_directory('turtlebot3_bringup')
    terrasense_layer_dir = get_package_share_directory('terra_sense')
    realsense_dir = 

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
        'unite_imu_method':'2',
        'rgb_camera.color_profile': '640x480x15',
        'depth_module.depth_profile': '640x480x15',
    }.items(),
    )
    
    
    classification = Node(
        package='terra_sense',
        executable='terrain_publisher.py',
    )

    turtlebot_real = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot_dir, 'launch', 'robot.launch.py')
        ),
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

    ld = LaunchDescription()


    # launch ekf to combine imu and wheel odometry
    ld.add_action(realsense)
    ld.add_action(classification)
    ld.add_action(turtlebot_real)
    ld.add_action(imu)


    return ld