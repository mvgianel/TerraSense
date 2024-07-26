from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    bringup_dir = get_package_share_directory('nav2_bringup')
    terrasense_layer_dir = get_package_share_directory('terra_sense')
    rviz_config_dir = os.path.join(bringup_dir, 'rviz', 'nav2_default_view.rviz')
    
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_dir, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'params_file': os.path.join(terrasense_layer_dir, 'config', 'nav2_params.yaml')
            }.items(),
        ),
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
        ),
        Node(
            package='terra_sense',
            executable='terrain_classifier',
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_dir],
        ),
        
        # change this
        
        Node(
            package='turtlebot3_bringup',
            executable='turtlesim_node',
        )
    ])