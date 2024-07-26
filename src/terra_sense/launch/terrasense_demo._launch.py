from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    terrasense_path = get_package_share_directory('terra_sense')
    
    return LaunchDescription([
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
        ),
        Node(
            package='terra_sense',
            executable='terrain_classifier',
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                LaunchConfiguration('launch_dir', default='install/nav2_bringup/launch/nav2_bringup_launch.py')
            ]),
            launch_arguments={'use_sim_time': 'false', 'params_file': os.path.join(terrasense_path, 'config', 'nav2_params.yaml')}.items(),
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        ),

        # change this
        
        Node(
            package='turtlebot3_bringup',
            executable='turtlesim_node',
        )
    ])