from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
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
            launch_arguments={'use_sim_time': 'false', 'params_file': 'install/turtlebot3_navigation/param/nav2_params.yaml'}.items(),
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