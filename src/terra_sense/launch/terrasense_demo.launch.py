from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    use_sim_time = LaunchConfiguration('simulation', default='True')

    bringup_dir = get_package_share_directory('nav2_bringup')
    turtlebot_gazebo_dir = get_package_share_directory('turtlebot3_gazebo')
    terrasense_layer_dir = get_package_share_directory('terra_sense')
    rviz_config_dir = os.path.join(bringup_dir, 'rviz', 'nav2_default_view.rviz')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
    'use_sim_time',
    default_value='true',
    description='Use simulation (Gazebo) clock if true')

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'params_file': os.path.join(terrasense_layer_dir, 'config', 'nav2_params.yaml'),
            'slam': 'False',
            'map': '',
            'use_sim_time' : use_sim_time,
            'autostart': 'True',
            'use_composition': 'True',
            'use_respawn': 'False'
        }.items(),
    )
    
    realsense = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
    )
    
    classification = Node(
        package='terra_sense',
        executable='terrain_classifier',
    )

    rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_dir],
        )
        
    turtlebot_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot_gazebo_dir, 'launch', 'turtlebot3_house.launch.py')
        ),
        condition=IfCondition(use_sim_time),
        launch_arguments={
            'use_sim_time' : use_sim_time,
            'x_pose' : '5',
            'y_pose' : '2',
        }.items(),
    )

    slam = IncludeLaunchDescription(
    PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')
    ),
    )

    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time_cmd)

    # launch ekf to combine imu and wheel odometry
    ld.add_action(nav2)
    # ld.add_action(realsense)
    # ld.add_action(classification)
    ld.add_action(turtlebot_sim)
    ld.add_action(slam)
    ld.add_action(rviz)

    return ld