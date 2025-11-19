from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace, Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from os.path import join

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_with = LaunchConfiguration('params_with_plugin')
    params_without = LaunchConfiguration('params_without_plugin')

    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    bringup = join(nav2_bringup_dir, 'launch', 'navigation_launch.py')

    tf_remap = [
        ('tf', '/tf'),                   # global TF tree
        ('tf_static', '/tf_static'),     # static TF
        ('map', '/map'),                 # map_server topic
        ('scan', '/scan'),               # laser scan input
        ('odom', '/odom'),               # odometry input
        ('cmd_vel', '/cmd_vel'),         # velocity commands
        ('initialpose', '/initialpose'), # RViz initial pose topic
        ('amcl_pose', '/amcl_pose'),     # AMCL output
        ('particlecloud', '/particlecloud'), # AMCL visualization
        ('set_pose', '/set_pose'),       # amcl set_pose service
        ('clear_costmap', '/global_costmap/clear_entirely_global_costmap'),
        ('clear_local_costmap', '/local_costmap/clear_entirely_local_costmap'),
    ]

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('params_with_plugin',
                              default_value='nav2_params.yaml',
                              description='YAML for stack with custom layer'),
        DeclareLaunchArgument('params_without_plugin',
                              default_value='nav2_params_noclass.yaml',
                              description='YAML for stack without custom layer'),

        # Stack A: with plugin
        GroupAction([
            PushRosNamespace('with_plugin'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bringup),
                launch_arguments={
                    # 'namespace': 'with_plugin',
                    # 'use_namespace':   'true',           # ← enable it!
                    'use_sim_time': use_sim_time,
                    'autostart': 'true',          # see step 2
                    # 'use_composition': 'False',    # <--- important
                    'slam': 'False',
                    # 'localization': 'false',
                    # 'map': ' ',
                    # 'container_name': 'with_plugin_nav2_container',
                    'remappings': str(tf_remap),
                    # 'log_level': 'debug',
                    'params_file': PathJoinSubstitution([
                    FindPackageShare('terra_sense'), 'config', params_with
                ])}.items()
            ),
        ]),
        # # Stack B: without plugin
        # GroupAction([
        #     PushRosNamespace('no_plugin'),
        #     IncludeLaunchDescription(
        #         PythonLaunchDescriptionSource(bringup),
        #         launch_arguments={
        #             # 'namespace': 'no_plugin',
        #             'use_namespace':   'true',           # ← enable it!
        #             'use_sim_time': use_sim_time,
        #             'autostart': 'true',          # see step 2
        #             # 'use_composition': 'False',    # <--- important
        #             # 'slam': 'False',
        #             # 'map': ' ',
        #             'localization': 'false',
        #             'remappings': str(tf_remap),
        #             'container_name': 'no_plugin_nav2_container',
        #             # 'log_level': 'debug',
        #             'params_file': PathJoinSubstitution([
        #             FindPackageShare('terra_sense'), 'config', params_without
        #         ])
        #         }.items()
        #     ),
        # ]),


        # Orchestrator node
        Node(
            package='terra_sense',
            executable='orchestrate_compare.py',
            name='orchestrate_compare',
            output='screen',
            parameters=[{
                'goal_x': 2.0,          # <-- set your goal
                'goal_y': 0.5,
                'goal_yaw': 0.0,
                'global_frame': 'map',
                'base_frame': 'base_link',
                'save_dir': '/tmp/nav2_compare'
            }]
        )
    ])
