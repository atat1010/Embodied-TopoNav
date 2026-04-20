import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 可选：启用 RealSense 作为数据源
    realsense_args = {
        'align_depth.enable': 'true',
        'enable_color': 'true',
        'enable_depth': 'true',
        'enable_infra1': 'false',
        'enable_infra2': 'false',
        'enable_gyro': 'false',
        'enable_accel': 'false',
        'rgb_camera.color_profile': '640x480x15',
        'depth_module.depth_profile': '640x480x15'
    }

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'),
            '/launch/rs_launch.py'
        ]),
        launch_arguments=realsense_args.items(),
        condition=IfCondition(LaunchConfiguration('use_realsense'))
    )

    # ORB-SLAM3 词典和参数文件路径
    home_dir = os.path.expanduser('~')
    orb_slam_dir = os.path.join(home_dir, 'semantic_slam_ws', 'src', 'ORB_SLAM3')
    orb_slam_lib_dir = os.path.join(orb_slam_dir, 'lib')
    pangolin_lib_dir = os.path.join(
        home_dir, 'download', 'paper_with_code', 'RTG-SLAM', 'thirdParty', 'Pangolin', 'build', 'src'
    )

    default_vocab_path = os.path.join(orb_slam_dir, 'Vocabulary', 'ORBvoc.txt')
    default_settings_path = os.path.join(orb_slam_dir, 'Examples', 'RGB-D', 'TUM3.yaml')
    default_model_path = os.path.join(home_dir, 'semantic_slam_ws', 'src', 'weights', 'yolov8n-seg.pt')
    default_semantic_rgbd_yaml = os.path.join(
        home_dir,
        'semantic_slam_ws',
        'src',
        'orb_slam3_ros2',
        'config',
        'semantic_rgbd_node.yaml'
    )
    default_slam_eval_yaml = os.path.join(
        home_dir,
        'semantic_slam_ws',
        'src',
        'orb_slam3_ros2',
        'config',
        'slam_eval_node.yaml'
    )
    default_semantic_pc_yaml = os.path.join(
        home_dir,
        'semantic_slam_ws',
        'src',
        'orb_slam3_ros2',
        'config',
        'semantic_pointcloud_builder.yaml'
    )
    default_bag_path = os.path.join(
        home_dir,
        'semantic_slam_ws',
        'src',
        'ros2bag',
        'freiburg3_walking',
        'ros2_walking_xyz',
        'ros2_walking_xyz.mcap'
    )

    ld_paths = [orb_slam_lib_dir, '/usr/local/lib']
    if os.path.isdir(pangolin_lib_dir):
        ld_paths.append(pangolin_lib_dir)
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    if current_ld:
        ld_paths.append(current_ld)

    set_ld_library_path = SetEnvironmentVariable(
        name='LD_LIBRARY_PATH',
        value=':'.join(ld_paths)
    )

    set_use_viewer_env = SetEnvironmentVariable(
        name='ORB_SLAM3_USE_VIEWER',
        value=LaunchConfiguration('use_viewer')
    )

    declare_args = [
        DeclareLaunchArgument('use_realsense', default_value='false'),
        DeclareLaunchArgument('use_viewer', default_value='true'),
        DeclareLaunchArgument('run_yolo', default_value='true'),
        DeclareLaunchArgument('run_eval', default_value='true'),
        DeclareLaunchArgument('run_semantic_map', default_value='true'),
        DeclareLaunchArgument('play_bag', default_value='true'),
        DeclareLaunchArgument('shutdown_when_bag_done', default_value='true'),
        DeclareLaunchArgument('bag_path', default_value=default_bag_path),
    ]

    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_path'), '--clock'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('play_bag'))
    )

    bag_done_shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=bag_play,
            on_exit=[
                EmitEvent(event=Shutdown(reason='ros2 bag playback finished'))
            ]
        ),
        condition=IfCondition(LaunchConfiguration('shutdown_when_bag_done'))
    )

    yolo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('yolo_semantic_ros2'),
            '/launch/yolo_semantic.launch.py'
        ]),
        condition=IfCondition(LaunchConfiguration('run_yolo')),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('play_bag'),
            'input_topic': '/camera/rgb/image_color',
            'combined_mask_topic': '/semantic/combined_mask',
            'overlay_topic': '/semantic/overlay',
            'model_path': default_model_path,
        }.items()
    )

    slam_node = Node(
        package='orb_slam3_ros2',
        executable='semantic_rgbd_node',
        name='semantic_rgbd_node',
        output='screen',
        prefix='xterm -e',
        arguments=[default_vocab_path, default_settings_path],
        parameters=[
            default_semantic_rgbd_yaml,
            {
                'use_sim_time': LaunchConfiguration('play_bag'),
                'yolo_expected': LaunchConfiguration('run_yolo'),
            }
        ]
    )

    eval_node = Node(
        package='orb_slam3_ros2',
        executable='slam_eval_node.py',
        name='slam_eval_node',
        output='screen',
        # prefix=['gnome-terminal --'],
        prefix='xterm -e',
        condition=IfCondition(LaunchConfiguration('run_eval')),
        parameters=[
            default_slam_eval_yaml,
            {
                'use_sim_time': LaunchConfiguration('play_bag'),
            }
        ]
    )

    semantic_map_node = Node(
        package='orb_slam3_ros2',
        executable='semantic_pointcloud_builder',
        name='semantic_pointcloud_builder',
        output='screen',
        # prefix='xterm -e',
        condition=IfCondition(LaunchConfiguration('run_semantic_map')),
        parameters=[
            default_semantic_pc_yaml,
            {
                'use_sim_time': LaunchConfiguration('play_bag'),
            }
        ]
    )

    return LaunchDescription([
        *declare_args,
        set_ld_library_path,
        set_use_viewer_env,
        realsense_launch,
        bag_play,
        bag_done_shutdown,
        yolo_launch,
        slam_node,
        eval_node,
        semantic_map_node
    ])