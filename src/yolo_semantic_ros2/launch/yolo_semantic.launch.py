import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    home_dir = os.path.expanduser('~')
    default_model_path = os.path.join(home_dir, 'semantic_slam_ws', 'src', 'weights', 'yolov8s-seg.pt')

    declare_args = [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('input_topic', default_value='/camera/rgb/image_color'),
        DeclareLaunchArgument('combined_mask_topic', default_value='/semantic/combined_mask'),
        DeclareLaunchArgument('overlay_topic', default_value='/semantic/overlay'),
        DeclareLaunchArgument('model_path', default_value=default_model_path),
    ]

    vlm_args = [
        DeclareLaunchArgument('base_url', default_value='https://dashscope.aliyuncs.com/compatible-mode/v1'),
        DeclareLaunchArgument('model_name', default_value='qwen3-omni-flash'),
    ]

    yolo_node = Node(
            package='yolo_semantic_ros2',
            executable='yolo_mask_node',
            name='yolo_mask_node',
            output='screen',
            prefix='xterm -e',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'input_topic': LaunchConfiguration('input_topic'),
                'combined_mask_topic': LaunchConfiguration('combined_mask_topic'),
                'overlay_topic': LaunchConfiguration('overlay_topic'),
                'model_path': LaunchConfiguration('model_path'),
                'conf': 0.35,
                'iou': 0.5,
                'device': 'auto',
                'imgsz': 640,
                'half': True,
                'publish_overlay': False,
                'target_classes': [0,56,62,64]
            }]
        )

    vlm_node = Node(
        package='yolo_semantic_ros2',
        executable='vlm_brain_node',
        name='vlm_brain_node',
        output='screen',
        # prefix='xterm -e',
        parameters=[{
            'base_url': LaunchConfiguration('base_url'),
            'model_name': LaunchConfiguration('model_name')
        }]
    )

    return LaunchDescription([
        *declare_args,
        *vlm_args,
        yolo_node,
        vlm_node
    ])
