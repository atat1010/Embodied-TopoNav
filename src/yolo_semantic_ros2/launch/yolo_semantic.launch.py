import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    home_dir = os.path.expanduser('~')
    default_model_path = os.path.join(home_dir, 'semantic_slam_ws', 'src', 'weights', 'yolov8n-seg.pt')

    declare_args = [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('input_topic', default_value='/camera/rgb/image_color'),
        DeclareLaunchArgument('combined_mask_topic', default_value='/semantic/combined_mask'),
        DeclareLaunchArgument('overlay_topic', default_value='/semantic/overlay'),
        DeclareLaunchArgument('model_path', default_value=default_model_path),
    ]

    return LaunchDescription([
        *declare_args,
        Node(
            package='yolo_semantic_ros2',
            executable='yolo_mask_node',
            name='yolo_mask_node',
            output='screen',
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
                'publish_overlay': True,
                'target_classes': [0,56,62,64]
            }]
        )
    ])
