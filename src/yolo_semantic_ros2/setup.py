from setuptools import find_packages, setup

package_name = 'yolo_semantic_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/yolo_semantic.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='qm',
    maintainer_email='1346186454@qq.com',
    description='YOLOv8 semantic mask publisher for ORB-SLAM3 integration.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_mask_node = yolo_semantic_ros2.yolo_mask_node:main',
            'vlm_brain_node = yolo_semantic_ros2.vlm_brain_node:main',
        ],
    },
)
