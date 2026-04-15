cmake_minimum_required(VERSION 3.8)
project(orb_slam3_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED) # SLAM 专属 3D 渲染库
find_package(PCL REQUIRED COMPONENTS common io filters)

set(ORB_SLAM3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ORB_SLAM3)

# 告诉编译器去哪里找 SLAM 的头文件 (.h)
include_directories(
  ${ORB_SLAM3_DIR}
  ${ORB_SLAM3_DIR}/include
  ${ORB_SLAM3_DIR}/include/CameraModels
  ${ORB_SLAM3_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(semantic_rgbd_node src/semantic_rgbd_node.cpp)
add_executable(semantic_pointcloud_builder src/semantic_pointcloud_builder.cpp)

# 链接 ROS 2 的库
ament_target_dependencies(semantic_rgbd_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  message_filters
)

ament_target_dependencies(semantic_pointcloud_builder
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  message_filters
  pcl_conversions
  pcl_ros
)

target_compile_features(semantic_pointcloud_builder PUBLIC cxx_std_17)

# 链接 ORB-SLAM3 核心动态库 (.so) 以及系统库
target_link_libraries(semantic_rgbd_node
  ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
)

target_link_libraries(semantic_pointcloud_builder
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

install(TARGETS 
  semantic_rgbd_node
  semantic_pointcloud_builder
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/slam_eval_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
