cmake_minimum_required(VERSION 3.8)
project(terra_sense)

set(lib_name ${PROJECT_NAME}_core)

# === Environment ===

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pluginlib REQUIRED)

set(dep_pkgs
  rclcpp
  rclpy
  std_msgs
  nav2_costmap_2d
  pluginlib
  sensor_msgs
  pcl_conversions
  pcl_ros
)


# either library or executable
add_library(${lib_name}  SHARED 
  src/terrain_layer.cpp)

# Cpp
include_directories(
  include
)

install(TARGETS ${lib_name}
DESTINATION lib
)
# DIRECTORY
# FILES

pluginlib_export_plugin_description_file(nav2_costmap_2d terrain_layer.xml)
ament_target_dependencies(${lib_name} ${dep_pkgs})

install(
  DIRECTORY 
  launch 
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY 
  config
  DESTINATION share/${PROJECT_NAME}
)

# Python
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/terrain_publisher.py
  scripts/speed_adjust.py
  scripts/terrain_publisher_cpu.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(${dep_pkgs})

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
