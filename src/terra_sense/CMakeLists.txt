cmake_minimum_required(VERSION 3.8)
project(terra-sense)

set(lib_name ${PROJECT_NAME}-core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_core REQUIRED)
find_package(pluginlib REQUIRED)

set(dep_pkgs
  rclcpp
  std_msgs
  nav2_costmap_2d
  pluginlib
)

# Cpp
include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${std_msgs_INCLUDE_DIRS}
  ${nav2_costmap_2d_INCLUDE_DIRS}
)
# either library or executable
add_library(${lib_name}  SHARED src/terrain_layer.cpp)

install(
  DIRECTORY 
  launch 
  DESTINATION /share/${PROJECT_NAME}
)

install(
  DIRECTORY 
  config
  DESTINATION /share/${PROJECT_NAME}
)

install(TARGETS ${lib_name}
DESTINATION lib/${lib_name}
)
# DIRECTORY
# FILES

pluginlib_export_plugin_description_file(nav2_costmap_2d terrain_costmap_layer.xml)
ament_target_dependencies(${lib_name} ${dep_pkgs})

# Python
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/terrain_publisher.py
  scripts/spped_adjust.py
  DESTINATION /share/${PROJECT_NAME}
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
