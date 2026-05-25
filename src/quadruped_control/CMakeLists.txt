cmake_minimum_required(VERSION 3.8)
project(quadruped_control)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(quadruped_interfaces REQUIRED)
find_package(std_msgs REQUIRED)

# ---- serial_bridge_node ----
add_executable(serial_bridge_node src/serial_bridge_node.cpp)
target_include_directories(serial_bridge_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(serial_bridge_node
  rclcpp
  quadruped_interfaces
  std_msgs
)

# ---- strategy_manager_node ----
add_executable(strategy_manager_node src/strategy_manager_node.cpp)
target_include_directories(strategy_manager_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(strategy_manager_node
  rclcpp
  quadruped_interfaces
  std_msgs
)

install(TARGETS
  serial_bridge_node
  strategy_manager_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

ament_package()
