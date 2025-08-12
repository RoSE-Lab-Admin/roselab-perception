cmake_minimum_required(VERSION 3.8)
project(realsense_bag_recorder_cpp)

# Enable C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Service files
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/RecordBag.srv"
)

# Executable
add_executable(bag_recorder_service src/bag_recorder_service.cpp)

# Link against the generated type support
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(bag_recorder_service ${cpp_typesupport_target})

# Add dependencies
ament_target_dependencies(bag_recorder_service
  rclcpp
)

# Include directories for generated headers
target_include_directories(bag_recorder_service PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp>
)

# Install target
install(TARGETS
  bag_recorder_service
  DESTINATION lib/${PROJECT_NAME}
)

# Install service files
install(DIRECTORY srv
  DESTINATION share/${PROJECT_NAME}
)

# Export package
ament_export_dependencies(rosidl_default_runtime)
ament_package()
