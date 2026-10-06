cmake_minimum_required(VERSION 3.8)
project(remote_data_collector_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BagInfo.msg"
  "msg/RecordingStatus.msg"
  "srv/GetStatus.srv"
  "srv/ListBags.srv"
  "srv/DeleteBag.srv"
  "srv/StartCollection.srv"
  "srv/StopCollection.srv"
  "action/RecordBag.action"
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
