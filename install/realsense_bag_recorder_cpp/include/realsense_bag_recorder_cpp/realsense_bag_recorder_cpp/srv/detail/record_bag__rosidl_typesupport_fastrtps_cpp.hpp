// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__rosidl_typesupport_fastrtps_cpp.hpp.em
// with input from realsense_bag_recorder_cpp:srv/RecordBag.idl
// generated code does not contain a copyright notice

#ifndef REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
#define REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_

#include <cstddef>
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_interface/macros.h"
#include "realsense_bag_recorder_cpp/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
#include "realsense_bag_recorder_cpp/srv/detail/record_bag__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

#include "fastcdr/Cdr.h"

namespace realsense_bag_recorder_cpp
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
cdr_serialize(
  const realsense_bag_recorder_cpp::srv::RecordBag_Request & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  realsense_bag_recorder_cpp::srv::RecordBag_Request & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
get_serialized_size(
  const realsense_bag_recorder_cpp::srv::RecordBag_Request & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
max_serialized_size_RecordBag_Request(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
cdr_serialize_key(
  const realsense_bag_recorder_cpp::srv::RecordBag_Request & ros_message,
  eprosima::fastcdr::Cdr &);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
get_serialized_size_key(
  const realsense_bag_recorder_cpp::srv::RecordBag_Request & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
max_serialized_size_key_RecordBag_Request(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace realsense_bag_recorder_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, realsense_bag_recorder_cpp, srv, RecordBag_Request)();

#ifdef __cplusplus
}
#endif

// already included above
// #include <cstddef>
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "realsense_bag_recorder_cpp/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
// already included above
// #include "realsense_bag_recorder_cpp/srv/detail/record_bag__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// already included above
// #include "fastcdr/Cdr.h"

namespace realsense_bag_recorder_cpp
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
cdr_serialize(
  const realsense_bag_recorder_cpp::srv::RecordBag_Response & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  realsense_bag_recorder_cpp::srv::RecordBag_Response & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
get_serialized_size(
  const realsense_bag_recorder_cpp::srv::RecordBag_Response & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
max_serialized_size_RecordBag_Response(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
cdr_serialize_key(
  const realsense_bag_recorder_cpp::srv::RecordBag_Response & ros_message,
  eprosima::fastcdr::Cdr &);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
get_serialized_size_key(
  const realsense_bag_recorder_cpp::srv::RecordBag_Response & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
max_serialized_size_key_RecordBag_Response(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace realsense_bag_recorder_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, realsense_bag_recorder_cpp, srv, RecordBag_Response)();

#ifdef __cplusplus
}
#endif

// already included above
// #include <cstddef>
// already included above
// #include "rosidl_runtime_c/message_type_support_struct.h"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "realsense_bag_recorder_cpp/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"
// already included above
// #include "realsense_bag_recorder_cpp/srv/detail/record_bag__struct.hpp"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// already included above
// #include "fastcdr/Cdr.h"

namespace realsense_bag_recorder_cpp
{

namespace srv
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
cdr_serialize(
  const realsense_bag_recorder_cpp::srv::RecordBag_Event & ros_message,
  eprosima::fastcdr::Cdr & cdr);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  realsense_bag_recorder_cpp::srv::RecordBag_Event & ros_message);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
get_serialized_size(
  const realsense_bag_recorder_cpp::srv::RecordBag_Event & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
max_serialized_size_RecordBag_Event(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
cdr_serialize_key(
  const realsense_bag_recorder_cpp::srv::RecordBag_Event & ros_message,
  eprosima::fastcdr::Cdr &);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
get_serialized_size_key(
  const realsense_bag_recorder_cpp::srv::RecordBag_Event & ros_message,
  size_t current_alignment);

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
max_serialized_size_key_RecordBag_Event(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment);

}  // namespace typesupport_fastrtps_cpp

}  // namespace srv

}  // namespace realsense_bag_recorder_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, realsense_bag_recorder_cpp, srv, RecordBag_Event)();

#ifdef __cplusplus
}
#endif

#include "rmw/types.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_interface/macros.h"
// already included above
// #include "realsense_bag_recorder_cpp/msg/rosidl_typesupport_fastrtps_cpp__visibility_control.h"

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_realsense_bag_recorder_cpp
const rosidl_service_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, realsense_bag_recorder_cpp, srv, RecordBag)();

#ifdef __cplusplus
}
#endif

#endif  // REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__ROSIDL_TYPESUPPORT_FASTRTPS_CPP_HPP_
