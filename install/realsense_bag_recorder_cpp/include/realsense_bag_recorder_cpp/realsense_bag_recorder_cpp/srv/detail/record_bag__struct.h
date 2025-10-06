// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from realsense_bag_recorder_cpp:srv/RecordBag.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "realsense_bag_recorder_cpp/srv/record_bag.h"


#ifndef REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__STRUCT_H_
#define REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'bag_name'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/RecordBag in the package realsense_bag_recorder_cpp.
typedef struct realsense_bag_recorder_cpp__srv__RecordBag_Request
{
  double duration;
  rosidl_runtime_c__String bag_name;
} realsense_bag_recorder_cpp__srv__RecordBag_Request;

// Struct for a sequence of realsense_bag_recorder_cpp__srv__RecordBag_Request.
typedef struct realsense_bag_recorder_cpp__srv__RecordBag_Request__Sequence
{
  realsense_bag_recorder_cpp__srv__RecordBag_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} realsense_bag_recorder_cpp__srv__RecordBag_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'message'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/RecordBag in the package realsense_bag_recorder_cpp.
typedef struct realsense_bag_recorder_cpp__srv__RecordBag_Response
{
  bool success;
  rosidl_runtime_c__String message;
} realsense_bag_recorder_cpp__srv__RecordBag_Response;

// Struct for a sequence of realsense_bag_recorder_cpp__srv__RecordBag_Response.
typedef struct realsense_bag_recorder_cpp__srv__RecordBag_Response__Sequence
{
  realsense_bag_recorder_cpp__srv__RecordBag_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} realsense_bag_recorder_cpp__srv__RecordBag_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  realsense_bag_recorder_cpp__srv__RecordBag_Event__request__MAX_SIZE = 1
};
// response
enum
{
  realsense_bag_recorder_cpp__srv__RecordBag_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/RecordBag in the package realsense_bag_recorder_cpp.
typedef struct realsense_bag_recorder_cpp__srv__RecordBag_Event
{
  service_msgs__msg__ServiceEventInfo info;
  realsense_bag_recorder_cpp__srv__RecordBag_Request__Sequence request;
  realsense_bag_recorder_cpp__srv__RecordBag_Response__Sequence response;
} realsense_bag_recorder_cpp__srv__RecordBag_Event;

// Struct for a sequence of realsense_bag_recorder_cpp__srv__RecordBag_Event.
typedef struct realsense_bag_recorder_cpp__srv__RecordBag_Event__Sequence
{
  realsense_bag_recorder_cpp__srv__RecordBag_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} realsense_bag_recorder_cpp__srv__RecordBag_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__STRUCT_H_
