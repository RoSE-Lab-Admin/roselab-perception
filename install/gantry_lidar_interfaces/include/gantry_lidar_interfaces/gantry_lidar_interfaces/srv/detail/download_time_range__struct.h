// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from gantry_lidar_interfaces:srv/DownloadTimeRange.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "gantry_lidar_interfaces/srv/download_time_range.h"


#ifndef GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DOWNLOAD_TIME_RANGE__STRUCT_H_
#define GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DOWNLOAD_TIME_RANGE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'start'
// Member 'end'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/DownloadTimeRange in the package gantry_lidar_interfaces.
typedef struct gantry_lidar_interfaces__srv__DownloadTimeRange_Request
{
  rosidl_runtime_c__String start;
  rosidl_runtime_c__String end;
} gantry_lidar_interfaces__srv__DownloadTimeRange_Request;

// Struct for a sequence of gantry_lidar_interfaces__srv__DownloadTimeRange_Request.
typedef struct gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence
{
  gantry_lidar_interfaces__srv__DownloadTimeRange_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'outdata'
// already included above
// #include "rosidl_runtime_c/string.h"

/// Struct defined in srv/DownloadTimeRange in the package gantry_lidar_interfaces.
typedef struct gantry_lidar_interfaces__srv__DownloadTimeRange_Response
{
  rosidl_runtime_c__String outdata;
} gantry_lidar_interfaces__srv__DownloadTimeRange_Response;

// Struct for a sequence of gantry_lidar_interfaces__srv__DownloadTimeRange_Response.
typedef struct gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence
{
  gantry_lidar_interfaces__srv__DownloadTimeRange_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence;

// Constants defined in the message

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__struct.h"

// constants for array fields with an upper bound
// request
enum
{
  gantry_lidar_interfaces__srv__DownloadTimeRange_Event__request__MAX_SIZE = 1
};
// response
enum
{
  gantry_lidar_interfaces__srv__DownloadTimeRange_Event__response__MAX_SIZE = 1
};

/// Struct defined in srv/DownloadTimeRange in the package gantry_lidar_interfaces.
typedef struct gantry_lidar_interfaces__srv__DownloadTimeRange_Event
{
  service_msgs__msg__ServiceEventInfo info;
  gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence request;
  gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence response;
} gantry_lidar_interfaces__srv__DownloadTimeRange_Event;

// Struct for a sequence of gantry_lidar_interfaces__srv__DownloadTimeRange_Event.
typedef struct gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence
{
  gantry_lidar_interfaces__srv__DownloadTimeRange_Event * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DOWNLOAD_TIME_RANGE__STRUCT_H_
