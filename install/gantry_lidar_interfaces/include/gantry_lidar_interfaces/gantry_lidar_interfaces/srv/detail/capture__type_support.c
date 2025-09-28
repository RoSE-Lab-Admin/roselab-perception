// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from gantry_lidar_interfaces:srv/Capture.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "gantry_lidar_interfaces/srv/detail/capture__rosidl_typesupport_introspection_c.h"
#include "gantry_lidar_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "gantry_lidar_interfaces/srv/detail/capture__functions.h"
#include "gantry_lidar_interfaces/srv/detail/capture__struct.h"


// Include directives for member types
// Member `outname`
// Member `sensors`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  gantry_lidar_interfaces__srv__Capture_Request__init(message_memory);
}

void gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_fini_function(void * message_memory)
{
  gantry_lidar_interfaces__srv__Capture_Request__fini(message_memory);
}

size_t gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__size_function__Capture_Request__sensors(
  const void * untyped_member)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return member->size;
}

const void * gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__get_const_function__Capture_Request__sensors(
  const void * untyped_member, size_t index)
{
  const rosidl_runtime_c__String__Sequence * member =
    (const rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void * gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__get_function__Capture_Request__sensors(
  void * untyped_member, size_t index)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  return &member->data[index];
}

void gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__fetch_function__Capture_Request__sensors(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const rosidl_runtime_c__String * item =
    ((const rosidl_runtime_c__String *)
    gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__get_const_function__Capture_Request__sensors(untyped_member, index));
  rosidl_runtime_c__String * value =
    (rosidl_runtime_c__String *)(untyped_value);
  *value = *item;
}

void gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__assign_function__Capture_Request__sensors(
  void * untyped_member, size_t index, const void * untyped_value)
{
  rosidl_runtime_c__String * item =
    ((rosidl_runtime_c__String *)
    gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__get_function__Capture_Request__sensors(untyped_member, index));
  const rosidl_runtime_c__String * value =
    (const rosidl_runtime_c__String *)(untyped_value);
  *item = *value;
}

bool gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__resize_function__Capture_Request__sensors(
  void * untyped_member, size_t size)
{
  rosidl_runtime_c__String__Sequence * member =
    (rosidl_runtime_c__String__Sequence *)(untyped_member);
  rosidl_runtime_c__String__Sequence__fini(member);
  return rosidl_runtime_c__String__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_message_member_array[3] = {
  {
    "outname",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gantry_lidar_interfaces__srv__Capture_Request, outname),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "sensors",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gantry_lidar_interfaces__srv__Capture_Request, sensors),  // bytes offset in struct
    NULL,  // default value
    gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__size_function__Capture_Request__sensors,  // size() function pointer
    gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__get_const_function__Capture_Request__sensors,  // get_const(index) function pointer
    gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__get_function__Capture_Request__sensors,  // get(index) function pointer
    gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__fetch_function__Capture_Request__sensors,  // fetch(index, &value) function pointer
    gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__assign_function__Capture_Request__sensors,  // assign(index, value) function pointer
    gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__resize_function__Capture_Request__sensors  // resize(index) function pointer
  },
  {
    "duration",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gantry_lidar_interfaces__srv__Capture_Request, duration),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_message_members = {
  "gantry_lidar_interfaces__srv",  // message namespace
  "Capture_Request",  // message name
  3,  // number of fields
  sizeof(gantry_lidar_interfaces__srv__Capture_Request),
  false,  // has_any_key_member_
  gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_message_member_array,  // message members
  gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_message_type_support_handle = {
  0,
  &gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_message_members,
  get_message_typesupport_handle_function,
  &gantry_lidar_interfaces__srv__Capture_Request__get_type_hash,
  &gantry_lidar_interfaces__srv__Capture_Request__get_type_description,
  &gantry_lidar_interfaces__srv__Capture_Request__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_gantry_lidar_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Request)() {
  if (!gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_message_type_support_handle.typesupport_identifier) {
    gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "gantry_lidar_interfaces/srv/detail/capture__rosidl_typesupport_introspection_c.h"
// already included above
// #include "gantry_lidar_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "gantry_lidar_interfaces/srv/detail/capture__functions.h"
// already included above
// #include "gantry_lidar_interfaces/srv/detail/capture__struct.h"


// Include directives for member types
// Member `outdata`
// already included above
// #include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  gantry_lidar_interfaces__srv__Capture_Response__init(message_memory);
}

void gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_fini_function(void * message_memory)
{
  gantry_lidar_interfaces__srv__Capture_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_message_member_array[1] = {
  {
    "outdata",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gantry_lidar_interfaces__srv__Capture_Response, outdata),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_message_members = {
  "gantry_lidar_interfaces__srv",  // message namespace
  "Capture_Response",  // message name
  1,  // number of fields
  sizeof(gantry_lidar_interfaces__srv__Capture_Response),
  false,  // has_any_key_member_
  gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_message_member_array,  // message members
  gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_message_type_support_handle = {
  0,
  &gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_message_members,
  get_message_typesupport_handle_function,
  &gantry_lidar_interfaces__srv__Capture_Response__get_type_hash,
  &gantry_lidar_interfaces__srv__Capture_Response__get_type_description,
  &gantry_lidar_interfaces__srv__Capture_Response__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_gantry_lidar_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Response)() {
  if (!gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_message_type_support_handle.typesupport_identifier) {
    gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "gantry_lidar_interfaces/srv/detail/capture__rosidl_typesupport_introspection_c.h"
// already included above
// #include "gantry_lidar_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "gantry_lidar_interfaces/srv/detail/capture__functions.h"
// already included above
// #include "gantry_lidar_interfaces/srv/detail/capture__struct.h"


// Include directives for member types
// Member `info`
#include "service_msgs/msg/service_event_info.h"
// Member `info`
#include "service_msgs/msg/detail/service_event_info__rosidl_typesupport_introspection_c.h"
// Member `request`
// Member `response`
#include "gantry_lidar_interfaces/srv/capture.h"
// Member `request`
// Member `response`
// already included above
// #include "gantry_lidar_interfaces/srv/detail/capture__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  gantry_lidar_interfaces__srv__Capture_Event__init(message_memory);
}

void gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_fini_function(void * message_memory)
{
  gantry_lidar_interfaces__srv__Capture_Event__fini(message_memory);
}

size_t gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__size_function__Capture_Event__request(
  const void * untyped_member)
{
  const gantry_lidar_interfaces__srv__Capture_Request__Sequence * member =
    (const gantry_lidar_interfaces__srv__Capture_Request__Sequence *)(untyped_member);
  return member->size;
}

const void * gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_const_function__Capture_Event__request(
  const void * untyped_member, size_t index)
{
  const gantry_lidar_interfaces__srv__Capture_Request__Sequence * member =
    (const gantry_lidar_interfaces__srv__Capture_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void * gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_function__Capture_Event__request(
  void * untyped_member, size_t index)
{
  gantry_lidar_interfaces__srv__Capture_Request__Sequence * member =
    (gantry_lidar_interfaces__srv__Capture_Request__Sequence *)(untyped_member);
  return &member->data[index];
}

void gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__fetch_function__Capture_Event__request(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const gantry_lidar_interfaces__srv__Capture_Request * item =
    ((const gantry_lidar_interfaces__srv__Capture_Request *)
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_const_function__Capture_Event__request(untyped_member, index));
  gantry_lidar_interfaces__srv__Capture_Request * value =
    (gantry_lidar_interfaces__srv__Capture_Request *)(untyped_value);
  *value = *item;
}

void gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__assign_function__Capture_Event__request(
  void * untyped_member, size_t index, const void * untyped_value)
{
  gantry_lidar_interfaces__srv__Capture_Request * item =
    ((gantry_lidar_interfaces__srv__Capture_Request *)
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_function__Capture_Event__request(untyped_member, index));
  const gantry_lidar_interfaces__srv__Capture_Request * value =
    (const gantry_lidar_interfaces__srv__Capture_Request *)(untyped_value);
  *item = *value;
}

bool gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__resize_function__Capture_Event__request(
  void * untyped_member, size_t size)
{
  gantry_lidar_interfaces__srv__Capture_Request__Sequence * member =
    (gantry_lidar_interfaces__srv__Capture_Request__Sequence *)(untyped_member);
  gantry_lidar_interfaces__srv__Capture_Request__Sequence__fini(member);
  return gantry_lidar_interfaces__srv__Capture_Request__Sequence__init(member, size);
}

size_t gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__size_function__Capture_Event__response(
  const void * untyped_member)
{
  const gantry_lidar_interfaces__srv__Capture_Response__Sequence * member =
    (const gantry_lidar_interfaces__srv__Capture_Response__Sequence *)(untyped_member);
  return member->size;
}

const void * gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_const_function__Capture_Event__response(
  const void * untyped_member, size_t index)
{
  const gantry_lidar_interfaces__srv__Capture_Response__Sequence * member =
    (const gantry_lidar_interfaces__srv__Capture_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void * gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_function__Capture_Event__response(
  void * untyped_member, size_t index)
{
  gantry_lidar_interfaces__srv__Capture_Response__Sequence * member =
    (gantry_lidar_interfaces__srv__Capture_Response__Sequence *)(untyped_member);
  return &member->data[index];
}

void gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__fetch_function__Capture_Event__response(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const gantry_lidar_interfaces__srv__Capture_Response * item =
    ((const gantry_lidar_interfaces__srv__Capture_Response *)
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_const_function__Capture_Event__response(untyped_member, index));
  gantry_lidar_interfaces__srv__Capture_Response * value =
    (gantry_lidar_interfaces__srv__Capture_Response *)(untyped_value);
  *value = *item;
}

void gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__assign_function__Capture_Event__response(
  void * untyped_member, size_t index, const void * untyped_value)
{
  gantry_lidar_interfaces__srv__Capture_Response * item =
    ((gantry_lidar_interfaces__srv__Capture_Response *)
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_function__Capture_Event__response(untyped_member, index));
  const gantry_lidar_interfaces__srv__Capture_Response * value =
    (const gantry_lidar_interfaces__srv__Capture_Response *)(untyped_value);
  *item = *value;
}

bool gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__resize_function__Capture_Event__response(
  void * untyped_member, size_t size)
{
  gantry_lidar_interfaces__srv__Capture_Response__Sequence * member =
    (gantry_lidar_interfaces__srv__Capture_Response__Sequence *)(untyped_member);
  gantry_lidar_interfaces__srv__Capture_Response__Sequence__fini(member);
  return gantry_lidar_interfaces__srv__Capture_Response__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_member_array[3] = {
  {
    "info",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(gantry_lidar_interfaces__srv__Capture_Event, info),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "request",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(gantry_lidar_interfaces__srv__Capture_Event, request),  // bytes offset in struct
    NULL,  // default value
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__size_function__Capture_Event__request,  // size() function pointer
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_const_function__Capture_Event__request,  // get_const(index) function pointer
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_function__Capture_Event__request,  // get(index) function pointer
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__fetch_function__Capture_Event__request,  // fetch(index, &value) function pointer
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__assign_function__Capture_Event__request,  // assign(index, value) function pointer
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__resize_function__Capture_Event__request  // resize(index) function pointer
  },
  {
    "response",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is key
    true,  // is array
    1,  // array size
    true,  // is upper bound
    offsetof(gantry_lidar_interfaces__srv__Capture_Event, response),  // bytes offset in struct
    NULL,  // default value
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__size_function__Capture_Event__response,  // size() function pointer
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_const_function__Capture_Event__response,  // get_const(index) function pointer
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__get_function__Capture_Event__response,  // get(index) function pointer
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__fetch_function__Capture_Event__response,  // fetch(index, &value) function pointer
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__assign_function__Capture_Event__response,  // assign(index, value) function pointer
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__resize_function__Capture_Event__response  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_members = {
  "gantry_lidar_interfaces__srv",  // message namespace
  "Capture_Event",  // message name
  3,  // number of fields
  sizeof(gantry_lidar_interfaces__srv__Capture_Event),
  false,  // has_any_key_member_
  gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_member_array,  // message members
  gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_init_function,  // function to initialize message memory (memory has to be allocated)
  gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_type_support_handle = {
  0,
  &gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_members,
  get_message_typesupport_handle_function,
  &gantry_lidar_interfaces__srv__Capture_Event__get_type_hash,
  &gantry_lidar_interfaces__srv__Capture_Event__get_type_description,
  &gantry_lidar_interfaces__srv__Capture_Event__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_gantry_lidar_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Event)() {
  gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, service_msgs, msg, ServiceEventInfo)();
  gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Request)();
  gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Response)();
  if (!gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_type_support_handle.typesupport_identifier) {
    gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "gantry_lidar_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "gantry_lidar_interfaces/srv/detail/capture__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers gantry_lidar_interfaces__srv__detail__capture__rosidl_typesupport_introspection_c__Capture_service_members = {
  "gantry_lidar_interfaces__srv",  // service namespace
  "Capture",  // service name
  // the following fields are initialized below on first access
  NULL,  // request message
  // gantry_lidar_interfaces__srv__detail__capture__rosidl_typesupport_introspection_c__Capture_Request_message_type_support_handle,
  NULL,  // response message
  // gantry_lidar_interfaces__srv__detail__capture__rosidl_typesupport_introspection_c__Capture_Response_message_type_support_handle
  NULL  // event_message
  // gantry_lidar_interfaces__srv__detail__capture__rosidl_typesupport_introspection_c__Capture_Response_message_type_support_handle
};


static rosidl_service_type_support_t gantry_lidar_interfaces__srv__detail__capture__rosidl_typesupport_introspection_c__Capture_service_type_support_handle = {
  0,
  &gantry_lidar_interfaces__srv__detail__capture__rosidl_typesupport_introspection_c__Capture_service_members,
  get_service_typesupport_handle_function,
  &gantry_lidar_interfaces__srv__Capture_Request__rosidl_typesupport_introspection_c__Capture_Request_message_type_support_handle,
  &gantry_lidar_interfaces__srv__Capture_Response__rosidl_typesupport_introspection_c__Capture_Response_message_type_support_handle,
  &gantry_lidar_interfaces__srv__Capture_Event__rosidl_typesupport_introspection_c__Capture_Event_message_type_support_handle,
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_CREATE_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    gantry_lidar_interfaces,
    srv,
    Capture
  ),
  ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_DESTROY_EVENT_MESSAGE_SYMBOL_NAME(
    rosidl_typesupport_c,
    gantry_lidar_interfaces,
    srv,
    Capture
  ),
  &gantry_lidar_interfaces__srv__Capture__get_type_hash,
  &gantry_lidar_interfaces__srv__Capture__get_type_description,
  &gantry_lidar_interfaces__srv__Capture__get_type_description_sources,
};

// Forward declaration of message type support functions for service members
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Request)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Response)(void);

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Event)(void);

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_gantry_lidar_interfaces
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture)(void) {
  if (!gantry_lidar_interfaces__srv__detail__capture__rosidl_typesupport_introspection_c__Capture_service_type_support_handle.typesupport_identifier) {
    gantry_lidar_interfaces__srv__detail__capture__rosidl_typesupport_introspection_c__Capture_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)gantry_lidar_interfaces__srv__detail__capture__rosidl_typesupport_introspection_c__Capture_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Response)()->data;
  }
  if (!service_members->event_members_) {
    service_members->event_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, gantry_lidar_interfaces, srv, Capture_Event)()->data;
  }

  return &gantry_lidar_interfaces__srv__detail__capture__rosidl_typesupport_introspection_c__Capture_service_type_support_handle;
}
