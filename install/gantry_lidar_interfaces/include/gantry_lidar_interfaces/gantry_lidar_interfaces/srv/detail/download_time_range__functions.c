// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from gantry_lidar_interfaces:srv/DownloadTimeRange.idl
// generated code does not contain a copyright notice
#include "gantry_lidar_interfaces/srv/detail/download_time_range__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `start`
// Member `end`
#include "rosidl_runtime_c/string_functions.h"

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__init(gantry_lidar_interfaces__srv__DownloadTimeRange_Request * msg)
{
  if (!msg) {
    return false;
  }
  // start
  if (!rosidl_runtime_c__String__init(&msg->start)) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Request__fini(msg);
    return false;
  }
  // end
  if (!rosidl_runtime_c__String__init(&msg->end)) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Request__fini(msg);
    return false;
  }
  return true;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__fini(gantry_lidar_interfaces__srv__DownloadTimeRange_Request * msg)
{
  if (!msg) {
    return;
  }
  // start
  rosidl_runtime_c__String__fini(&msg->start);
  // end
  rosidl_runtime_c__String__fini(&msg->end);
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__are_equal(const gantry_lidar_interfaces__srv__DownloadTimeRange_Request * lhs, const gantry_lidar_interfaces__srv__DownloadTimeRange_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // start
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->start), &(rhs->start)))
  {
    return false;
  }
  // end
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->end), &(rhs->end)))
  {
    return false;
  }
  return true;
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__copy(
  const gantry_lidar_interfaces__srv__DownloadTimeRange_Request * input,
  gantry_lidar_interfaces__srv__DownloadTimeRange_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // start
  if (!rosidl_runtime_c__String__copy(
      &(input->start), &(output->start)))
  {
    return false;
  }
  // end
  if (!rosidl_runtime_c__String__copy(
      &(input->end), &(output->end)))
  {
    return false;
  }
  return true;
}

gantry_lidar_interfaces__srv__DownloadTimeRange_Request *
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gantry_lidar_interfaces__srv__DownloadTimeRange_Request * msg = (gantry_lidar_interfaces__srv__DownloadTimeRange_Request *)allocator.allocate(sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Request));
  bool success = gantry_lidar_interfaces__srv__DownloadTimeRange_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__destroy(gantry_lidar_interfaces__srv__DownloadTimeRange_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__init(gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gantry_lidar_interfaces__srv__DownloadTimeRange_Request * data = NULL;

  if (size) {
    data = (gantry_lidar_interfaces__srv__DownloadTimeRange_Request *)allocator.zero_allocate(size, sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = gantry_lidar_interfaces__srv__DownloadTimeRange_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        gantry_lidar_interfaces__srv__DownloadTimeRange_Request__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__fini(gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      gantry_lidar_interfaces__srv__DownloadTimeRange_Request__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence *
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence * array = (gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence *)allocator.allocate(sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__destroy(gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__are_equal(const gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence * lhs, const gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__copy(
  const gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence * input,
  gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Request);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    gantry_lidar_interfaces__srv__DownloadTimeRange_Request * data =
      (gantry_lidar_interfaces__srv__DownloadTimeRange_Request *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Request__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          gantry_lidar_interfaces__srv__DownloadTimeRange_Request__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `outdata`
// already included above
// #include "rosidl_runtime_c/string_functions.h"

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__init(gantry_lidar_interfaces__srv__DownloadTimeRange_Response * msg)
{
  if (!msg) {
    return false;
  }
  // outdata
  if (!rosidl_runtime_c__String__init(&msg->outdata)) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Response__fini(msg);
    return false;
  }
  return true;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__fini(gantry_lidar_interfaces__srv__DownloadTimeRange_Response * msg)
{
  if (!msg) {
    return;
  }
  // outdata
  rosidl_runtime_c__String__fini(&msg->outdata);
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__are_equal(const gantry_lidar_interfaces__srv__DownloadTimeRange_Response * lhs, const gantry_lidar_interfaces__srv__DownloadTimeRange_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // outdata
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->outdata), &(rhs->outdata)))
  {
    return false;
  }
  return true;
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__copy(
  const gantry_lidar_interfaces__srv__DownloadTimeRange_Response * input,
  gantry_lidar_interfaces__srv__DownloadTimeRange_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // outdata
  if (!rosidl_runtime_c__String__copy(
      &(input->outdata), &(output->outdata)))
  {
    return false;
  }
  return true;
}

gantry_lidar_interfaces__srv__DownloadTimeRange_Response *
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gantry_lidar_interfaces__srv__DownloadTimeRange_Response * msg = (gantry_lidar_interfaces__srv__DownloadTimeRange_Response *)allocator.allocate(sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Response));
  bool success = gantry_lidar_interfaces__srv__DownloadTimeRange_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__destroy(gantry_lidar_interfaces__srv__DownloadTimeRange_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__init(gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gantry_lidar_interfaces__srv__DownloadTimeRange_Response * data = NULL;

  if (size) {
    data = (gantry_lidar_interfaces__srv__DownloadTimeRange_Response *)allocator.zero_allocate(size, sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = gantry_lidar_interfaces__srv__DownloadTimeRange_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        gantry_lidar_interfaces__srv__DownloadTimeRange_Response__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__fini(gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      gantry_lidar_interfaces__srv__DownloadTimeRange_Response__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence *
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence * array = (gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence *)allocator.allocate(sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__destroy(gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__are_equal(const gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence * lhs, const gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__copy(
  const gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence * input,
  gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Response);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    gantry_lidar_interfaces__srv__DownloadTimeRange_Response * data =
      (gantry_lidar_interfaces__srv__DownloadTimeRange_Response *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Response__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          gantry_lidar_interfaces__srv__DownloadTimeRange_Response__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `info`
#include "service_msgs/msg/detail/service_event_info__functions.h"
// Member `request`
// Member `response`
// already included above
// #include "gantry_lidar_interfaces/srv/detail/download_time_range__functions.h"

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__init(gantry_lidar_interfaces__srv__DownloadTimeRange_Event * msg)
{
  if (!msg) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__init(&msg->info)) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Event__fini(msg);
    return false;
  }
  // request
  if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__init(&msg->request, 0)) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Event__fini(msg);
    return false;
  }
  // response
  if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__init(&msg->response, 0)) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Event__fini(msg);
    return false;
  }
  return true;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__fini(gantry_lidar_interfaces__srv__DownloadTimeRange_Event * msg)
{
  if (!msg) {
    return;
  }
  // info
  service_msgs__msg__ServiceEventInfo__fini(&msg->info);
  // request
  gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__fini(&msg->request);
  // response
  gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__fini(&msg->response);
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__are_equal(const gantry_lidar_interfaces__srv__DownloadTimeRange_Event * lhs, const gantry_lidar_interfaces__srv__DownloadTimeRange_Event * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__are_equal(
      &(lhs->info), &(rhs->info)))
  {
    return false;
  }
  // request
  if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__are_equal(
      &(lhs->request), &(rhs->request)))
  {
    return false;
  }
  // response
  if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__are_equal(
      &(lhs->response), &(rhs->response)))
  {
    return false;
  }
  return true;
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__copy(
  const gantry_lidar_interfaces__srv__DownloadTimeRange_Event * input,
  gantry_lidar_interfaces__srv__DownloadTimeRange_Event * output)
{
  if (!input || !output) {
    return false;
  }
  // info
  if (!service_msgs__msg__ServiceEventInfo__copy(
      &(input->info), &(output->info)))
  {
    return false;
  }
  // request
  if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Request__Sequence__copy(
      &(input->request), &(output->request)))
  {
    return false;
  }
  // response
  if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Response__Sequence__copy(
      &(input->response), &(output->response)))
  {
    return false;
  }
  return true;
}

gantry_lidar_interfaces__srv__DownloadTimeRange_Event *
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__create(void)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gantry_lidar_interfaces__srv__DownloadTimeRange_Event * msg = (gantry_lidar_interfaces__srv__DownloadTimeRange_Event *)allocator.allocate(sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Event), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Event));
  bool success = gantry_lidar_interfaces__srv__DownloadTimeRange_Event__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__destroy(gantry_lidar_interfaces__srv__DownloadTimeRange_Event * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Event__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence__init(gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gantry_lidar_interfaces__srv__DownloadTimeRange_Event * data = NULL;

  if (size) {
    data = (gantry_lidar_interfaces__srv__DownloadTimeRange_Event *)allocator.zero_allocate(size, sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Event), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = gantry_lidar_interfaces__srv__DownloadTimeRange_Event__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        gantry_lidar_interfaces__srv__DownloadTimeRange_Event__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence__fini(gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      gantry_lidar_interfaces__srv__DownloadTimeRange_Event__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence *
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence * array = (gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence *)allocator.allocate(sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence__destroy(gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence__are_equal(const gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence * lhs, const gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Event__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence__copy(
  const gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence * input,
  gantry_lidar_interfaces__srv__DownloadTimeRange_Event__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(gantry_lidar_interfaces__srv__DownloadTimeRange_Event);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    gantry_lidar_interfaces__srv__DownloadTimeRange_Event * data =
      (gantry_lidar_interfaces__srv__DownloadTimeRange_Event *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Event__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          gantry_lidar_interfaces__srv__DownloadTimeRange_Event__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!gantry_lidar_interfaces__srv__DownloadTimeRange_Event__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
