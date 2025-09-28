// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from gantry_lidar_interfaces:srv/DeleteName.idl
// generated code does not contain a copyright notice

#include "gantry_lidar_interfaces/srv/detail/delete_name__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_gantry_lidar_interfaces
const rosidl_type_hash_t *
gantry_lidar_interfaces__srv__DeleteName__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xac, 0x09, 0xf4, 0xc2, 0x1b, 0x50, 0xe2, 0x2f,
      0xe3, 0x75, 0x01, 0x3f, 0xa0, 0xfd, 0xcb, 0x5b,
      0x41, 0xc2, 0xe5, 0x91, 0x38, 0x85, 0xa7, 0x30,
      0xff, 0xae, 0x2c, 0x7c, 0xd3, 0x6d, 0xc5, 0xf9,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_gantry_lidar_interfaces
const rosidl_type_hash_t *
gantry_lidar_interfaces__srv__DeleteName_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x4a, 0xc3, 0xeb, 0x1e, 0x99, 0xd3, 0x03, 0xc2,
      0x17, 0x36, 0x72, 0x6b, 0xc1, 0xab, 0xdd, 0x32,
      0xa6, 0x70, 0x66, 0x4d, 0x13, 0x5f, 0xd8, 0x92,
      0x9a, 0xfa, 0xed, 0xdf, 0xaf, 0xaa, 0x0d, 0xdf,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_gantry_lidar_interfaces
const rosidl_type_hash_t *
gantry_lidar_interfaces__srv__DeleteName_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x13, 0x73, 0x80, 0x7f, 0x61, 0xf8, 0x31, 0x1a,
      0xcd, 0x9f, 0xf4, 0x4b, 0x96, 0x37, 0xe8, 0x0f,
      0x48, 0x0e, 0x69, 0xda, 0x47, 0x3b, 0x5f, 0x21,
      0xaa, 0xb5, 0x72, 0x77, 0xff, 0x30, 0x2a, 0xae,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_gantry_lidar_interfaces
const rosidl_type_hash_t *
gantry_lidar_interfaces__srv__DeleteName_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x68, 0x6a, 0x5c, 0x32, 0x7d, 0x97, 0x96, 0x6c,
      0xd3, 0x59, 0xb4, 0xdd, 0xa7, 0x97, 0x9e, 0x08,
      0x14, 0xe2, 0xef, 0xe1, 0x16, 0xdd, 0x49, 0x0f,
      0x3c, 0x4a, 0x9b, 0xc3, 0x63, 0xea, 0xb7, 0x75,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "builtin_interfaces/msg/detail/time__functions.h"
#include "service_msgs/msg/detail/service_event_info__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t service_msgs__msg__ServiceEventInfo__EXPECTED_HASH = {1, {
    0x41, 0xbc, 0xbb, 0xe0, 0x7a, 0x75, 0xc9, 0xb5,
    0x2b, 0xc9, 0x6b, 0xfd, 0x5c, 0x24, 0xd7, 0xf0,
    0xfc, 0x0a, 0x08, 0xc0, 0xcb, 0x79, 0x21, 0xb3,
    0x37, 0x3c, 0x57, 0x32, 0x34, 0x5a, 0x6f, 0x45,
  }};
#endif

static char gantry_lidar_interfaces__srv__DeleteName__TYPE_NAME[] = "gantry_lidar_interfaces/srv/DeleteName";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char gantry_lidar_interfaces__srv__DeleteName_Event__TYPE_NAME[] = "gantry_lidar_interfaces/srv/DeleteName_Event";
static char gantry_lidar_interfaces__srv__DeleteName_Request__TYPE_NAME[] = "gantry_lidar_interfaces/srv/DeleteName_Request";
static char gantry_lidar_interfaces__srv__DeleteName_Response__TYPE_NAME[] = "gantry_lidar_interfaces/srv/DeleteName_Response";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";

// Define type names, field names, and default values
static char gantry_lidar_interfaces__srv__DeleteName__FIELD_NAME__request_message[] = "request_message";
static char gantry_lidar_interfaces__srv__DeleteName__FIELD_NAME__response_message[] = "response_message";
static char gantry_lidar_interfaces__srv__DeleteName__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field gantry_lidar_interfaces__srv__DeleteName__FIELDS[] = {
  {
    {gantry_lidar_interfaces__srv__DeleteName__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {gantry_lidar_interfaces__srv__DeleteName_Request__TYPE_NAME, 46, 46},
    },
    {NULL, 0, 0},
  },
  {
    {gantry_lidar_interfaces__srv__DeleteName__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {gantry_lidar_interfaces__srv__DeleteName_Response__TYPE_NAME, 47, 47},
    },
    {NULL, 0, 0},
  },
  {
    {gantry_lidar_interfaces__srv__DeleteName__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {gantry_lidar_interfaces__srv__DeleteName_Event__TYPE_NAME, 44, 44},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription gantry_lidar_interfaces__srv__DeleteName__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {gantry_lidar_interfaces__srv__DeleteName_Event__TYPE_NAME, 44, 44},
    {NULL, 0, 0},
  },
  {
    {gantry_lidar_interfaces__srv__DeleteName_Request__TYPE_NAME, 46, 46},
    {NULL, 0, 0},
  },
  {
    {gantry_lidar_interfaces__srv__DeleteName_Response__TYPE_NAME, 47, 47},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
gantry_lidar_interfaces__srv__DeleteName__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {gantry_lidar_interfaces__srv__DeleteName__TYPE_NAME, 38, 38},
      {gantry_lidar_interfaces__srv__DeleteName__FIELDS, 3, 3},
    },
    {gantry_lidar_interfaces__srv__DeleteName__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = gantry_lidar_interfaces__srv__DeleteName_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = gantry_lidar_interfaces__srv__DeleteName_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = gantry_lidar_interfaces__srv__DeleteName_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[4].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char gantry_lidar_interfaces__srv__DeleteName_Request__FIELD_NAME__name[] = "name";

static rosidl_runtime_c__type_description__Field gantry_lidar_interfaces__srv__DeleteName_Request__FIELDS[] = {
  {
    {gantry_lidar_interfaces__srv__DeleteName_Request__FIELD_NAME__name, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
gantry_lidar_interfaces__srv__DeleteName_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {gantry_lidar_interfaces__srv__DeleteName_Request__TYPE_NAME, 46, 46},
      {gantry_lidar_interfaces__srv__DeleteName_Request__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char gantry_lidar_interfaces__srv__DeleteName_Response__FIELD_NAME__outdata[] = "outdata";

static rosidl_runtime_c__type_description__Field gantry_lidar_interfaces__srv__DeleteName_Response__FIELDS[] = {
  {
    {gantry_lidar_interfaces__srv__DeleteName_Response__FIELD_NAME__outdata, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_STRING,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
gantry_lidar_interfaces__srv__DeleteName_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {gantry_lidar_interfaces__srv__DeleteName_Response__TYPE_NAME, 47, 47},
      {gantry_lidar_interfaces__srv__DeleteName_Response__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char gantry_lidar_interfaces__srv__DeleteName_Event__FIELD_NAME__info[] = "info";
static char gantry_lidar_interfaces__srv__DeleteName_Event__FIELD_NAME__request[] = "request";
static char gantry_lidar_interfaces__srv__DeleteName_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field gantry_lidar_interfaces__srv__DeleteName_Event__FIELDS[] = {
  {
    {gantry_lidar_interfaces__srv__DeleteName_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {gantry_lidar_interfaces__srv__DeleteName_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {gantry_lidar_interfaces__srv__DeleteName_Request__TYPE_NAME, 46, 46},
    },
    {NULL, 0, 0},
  },
  {
    {gantry_lidar_interfaces__srv__DeleteName_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {gantry_lidar_interfaces__srv__DeleteName_Response__TYPE_NAME, 47, 47},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription gantry_lidar_interfaces__srv__DeleteName_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {gantry_lidar_interfaces__srv__DeleteName_Request__TYPE_NAME, 46, 46},
    {NULL, 0, 0},
  },
  {
    {gantry_lidar_interfaces__srv__DeleteName_Response__TYPE_NAME, 47, 47},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
gantry_lidar_interfaces__srv__DeleteName_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {gantry_lidar_interfaces__srv__DeleteName_Event__TYPE_NAME, 44, 44},
      {gantry_lidar_interfaces__srv__DeleteName_Event__FIELDS, 3, 3},
    },
    {gantry_lidar_interfaces__srv__DeleteName_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[1].fields = gantry_lidar_interfaces__srv__DeleteName_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = gantry_lidar_interfaces__srv__DeleteName_Response__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[3].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "string name\n"
  "---\n"
  "string outdata";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
gantry_lidar_interfaces__srv__DeleteName__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {gantry_lidar_interfaces__srv__DeleteName__TYPE_NAME, 38, 38},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 30, 30},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
gantry_lidar_interfaces__srv__DeleteName_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {gantry_lidar_interfaces__srv__DeleteName_Request__TYPE_NAME, 46, 46},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
gantry_lidar_interfaces__srv__DeleteName_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {gantry_lidar_interfaces__srv__DeleteName_Response__TYPE_NAME, 47, 47},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
gantry_lidar_interfaces__srv__DeleteName_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {gantry_lidar_interfaces__srv__DeleteName_Event__TYPE_NAME, 44, 44},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
gantry_lidar_interfaces__srv__DeleteName__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *gantry_lidar_interfaces__srv__DeleteName__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *gantry_lidar_interfaces__srv__DeleteName_Event__get_individual_type_description_source(NULL);
    sources[3] = *gantry_lidar_interfaces__srv__DeleteName_Request__get_individual_type_description_source(NULL);
    sources[4] = *gantry_lidar_interfaces__srv__DeleteName_Response__get_individual_type_description_source(NULL);
    sources[5] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
gantry_lidar_interfaces__srv__DeleteName_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *gantry_lidar_interfaces__srv__DeleteName_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
gantry_lidar_interfaces__srv__DeleteName_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *gantry_lidar_interfaces__srv__DeleteName_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
gantry_lidar_interfaces__srv__DeleteName_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *gantry_lidar_interfaces__srv__DeleteName_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *gantry_lidar_interfaces__srv__DeleteName_Request__get_individual_type_description_source(NULL);
    sources[3] = *gantry_lidar_interfaces__srv__DeleteName_Response__get_individual_type_description_source(NULL);
    sources[4] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}
