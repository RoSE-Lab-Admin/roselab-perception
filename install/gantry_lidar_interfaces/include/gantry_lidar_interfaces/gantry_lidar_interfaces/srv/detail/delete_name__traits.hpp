// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from gantry_lidar_interfaces:srv/DeleteName.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "gantry_lidar_interfaces/srv/delete_name.hpp"


#ifndef GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DELETE_NAME__TRAITS_HPP_
#define GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DELETE_NAME__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "gantry_lidar_interfaces/srv/detail/delete_name__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace gantry_lidar_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const DeleteName_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: name
  {
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DeleteName_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    rosidl_generator_traits::value_to_yaml(msg.name, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DeleteName_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace gantry_lidar_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use gantry_lidar_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const gantry_lidar_interfaces::srv::DeleteName_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  gantry_lidar_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use gantry_lidar_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const gantry_lidar_interfaces::srv::DeleteName_Request & msg)
{
  return gantry_lidar_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<gantry_lidar_interfaces::srv::DeleteName_Request>()
{
  return "gantry_lidar_interfaces::srv::DeleteName_Request";
}

template<>
inline const char * name<gantry_lidar_interfaces::srv::DeleteName_Request>()
{
  return "gantry_lidar_interfaces/srv/DeleteName_Request";
}

template<>
struct has_fixed_size<gantry_lidar_interfaces::srv::DeleteName_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<gantry_lidar_interfaces::srv::DeleteName_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<gantry_lidar_interfaces::srv::DeleteName_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace gantry_lidar_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const DeleteName_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: outdata
  {
    out << "outdata: ";
    rosidl_generator_traits::value_to_yaml(msg.outdata, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DeleteName_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: outdata
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "outdata: ";
    rosidl_generator_traits::value_to_yaml(msg.outdata, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DeleteName_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace gantry_lidar_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use gantry_lidar_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const gantry_lidar_interfaces::srv::DeleteName_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  gantry_lidar_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use gantry_lidar_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const gantry_lidar_interfaces::srv::DeleteName_Response & msg)
{
  return gantry_lidar_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<gantry_lidar_interfaces::srv::DeleteName_Response>()
{
  return "gantry_lidar_interfaces::srv::DeleteName_Response";
}

template<>
inline const char * name<gantry_lidar_interfaces::srv::DeleteName_Response>()
{
  return "gantry_lidar_interfaces/srv/DeleteName_Response";
}

template<>
struct has_fixed_size<gantry_lidar_interfaces::srv::DeleteName_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<gantry_lidar_interfaces::srv::DeleteName_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<gantry_lidar_interfaces::srv::DeleteName_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace gantry_lidar_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const DeleteName_Event & msg,
  std::ostream & out)
{
  out << "{";
  // member: info
  {
    out << "info: ";
    to_flow_style_yaml(msg.info, out);
    out << ", ";
  }

  // member: request
  {
    if (msg.request.size() == 0) {
      out << "request: []";
    } else {
      out << "request: [";
      size_t pending_items = msg.request.size();
      for (auto item : msg.request) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: response
  {
    if (msg.response.size() == 0) {
      out << "response: []";
    } else {
      out << "response: [";
      size_t pending_items = msg.response.size();
      for (auto item : msg.response) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const DeleteName_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: info
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "info:\n";
    to_block_style_yaml(msg.info, out, indentation + 2);
  }

  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.request.size() == 0) {
      out << "request: []\n";
    } else {
      out << "request:\n";
      for (auto item : msg.request) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.response.size() == 0) {
      out << "response: []\n";
    } else {
      out << "response:\n";
      for (auto item : msg.response) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const DeleteName_Event & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace gantry_lidar_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use gantry_lidar_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const gantry_lidar_interfaces::srv::DeleteName_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  gantry_lidar_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use gantry_lidar_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const gantry_lidar_interfaces::srv::DeleteName_Event & msg)
{
  return gantry_lidar_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<gantry_lidar_interfaces::srv::DeleteName_Event>()
{
  return "gantry_lidar_interfaces::srv::DeleteName_Event";
}

template<>
inline const char * name<gantry_lidar_interfaces::srv::DeleteName_Event>()
{
  return "gantry_lidar_interfaces/srv/DeleteName_Event";
}

template<>
struct has_fixed_size<gantry_lidar_interfaces::srv::DeleteName_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<gantry_lidar_interfaces::srv::DeleteName_Event>
  : std::integral_constant<bool, has_bounded_size<gantry_lidar_interfaces::srv::DeleteName_Request>::value && has_bounded_size<gantry_lidar_interfaces::srv::DeleteName_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<gantry_lidar_interfaces::srv::DeleteName_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<gantry_lidar_interfaces::srv::DeleteName>()
{
  return "gantry_lidar_interfaces::srv::DeleteName";
}

template<>
inline const char * name<gantry_lidar_interfaces::srv::DeleteName>()
{
  return "gantry_lidar_interfaces/srv/DeleteName";
}

template<>
struct has_fixed_size<gantry_lidar_interfaces::srv::DeleteName>
  : std::integral_constant<
    bool,
    has_fixed_size<gantry_lidar_interfaces::srv::DeleteName_Request>::value &&
    has_fixed_size<gantry_lidar_interfaces::srv::DeleteName_Response>::value
  >
{
};

template<>
struct has_bounded_size<gantry_lidar_interfaces::srv::DeleteName>
  : std::integral_constant<
    bool,
    has_bounded_size<gantry_lidar_interfaces::srv::DeleteName_Request>::value &&
    has_bounded_size<gantry_lidar_interfaces::srv::DeleteName_Response>::value
  >
{
};

template<>
struct is_service<gantry_lidar_interfaces::srv::DeleteName>
  : std::true_type
{
};

template<>
struct is_service_request<gantry_lidar_interfaces::srv::DeleteName_Request>
  : std::true_type
{
};

template<>
struct is_service_response<gantry_lidar_interfaces::srv::DeleteName_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DELETE_NAME__TRAITS_HPP_
