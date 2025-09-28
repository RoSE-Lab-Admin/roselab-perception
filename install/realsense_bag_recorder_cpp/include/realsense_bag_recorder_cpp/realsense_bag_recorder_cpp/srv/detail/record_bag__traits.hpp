// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from realsense_bag_recorder_cpp:srv/RecordBag.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "realsense_bag_recorder_cpp/srv/record_bag.hpp"


#ifndef REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__TRAITS_HPP_
#define REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "realsense_bag_recorder_cpp/srv/detail/record_bag__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace realsense_bag_recorder_cpp
{

namespace srv
{

inline void to_flow_style_yaml(
  const RecordBag_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: duration
  {
    out << "duration: ";
    rosidl_generator_traits::value_to_yaml(msg.duration, out);
    out << ", ";
  }

  // member: bag_name
  {
    out << "bag_name: ";
    rosidl_generator_traits::value_to_yaml(msg.bag_name, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RecordBag_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: duration
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "duration: ";
    rosidl_generator_traits::value_to_yaml(msg.duration, out);
    out << "\n";
  }

  // member: bag_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "bag_name: ";
    rosidl_generator_traits::value_to_yaml(msg.bag_name, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RecordBag_Request & msg, bool use_flow_style = false)
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

}  // namespace realsense_bag_recorder_cpp

namespace rosidl_generator_traits
{

[[deprecated("use realsense_bag_recorder_cpp::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const realsense_bag_recorder_cpp::srv::RecordBag_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  realsense_bag_recorder_cpp::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use realsense_bag_recorder_cpp::srv::to_yaml() instead")]]
inline std::string to_yaml(const realsense_bag_recorder_cpp::srv::RecordBag_Request & msg)
{
  return realsense_bag_recorder_cpp::srv::to_yaml(msg);
}

template<>
inline const char * data_type<realsense_bag_recorder_cpp::srv::RecordBag_Request>()
{
  return "realsense_bag_recorder_cpp::srv::RecordBag_Request";
}

template<>
inline const char * name<realsense_bag_recorder_cpp::srv::RecordBag_Request>()
{
  return "realsense_bag_recorder_cpp/srv/RecordBag_Request";
}

template<>
struct has_fixed_size<realsense_bag_recorder_cpp::srv::RecordBag_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<realsense_bag_recorder_cpp::srv::RecordBag_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<realsense_bag_recorder_cpp::srv::RecordBag_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace realsense_bag_recorder_cpp
{

namespace srv
{

inline void to_flow_style_yaml(
  const RecordBag_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RecordBag_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RecordBag_Response & msg, bool use_flow_style = false)
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

}  // namespace realsense_bag_recorder_cpp

namespace rosidl_generator_traits
{

[[deprecated("use realsense_bag_recorder_cpp::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const realsense_bag_recorder_cpp::srv::RecordBag_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  realsense_bag_recorder_cpp::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use realsense_bag_recorder_cpp::srv::to_yaml() instead")]]
inline std::string to_yaml(const realsense_bag_recorder_cpp::srv::RecordBag_Response & msg)
{
  return realsense_bag_recorder_cpp::srv::to_yaml(msg);
}

template<>
inline const char * data_type<realsense_bag_recorder_cpp::srv::RecordBag_Response>()
{
  return "realsense_bag_recorder_cpp::srv::RecordBag_Response";
}

template<>
inline const char * name<realsense_bag_recorder_cpp::srv::RecordBag_Response>()
{
  return "realsense_bag_recorder_cpp/srv/RecordBag_Response";
}

template<>
struct has_fixed_size<realsense_bag_recorder_cpp::srv::RecordBag_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<realsense_bag_recorder_cpp::srv::RecordBag_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<realsense_bag_recorder_cpp::srv::RecordBag_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'info'
#include "service_msgs/msg/detail/service_event_info__traits.hpp"

namespace realsense_bag_recorder_cpp
{

namespace srv
{

inline void to_flow_style_yaml(
  const RecordBag_Event & msg,
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
  const RecordBag_Event & msg,
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

inline std::string to_yaml(const RecordBag_Event & msg, bool use_flow_style = false)
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

}  // namespace realsense_bag_recorder_cpp

namespace rosidl_generator_traits
{

[[deprecated("use realsense_bag_recorder_cpp::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const realsense_bag_recorder_cpp::srv::RecordBag_Event & msg,
  std::ostream & out, size_t indentation = 0)
{
  realsense_bag_recorder_cpp::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use realsense_bag_recorder_cpp::srv::to_yaml() instead")]]
inline std::string to_yaml(const realsense_bag_recorder_cpp::srv::RecordBag_Event & msg)
{
  return realsense_bag_recorder_cpp::srv::to_yaml(msg);
}

template<>
inline const char * data_type<realsense_bag_recorder_cpp::srv::RecordBag_Event>()
{
  return "realsense_bag_recorder_cpp::srv::RecordBag_Event";
}

template<>
inline const char * name<realsense_bag_recorder_cpp::srv::RecordBag_Event>()
{
  return "realsense_bag_recorder_cpp/srv/RecordBag_Event";
}

template<>
struct has_fixed_size<realsense_bag_recorder_cpp::srv::RecordBag_Event>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<realsense_bag_recorder_cpp::srv::RecordBag_Event>
  : std::integral_constant<bool, has_bounded_size<realsense_bag_recorder_cpp::srv::RecordBag_Request>::value && has_bounded_size<realsense_bag_recorder_cpp::srv::RecordBag_Response>::value && has_bounded_size<service_msgs::msg::ServiceEventInfo>::value> {};

template<>
struct is_message<realsense_bag_recorder_cpp::srv::RecordBag_Event>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<realsense_bag_recorder_cpp::srv::RecordBag>()
{
  return "realsense_bag_recorder_cpp::srv::RecordBag";
}

template<>
inline const char * name<realsense_bag_recorder_cpp::srv::RecordBag>()
{
  return "realsense_bag_recorder_cpp/srv/RecordBag";
}

template<>
struct has_fixed_size<realsense_bag_recorder_cpp::srv::RecordBag>
  : std::integral_constant<
    bool,
    has_fixed_size<realsense_bag_recorder_cpp::srv::RecordBag_Request>::value &&
    has_fixed_size<realsense_bag_recorder_cpp::srv::RecordBag_Response>::value
  >
{
};

template<>
struct has_bounded_size<realsense_bag_recorder_cpp::srv::RecordBag>
  : std::integral_constant<
    bool,
    has_bounded_size<realsense_bag_recorder_cpp::srv::RecordBag_Request>::value &&
    has_bounded_size<realsense_bag_recorder_cpp::srv::RecordBag_Response>::value
  >
{
};

template<>
struct is_service<realsense_bag_recorder_cpp::srv::RecordBag>
  : std::true_type
{
};

template<>
struct is_service_request<realsense_bag_recorder_cpp::srv::RecordBag_Request>
  : std::true_type
{
};

template<>
struct is_service_response<realsense_bag_recorder_cpp::srv::RecordBag_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__TRAITS_HPP_
