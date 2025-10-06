// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from gantry_lidar_interfaces:srv/Capture.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "gantry_lidar_interfaces/srv/capture.hpp"


#ifndef GANTRY_LIDAR_INTERFACES__SRV__DETAIL__CAPTURE__BUILDER_HPP_
#define GANTRY_LIDAR_INTERFACES__SRV__DETAIL__CAPTURE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "gantry_lidar_interfaces/srv/detail/capture__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace gantry_lidar_interfaces
{

namespace srv
{

namespace builder
{

class Init_Capture_Request_duration
{
public:
  explicit Init_Capture_Request_duration(::gantry_lidar_interfaces::srv::Capture_Request & msg)
  : msg_(msg)
  {}
  ::gantry_lidar_interfaces::srv::Capture_Request duration(::gantry_lidar_interfaces::srv::Capture_Request::_duration_type arg)
  {
    msg_.duration = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::Capture_Request msg_;
};

class Init_Capture_Request_sensors
{
public:
  explicit Init_Capture_Request_sensors(::gantry_lidar_interfaces::srv::Capture_Request & msg)
  : msg_(msg)
  {}
  Init_Capture_Request_duration sensors(::gantry_lidar_interfaces::srv::Capture_Request::_sensors_type arg)
  {
    msg_.sensors = std::move(arg);
    return Init_Capture_Request_duration(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::Capture_Request msg_;
};

class Init_Capture_Request_outname
{
public:
  Init_Capture_Request_outname()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Capture_Request_sensors outname(::gantry_lidar_interfaces::srv::Capture_Request::_outname_type arg)
  {
    msg_.outname = std::move(arg);
    return Init_Capture_Request_sensors(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::Capture_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gantry_lidar_interfaces::srv::Capture_Request>()
{
  return gantry_lidar_interfaces::srv::builder::Init_Capture_Request_outname();
}

}  // namespace gantry_lidar_interfaces


namespace gantry_lidar_interfaces
{

namespace srv
{

namespace builder
{

class Init_Capture_Response_outdata
{
public:
  Init_Capture_Response_outdata()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::gantry_lidar_interfaces::srv::Capture_Response outdata(::gantry_lidar_interfaces::srv::Capture_Response::_outdata_type arg)
  {
    msg_.outdata = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::Capture_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gantry_lidar_interfaces::srv::Capture_Response>()
{
  return gantry_lidar_interfaces::srv::builder::Init_Capture_Response_outdata();
}

}  // namespace gantry_lidar_interfaces


namespace gantry_lidar_interfaces
{

namespace srv
{

namespace builder
{

class Init_Capture_Event_response
{
public:
  explicit Init_Capture_Event_response(::gantry_lidar_interfaces::srv::Capture_Event & msg)
  : msg_(msg)
  {}
  ::gantry_lidar_interfaces::srv::Capture_Event response(::gantry_lidar_interfaces::srv::Capture_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::Capture_Event msg_;
};

class Init_Capture_Event_request
{
public:
  explicit Init_Capture_Event_request(::gantry_lidar_interfaces::srv::Capture_Event & msg)
  : msg_(msg)
  {}
  Init_Capture_Event_response request(::gantry_lidar_interfaces::srv::Capture_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_Capture_Event_response(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::Capture_Event msg_;
};

class Init_Capture_Event_info
{
public:
  Init_Capture_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Capture_Event_request info(::gantry_lidar_interfaces::srv::Capture_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_Capture_Event_request(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::Capture_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gantry_lidar_interfaces::srv::Capture_Event>()
{
  return gantry_lidar_interfaces::srv::builder::Init_Capture_Event_info();
}

}  // namespace gantry_lidar_interfaces

#endif  // GANTRY_LIDAR_INTERFACES__SRV__DETAIL__CAPTURE__BUILDER_HPP_
