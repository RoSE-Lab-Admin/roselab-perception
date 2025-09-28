// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from gantry_lidar_interfaces:srv/DownloadName.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "gantry_lidar_interfaces/srv/download_name.hpp"


#ifndef GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DOWNLOAD_NAME__BUILDER_HPP_
#define GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DOWNLOAD_NAME__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "gantry_lidar_interfaces/srv/detail/download_name__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace gantry_lidar_interfaces
{

namespace srv
{

namespace builder
{

class Init_DownloadName_Request_name
{
public:
  Init_DownloadName_Request_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::gantry_lidar_interfaces::srv::DownloadName_Request name(::gantry_lidar_interfaces::srv::DownloadName_Request::_name_type arg)
  {
    msg_.name = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadName_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gantry_lidar_interfaces::srv::DownloadName_Request>()
{
  return gantry_lidar_interfaces::srv::builder::Init_DownloadName_Request_name();
}

}  // namespace gantry_lidar_interfaces


namespace gantry_lidar_interfaces
{

namespace srv
{

namespace builder
{

class Init_DownloadName_Response_outdata
{
public:
  Init_DownloadName_Response_outdata()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::gantry_lidar_interfaces::srv::DownloadName_Response outdata(::gantry_lidar_interfaces::srv::DownloadName_Response::_outdata_type arg)
  {
    msg_.outdata = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadName_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gantry_lidar_interfaces::srv::DownloadName_Response>()
{
  return gantry_lidar_interfaces::srv::builder::Init_DownloadName_Response_outdata();
}

}  // namespace gantry_lidar_interfaces


namespace gantry_lidar_interfaces
{

namespace srv
{

namespace builder
{

class Init_DownloadName_Event_response
{
public:
  explicit Init_DownloadName_Event_response(::gantry_lidar_interfaces::srv::DownloadName_Event & msg)
  : msg_(msg)
  {}
  ::gantry_lidar_interfaces::srv::DownloadName_Event response(::gantry_lidar_interfaces::srv::DownloadName_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadName_Event msg_;
};

class Init_DownloadName_Event_request
{
public:
  explicit Init_DownloadName_Event_request(::gantry_lidar_interfaces::srv::DownloadName_Event & msg)
  : msg_(msg)
  {}
  Init_DownloadName_Event_response request(::gantry_lidar_interfaces::srv::DownloadName_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_DownloadName_Event_response(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadName_Event msg_;
};

class Init_DownloadName_Event_info
{
public:
  Init_DownloadName_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DownloadName_Event_request info(::gantry_lidar_interfaces::srv::DownloadName_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_DownloadName_Event_request(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadName_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gantry_lidar_interfaces::srv::DownloadName_Event>()
{
  return gantry_lidar_interfaces::srv::builder::Init_DownloadName_Event_info();
}

}  // namespace gantry_lidar_interfaces

#endif  // GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DOWNLOAD_NAME__BUILDER_HPP_
