// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from gantry_lidar_interfaces:srv/DownloadTimeRange.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "gantry_lidar_interfaces/srv/download_time_range.hpp"


#ifndef GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DOWNLOAD_TIME_RANGE__BUILDER_HPP_
#define GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DOWNLOAD_TIME_RANGE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "gantry_lidar_interfaces/srv/detail/download_time_range__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace gantry_lidar_interfaces
{

namespace srv
{

namespace builder
{

class Init_DownloadTimeRange_Request_end
{
public:
  explicit Init_DownloadTimeRange_Request_end(::gantry_lidar_interfaces::srv::DownloadTimeRange_Request & msg)
  : msg_(msg)
  {}
  ::gantry_lidar_interfaces::srv::DownloadTimeRange_Request end(::gantry_lidar_interfaces::srv::DownloadTimeRange_Request::_end_type arg)
  {
    msg_.end = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadTimeRange_Request msg_;
};

class Init_DownloadTimeRange_Request_start
{
public:
  Init_DownloadTimeRange_Request_start()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DownloadTimeRange_Request_end start(::gantry_lidar_interfaces::srv::DownloadTimeRange_Request::_start_type arg)
  {
    msg_.start = std::move(arg);
    return Init_DownloadTimeRange_Request_end(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadTimeRange_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gantry_lidar_interfaces::srv::DownloadTimeRange_Request>()
{
  return gantry_lidar_interfaces::srv::builder::Init_DownloadTimeRange_Request_start();
}

}  // namespace gantry_lidar_interfaces


namespace gantry_lidar_interfaces
{

namespace srv
{

namespace builder
{

class Init_DownloadTimeRange_Response_outdata
{
public:
  Init_DownloadTimeRange_Response_outdata()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::gantry_lidar_interfaces::srv::DownloadTimeRange_Response outdata(::gantry_lidar_interfaces::srv::DownloadTimeRange_Response::_outdata_type arg)
  {
    msg_.outdata = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadTimeRange_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gantry_lidar_interfaces::srv::DownloadTimeRange_Response>()
{
  return gantry_lidar_interfaces::srv::builder::Init_DownloadTimeRange_Response_outdata();
}

}  // namespace gantry_lidar_interfaces


namespace gantry_lidar_interfaces
{

namespace srv
{

namespace builder
{

class Init_DownloadTimeRange_Event_response
{
public:
  explicit Init_DownloadTimeRange_Event_response(::gantry_lidar_interfaces::srv::DownloadTimeRange_Event & msg)
  : msg_(msg)
  {}
  ::gantry_lidar_interfaces::srv::DownloadTimeRange_Event response(::gantry_lidar_interfaces::srv::DownloadTimeRange_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadTimeRange_Event msg_;
};

class Init_DownloadTimeRange_Event_request
{
public:
  explicit Init_DownloadTimeRange_Event_request(::gantry_lidar_interfaces::srv::DownloadTimeRange_Event & msg)
  : msg_(msg)
  {}
  Init_DownloadTimeRange_Event_response request(::gantry_lidar_interfaces::srv::DownloadTimeRange_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_DownloadTimeRange_Event_response(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadTimeRange_Event msg_;
};

class Init_DownloadTimeRange_Event_info
{
public:
  Init_DownloadTimeRange_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DownloadTimeRange_Event_request info(::gantry_lidar_interfaces::srv::DownloadTimeRange_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_DownloadTimeRange_Event_request(msg_);
  }

private:
  ::gantry_lidar_interfaces::srv::DownloadTimeRange_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::gantry_lidar_interfaces::srv::DownloadTimeRange_Event>()
{
  return gantry_lidar_interfaces::srv::builder::Init_DownloadTimeRange_Event_info();
}

}  // namespace gantry_lidar_interfaces

#endif  // GANTRY_LIDAR_INTERFACES__SRV__DETAIL__DOWNLOAD_TIME_RANGE__BUILDER_HPP_
