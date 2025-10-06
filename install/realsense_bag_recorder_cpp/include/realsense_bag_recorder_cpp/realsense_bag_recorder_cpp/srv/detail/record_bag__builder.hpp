// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from realsense_bag_recorder_cpp:srv/RecordBag.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "realsense_bag_recorder_cpp/srv/record_bag.hpp"


#ifndef REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__BUILDER_HPP_
#define REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "realsense_bag_recorder_cpp/srv/detail/record_bag__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace realsense_bag_recorder_cpp
{

namespace srv
{

namespace builder
{

class Init_RecordBag_Request_bag_name
{
public:
  explicit Init_RecordBag_Request_bag_name(::realsense_bag_recorder_cpp::srv::RecordBag_Request & msg)
  : msg_(msg)
  {}
  ::realsense_bag_recorder_cpp::srv::RecordBag_Request bag_name(::realsense_bag_recorder_cpp::srv::RecordBag_Request::_bag_name_type arg)
  {
    msg_.bag_name = std::move(arg);
    return std::move(msg_);
  }

private:
  ::realsense_bag_recorder_cpp::srv::RecordBag_Request msg_;
};

class Init_RecordBag_Request_duration
{
public:
  Init_RecordBag_Request_duration()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RecordBag_Request_bag_name duration(::realsense_bag_recorder_cpp::srv::RecordBag_Request::_duration_type arg)
  {
    msg_.duration = std::move(arg);
    return Init_RecordBag_Request_bag_name(msg_);
  }

private:
  ::realsense_bag_recorder_cpp::srv::RecordBag_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::realsense_bag_recorder_cpp::srv::RecordBag_Request>()
{
  return realsense_bag_recorder_cpp::srv::builder::Init_RecordBag_Request_duration();
}

}  // namespace realsense_bag_recorder_cpp


namespace realsense_bag_recorder_cpp
{

namespace srv
{

namespace builder
{

class Init_RecordBag_Response_message
{
public:
  explicit Init_RecordBag_Response_message(::realsense_bag_recorder_cpp::srv::RecordBag_Response & msg)
  : msg_(msg)
  {}
  ::realsense_bag_recorder_cpp::srv::RecordBag_Response message(::realsense_bag_recorder_cpp::srv::RecordBag_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::realsense_bag_recorder_cpp::srv::RecordBag_Response msg_;
};

class Init_RecordBag_Response_success
{
public:
  Init_RecordBag_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RecordBag_Response_message success(::realsense_bag_recorder_cpp::srv::RecordBag_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_RecordBag_Response_message(msg_);
  }

private:
  ::realsense_bag_recorder_cpp::srv::RecordBag_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::realsense_bag_recorder_cpp::srv::RecordBag_Response>()
{
  return realsense_bag_recorder_cpp::srv::builder::Init_RecordBag_Response_success();
}

}  // namespace realsense_bag_recorder_cpp


namespace realsense_bag_recorder_cpp
{

namespace srv
{

namespace builder
{

class Init_RecordBag_Event_response
{
public:
  explicit Init_RecordBag_Event_response(::realsense_bag_recorder_cpp::srv::RecordBag_Event & msg)
  : msg_(msg)
  {}
  ::realsense_bag_recorder_cpp::srv::RecordBag_Event response(::realsense_bag_recorder_cpp::srv::RecordBag_Event::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::realsense_bag_recorder_cpp::srv::RecordBag_Event msg_;
};

class Init_RecordBag_Event_request
{
public:
  explicit Init_RecordBag_Event_request(::realsense_bag_recorder_cpp::srv::RecordBag_Event & msg)
  : msg_(msg)
  {}
  Init_RecordBag_Event_response request(::realsense_bag_recorder_cpp::srv::RecordBag_Event::_request_type arg)
  {
    msg_.request = std::move(arg);
    return Init_RecordBag_Event_response(msg_);
  }

private:
  ::realsense_bag_recorder_cpp::srv::RecordBag_Event msg_;
};

class Init_RecordBag_Event_info
{
public:
  Init_RecordBag_Event_info()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RecordBag_Event_request info(::realsense_bag_recorder_cpp::srv::RecordBag_Event::_info_type arg)
  {
    msg_.info = std::move(arg);
    return Init_RecordBag_Event_request(msg_);
  }

private:
  ::realsense_bag_recorder_cpp::srv::RecordBag_Event msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::realsense_bag_recorder_cpp::srv::RecordBag_Event>()
{
  return realsense_bag_recorder_cpp::srv::builder::Init_RecordBag_Event_info();
}

}  // namespace realsense_bag_recorder_cpp

#endif  // REALSENSE_BAG_RECORDER_CPP__SRV__DETAIL__RECORD_BAG__BUILDER_HPP_
