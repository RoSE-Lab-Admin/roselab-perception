// generated from rosidl_generator_c/resource/rosidl_generator_c__visibility_control.h.in
// generated code does not contain a copyright notice

#ifndef REALSENSE_BAG_RECORDER_CPP__MSG__ROSIDL_GENERATOR_C__VISIBILITY_CONTROL_H_
#define REALSENSE_BAG_RECORDER_CPP__MSG__ROSIDL_GENERATOR_C__VISIBILITY_CONTROL_H_

#ifdef __cplusplus
extern "C"
{
#endif

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define ROSIDL_GENERATOR_C_EXPORT_realsense_bag_recorder_cpp __attribute__ ((dllexport))
    #define ROSIDL_GENERATOR_C_IMPORT_realsense_bag_recorder_cpp __attribute__ ((dllimport))
  #else
    #define ROSIDL_GENERATOR_C_EXPORT_realsense_bag_recorder_cpp __declspec(dllexport)
    #define ROSIDL_GENERATOR_C_IMPORT_realsense_bag_recorder_cpp __declspec(dllimport)
  #endif
  #ifdef ROSIDL_GENERATOR_C_BUILDING_DLL_realsense_bag_recorder_cpp
    #define ROSIDL_GENERATOR_C_PUBLIC_realsense_bag_recorder_cpp ROSIDL_GENERATOR_C_EXPORT_realsense_bag_recorder_cpp
  #else
    #define ROSIDL_GENERATOR_C_PUBLIC_realsense_bag_recorder_cpp ROSIDL_GENERATOR_C_IMPORT_realsense_bag_recorder_cpp
  #endif
#else
  #define ROSIDL_GENERATOR_C_EXPORT_realsense_bag_recorder_cpp __attribute__ ((visibility("default")))
  #define ROSIDL_GENERATOR_C_IMPORT_realsense_bag_recorder_cpp
  #if __GNUC__ >= 4
    #define ROSIDL_GENERATOR_C_PUBLIC_realsense_bag_recorder_cpp __attribute__ ((visibility("default")))
  #else
    #define ROSIDL_GENERATOR_C_PUBLIC_realsense_bag_recorder_cpp
  #endif
#endif

#ifdef __cplusplus
}
#endif

#endif  // REALSENSE_BAG_RECORDER_CPP__MSG__ROSIDL_GENERATOR_C__VISIBILITY_CONTROL_H_
