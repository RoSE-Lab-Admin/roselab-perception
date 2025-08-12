#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <thread>
#include <fstream>
#include <sstream>

#include "rclcpp/rclcpp.hpp"
#include "realsense_bag_recorder_cpp/srv/record_bag.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;
using std::placeholders::_2;

class BagRecorderService : public rclcpp::Node
{
public:
  using RecordBag = realsense_bag_recorder_cpp::srv::RecordBag;

  BagRecorderService()
  : Node("bag_recorder_service")
  {
    this->declare_parameter<std::string>("bag_directory", std::string(std::getenv("HOME")) + "/ROSELAB_BAGS");
    this->declare_parameter<bool>("wait_for_topics", true);

    service_ = this->create_service<RecordBag>(
      "trigger_bag_recording",
      std::bind(&BagRecorderService::handle_request, this, _1, _2)
    );

    RCLCPP_INFO(this->get_logger(), "Custom service ready: trigger_bag_recording");
  }

private:
  rclcpp::Service<RecordBag>::SharedPtr service_;

  const std::vector<std::string> topics_ = {
    "/camera/color/image_raw",
    "/camera/depth/image_rect_raw",
    "/camera/color/camera_info",
    "/camera/depth/camera_info",
    "/camera/accel/sample",
    "/camera/gyro/sample"
  };
  // May want to add the camera extrinsics here too

  void handle_request(
    const std::shared_ptr<RecordBag::Request> request,
    std::shared_ptr<RecordBag::Response> response)
  {
    double duration = request->duration;
    std::string bag_name = request->bag_name;

    if (duration <= 0.0) {
      response->success = false;
      response->message = "Invalid duration. Must be > 0.";
      RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
      return;
    }

    std::string bag_directory;
    bool wait_for_topics;
    this->get_parameter("bag_directory", bag_directory);
    this->get_parameter("wait_for_topics", wait_for_topics);

    if (wait_for_topics) {
      for (const auto& topic : topics_) {
        if (!this->wait_for_topic(topic, 5s)) {
          response->success = false;
          response->message = "Topic timeout: " + topic;
          RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
          return;
        }
      }
    }

    if (bag_name.empty()) {
      bag_name = "bag_" + std::to_string(this->now().seconds());
    }
    else {
      bag_name +=  "_" + std::to_string(this->now().seconds());
    }

    std::string bag_path = bag_directory + "/" + bag_name;
    std::filesystem::create_directories(bag_directory);

    std::stringstream cmd;
    cmd << "ros2 bag record -o " << bag_path << "-d " << duration;
    for (const auto& topic : topics_) {
      cmd << " " << topic;
    }

    std::string full_cmd = cmd.str(); //+ " & echo $! > /tmp/bag_pid.txt";
    int result = std::system(full_cmd.c_str());

    if (result != 0) {
      response->success = false;
      response->message = "Failed to start rosbag process.";
      return;
    }

//    RCLCPP_INFO(this->get_logger(), "Recording for %.1f seconds...", duration);
//    std::this_thread::sleep_for(std::chrono::duration<double>(duration));
//
//    kill_rosbag_process();

    response->success = true;
    response->message = "Recorded bag at: " + bag_path;
    RCLCPP_INFO(this->get_logger(), "Recording complete: %s", bag_path.c_str());
  }
//
//  void kill_rosbag_process()
//  {
//    std::ifstream pid_file("/tmp/bag_pid.txt");
//    int pid = 0;
//    pid_file >> pid;
//    if (pid > 0) {
//      std::stringstream kill_cmd;
//      kill_cmd << "kill -2 " << pid;
//      std::system(kill_cmd.str().c_str());
//    }
//  }

  bool wait_for_topic(const std::string& topic, std::chrono::seconds timeout)
  {
    auto start = this->now();
    while ((this->now() - start).seconds() < timeout.count()) {
      auto topics_and_types = this->get_topic_names_and_types();
      if (topics_and_types.count(topic)) {
        return true;
      }
      std::this_thread::sleep_for(500ms);
    }
    return false;
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BagRecorderService>());
  rclcpp::shutdown();
  return 0;
}
