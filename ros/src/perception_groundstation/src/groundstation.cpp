#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <chrono>
#include <string>
#include <memory>
#include <iostream>

using namespace std::chrono_literals;

class TestConsole : public rclcpp::Node {
public:
  int trial_index;
  bool bags_running = false;
  std::string path;
  double start_time = 0.0; // set in main

  writer_ = std::make_unique<rosbag2_cpp::Writer>();

  // TODO Service clients
  // Client for mastcam

  TestConsole(int trial) : Node("perception_groundstation"), trial_index(trial),
  {
    // Endpoints for mastcam?

    // create mastcam client (rclcpp_action::create_client<MastCam>();)
    
    // Trial Name: e.g., /lidar_scans/T####/
    char buf[64];
    std::snprintf(buf, sizeof(buf), "T%04d", trial_index);
    path = std::string("/lidar_scans/") + buf;
  }

  /// @brief Starts just the local bagging for non-mastcam rover topics
  void start_bags() {
    writer_->open("trial_test"); //TODO better bag name
    topics = [
      // All the junk from cube rover
    ]
    // Use service for mastcam
  }

  /// @brief Stops local bags
  void stop_bags() {
    // stop it
  }

  /// @brief Starts a full lidar scan. Blocks until it finishes
  void start_lidar_block_until_accepted() {
    // Start Lidar Scan Node
    lidar_ = std::make_shared<WorkerNode>();
    lidar_exec_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    lidar_exec_->add_node(lidar_);

    // Spin the lidar scan in a background thread
    lidar_thread_ = std::thread([this]() {
      lidar_exec_->spin();
    });
  }

  /// @brief Starts rover teleop node. Non blocking
  void start_rover() {
    // Run rover teleop node
    rover_ = std::make_shared<WorkerNode>();
    rover_exec_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    rover_exec_->add_node(rover_);

    // Spin the rover in a background thread
    rover_thread_ = std::thread([this]() {
      rover_exec_->spin();
    });
  }

private:
  
};

int main(int argc, char **argv) {
  std::cout << "------------Perception Teleop Ground Station------------\n";
  std::cout << "Ensure the following: \n 1) LiDARs are launched on gantry computer.\n2) LiDAR service is running on gantry computer.\n3) MastCam initalize script has been run.\n4)MastCam service is started.\n5)MOCAP IS DISABLED!!!"
  // Startup node
  rclcpp::init(argc, argv);
    auto node = std::make_shared<TestConsole>(i);
    
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  while (true) {
    std::cout << "PLEASE RE-ENABLE MOCAP";
    // --- Operator inputs to start test ---
    std::cout << "Test number: ";
    int test_num{};
    std::cin >> test_num;
    std::cout << "Press ENTER to begin first LiDAR scan...";
    std::string dummy;
    std::getline(std::cin, dummy);

    try {
      // ─── Stage 1: Initial LiDAR ───────
      auto t0 = std::chrono::steady_clock::now();
      node->get_logger().info("=== Stage 1: First LiDAR Scan ===");

      node->start_bags();
      // Wait for bag start completion?
      if (rclcpp::spin_until_future_complete(node, node->pending_bag_start()) == rclcpp::FutureReturnCode::SUCCESS) {
        node->bags_running = true;
      }

      node->start_time = node->now().seconds();

      // Start LiDAR (wait for acceptance), then wait for result
      auto lidar_gh1 = node->start_lidar_block_until_accepted();
      if (!lidar_gh1) throw std::runtime_error("LiDAR start rejected (stage 1).");

      // sleep/pad while scan runS
      // std::this_thread::sleep_for(std::chrono::seconds(DURATION)); // TODO if needed

      auto result_future_1 = node->lidar_client_->async_get_result(lidar_gh1);
      if (rclcpp::spin_until_future_complete(node, result_future_1)
          != rclcpp::FutureReturnCode::SUCCESS) {
        throw std::runtime_error("Error waiting for LiDAR result (stage 1).");
      }
      auto wrapped_1 = result_future_1.get();

      // DOWNLOAD LIDAR
      node->download_lidar(); 

      auto t1 = std::chrono::steady_clock::now();
      auto dt1 = std::chrono::duration<double>(t1 - t0).count();
      node->get_logger().info((std::string("--- Stage 1 duration: ") + std::to_string(dt1) + "s ---").c_str());

      // ─── Stage 2: Teleop ────
      node->get_logger().info("=== Stage 2: Rover Teleop ===");

      std::cout << "Press ENTER to enable teleop...";
      std::string dummy;
      std::getline(std::cin, dummy);

      node->start_rover();
      node->start_bags();
      (void)rclcpp::spin_until_future_complete(node, node->pending_bag_start());
      node->bags_running = true;
      node->start_mastcam();

      std::cout << "Teleop eneabled, use controller to drive RoSEY...\n";
      
      std::cout << "Press ENTER to end teleop...";
      std::string dummy;
      std::getline(std::cin, dummy);

      t1 = std::chrono::steady_clock::now();
      auto dt2 = std::chrono::duration<double>(t1 - t0).count();
      node->get_logger().info((std::string("--- Stage 2 duration: ") + std::to_string(dt2) + "s ---").c_str());


      // ─── Stage 3: Final LiDAR ────────────────────────────────────────────────
      t0 = std::chrono::steady_clock::now();
      node->get_logger().info("=== Stage 3: End of Trial LiDAR Scan ===");

      auto lidar_gh2 = node->start_lidar_block_until_accepted();
      if (!lidar_gh2) throw std::runtime_error("LiDAR goal rejected (stage 3).");

      // needed?
      // std::this_thread::sleep_for(2s);

      auto result_future_2 = node->lidar_client_->async_get_result(lidar_gh2);
      if (rclcpp::spin_until_future_complete(node, result_future_2)
          != rclcpp::FutureReturnCode::SUCCESS) {
        throw std::runtime_error("Error waiting for LiDAR result (stage 3).");
      }

      // Stop bags
      node->stop_bags();
      (void)rclcpp::spin_until_future_complete(node, node->pending_bag_stop());
      node->bags_running = false;

      // // Post-process data for this trial
      // data_process(node->path);

      // Download second scan
      node->download_lidar();

      t1 = std::chrono::steady_clock::now();
      auto dt5 = std::chrono::duration<double>(t1 - t0).count();
      node->get_logger().info((std::string("--- Stage 3 duration: ") + std::to_string(dt5) + "s ---").c_str());

      // Optional grace pause
      // std::this_thread::sleep_for(10s);

      node->delete_lidar();

    } catch (const std::exception &e) {
      node->get_logger().error((std::string("Trial aborted due to error: ") + e.what()).c_str());
    }

    // --- Finally/cleanup ---
    if (node->bags_running) {
      node->get_logger().info("Cleaning up: stopping residual bag recording…");
      try {
        node->stop_bags();
        (void)rclcpp::spin_until_future_complete(node, node->pending_bag_stop());
      } catch (...) {

      }
      node->bags_running = false;
    }

    node->get_logger().info((std::string("Trial complete. Bags saved to ") + node->path).c_str());
    std::cout<<"\n Continue? (y/n): ";
    std::string cont;
    std::getline(std::cin, cont);
    if(cont=="n"){
      break;
    }
    
  }
  node->destroy_node();
  rclcpp::shutdown();
  return 0;
}
