// --- Add these includes at the top (POSIX-only). If you need Windows, wrap with #ifdefs and use CreateProcess. ---
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

// --- Add these small helpers to your TestConsole class (no threads, subprocess-based) ---
private:
  // Spawn a child running `sh -c <cmd>`. Returns child's PID or -1 on error.
  pid_t spawn_shell(const std::string& cmd) {
    pid_t pid = fork();
    if (pid == -1) {
      RCLCPP_ERROR(get_logger(), "fork() failed for: %s", cmd.c_str());
      return -1;
    }
    if (pid == 0) {
      // Child: create a new process group so we can signal the whole tree with kill(-pgid, SIGINT)
      setpgid(0, 0);
      execlp("sh", "sh", "-c", cmd.c_str(), (char*)nullptr);
      _exit(127); // exec failed
    }
    // Parent: place child in its own process group
    setpgid(pid, pid);
    return pid;
  }

  // Send SIGINT to a process group (created by spawn_shell). Then wait for it to exit.
  // If it doesn't exit, escalate to SIGTERM, then SIGKILL.
  void stop_proc_group_and_wait(pid_t pid, int timeout_ms = 5000) {
    if (pid <= 0) return;
    const pid_t pgid = pid; // we set pgid == pid
    // Try SIGINT (graceful)
    kill(-pgid, SIGINT);

    // Poll for exit up to timeout
    const int step_ms = 100;
    int waited = 0;
    while (waited < timeout_ms) {
      int status = 0;
      pid_t r = waitpid(pid, &status, WNOHANG);
      if (r == pid) return; // exited
      usleep(step_ms * 1000);
      waited += step_ms;
    }

    // Escalate to SIGTERM
    kill(-pgid, SIGTERM);
    waited = 0;
    while (waited < timeout_ms) {
      int status = 0;
      pid_t r = waitpid(pid, &status, WNOHANG);
      if (r == pid) return; // exited
      usleep(step_ms * 1000);
      waited += step_ms;
    }

    // Final: SIGKILL
    kill(-pgid, SIGKILL);
    (void)waitpid(pid, nullptr, 0);
  }

public:
  // Start a non-blocking ros2 bag recorder. Returns PID. No threads used.
  pid_t start_bag_async(const std::vector<std::string>& topics, const std::string& out_dir) {
    std::string cmd = "ros2 bag record -o " + out_dir;
    for (const auto& t : topics) cmd += " " + t;
    RCLCPP_INFO(get_logger(), "Bagging: %s", cmd.c_str());
    return spawn_shell(cmd);
  }

  // Start teleop as a separate process. Returns PID.
  // Replace with your actual teleop launch or node command.
  pid_t start_teleop_async() {
    // Example options:
    // 1) ros2 launch my_robot_bringup teleop.launch.py
    // 2) ros2 run my_robot_teleop teleop_node
    const std::string cmd = "ros2 run my_robot_teleop teleop_node";
    RCLCPP_INFO(get_logger(), "Starting teleop: %s", cmd.c_str());
    return spawn_shell(cmd);
  }

// --- In main(), inside the Stage 2 block, replace the prior placeholder with the following: ---
{
  RCLCPP_INFO(node->get_logger(), "Stage 2 Teleop");

  std::cout << "Press ENTER to start teleop segment (will start MastCam capture + bagging first)...";
  std::string dummy;
  std::getline(std::cin, dummy);

  // 1) Start MastCam capture (service). Runs during teleop.
  //    If your capture requires an explicit stop and you have no stop srv,
  //    set a large duration and it will end on its own; otherwise call the stop srv here when ending.
  const std::string mast_name = "test_" + std::to_string(test_num) + "_teleop_cam";
  node->mastcam_capture(mast_name, /*duration_sec=*/3600.0f); // pseudo "run until we stop teleop"

  // 2) Start other bagging (non-blocking). Example topics; adjust as needed.
  std::vector<std::string> topics_general = {
    "/tf",
    "/tf_static",
    "/imu/data",
    "/odometry/filtered",
    "/wheel_states"
    // add rover control topics you want recorded during teleop
  };
  const std::string bag_dir_general = node->trial_dir + "/stage2_general";
  pid_t bag_general_pid = node->start_bag_async(topics_general, bag_dir_general);

  // 3) Optionally, start a separate MastCam bag for camera topics (if you want raw streams alongside service capture).
  //    If you don't need this, skip and rely on the capture service outputs.
  std::vector<std::string> topics_mastcam = {
    "/mastcam/image_raw",
    "/mastcam/camera_info"
    // add rectified or compressed topics if needed
  };
  const std::string bag_dir_mast = node->trial_dir + "/stage2_mastcam";
  pid_t bag_mast_pid = node->start_bag_async(topics_mastcam, bag_dir_mast);

  // 4) Start teleop process (non-blocking).
  pid_t teleop_pid = node->start_teleop_async();

  std::cout << "Teleop running. Use controller to drive.\n";
  std::cout << "Press ENTER to END teleop (will stop teleop, then MastCam and other bagging)...";
  std::getline(std::cin, dummy);

  // 5) Stop teleop first (graceful).
  node->stop_proc_group_and_wait(teleop_pid);

  // 6) Stop MastCam bagging and other bagging (graceful). Order: MastCam bag first, then general.
  node->stop_proc_group_and_wait(bag_mast_pid);
  node->stop_proc_group_and_wait(bag_general_pid);

  // 7) If MastCam capture needs explicit end/trim, do it now.
  //    With the provided srv, there is no explicit "stop"; capture ends by duration.
  //    If you have a stop srv, call it here. Then download results.
  node->mastcam_download(mast_name);
}
