#include <librealsense2/rs.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char * argv[]) try
{
    rs2::pipeline pipe;

    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);

    pipe.start(cfg);

    // Time tracking for FPS
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    for (int i = 0; i < 100; ++i) 
    {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::depth_frame depth = frames.get_depth_frame();
        rs2::video_frame color = frames.get_color_frame();

        int depth_width = depth.get_width();
        int depth_height = depth.get_height();
        float center_distance = depth.get_distance(depth_width / 2, depth_height / 2);

        int color_width = color.get_width();
        int color_height = color.get_height();

        frame_count++;

        // Calculate elapsed time every frame
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start_time;
        double fps = frame_count / elapsed.count();

        std::cout << "Frame " << frame_count
                  << ": Depth: " << depth_width << "x" << depth_height
                  << ", Color: " << color_width << "x" << color_height
                  << ", Approx. FPS: " << fps
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
