#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>

// Frames from the camera or a video file.
// A camera is read on a background thread that always keeps the newest frame,
// so slow processing skips stale frames instead of falling behind. A video file
// is read frame by frame, or, with realtime playback, paced like a camera.
class Camera {
public:
    Camera() = default;
    ~Camera();
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    bool openDevice(int index, int width, int height);
    bool openFile(const std::string& path, bool realtime = false);

    // Next frame and its time in ms (camera: since opening; file: the frame's
    // position in the video). Returns false at the end of a file.
    bool read(cv::Mat& frame, double& timeMs);

    bool isLive() const { return live_; }
    cv::Size frameSize() const { return size_; }
    double fps() const { return fps_; }
    long long dropped() const;  // frames skipped because processing was still busy

private:
    void captureLoop();
    bool grab(cv::Mat& frame, double& timeMs);

    cv::VideoCapture cap_;
    bool live_ = false;
    bool threaded_ = false;
    double fps_ = 30.0;
    cv::Size size_;
    long long frameCount_ = 0;
    std::chrono::steady_clock::time_point start_;

    std::thread thread_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    cv::Mat latest_;
    double latestMs_ = 0;
    bool fresh_ = false;  // latest_ hasn't been handed out yet
    bool ended_ = false;
    bool stop_ = false;
    long long dropped_ = 0;
};
