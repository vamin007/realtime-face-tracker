#pragma once
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>

// Thin wrapper around cv::VideoCapture: either a live camera or a video file.
class Camera {
public:
    bool openDevice(int index, int width, int height);
    bool openFile(const std::string& path);
    bool read(cv::Mat& frame);
    bool isLive() const { return live_; }

private:
    cv::VideoCapture cap_;
    bool live_ = false;
};
