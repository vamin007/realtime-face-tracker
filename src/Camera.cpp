#include "Camera.h"

bool Camera::openDevice(int index, int width, int height) {
    // AVFoundation is the native macOS camera API.
    if (!cap_.open(index, cv::CAP_AVFOUNDATION)) return false;
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    live_ = true;
    return true;
}

bool Camera::openFile(const std::string& path) {
    live_ = false;
    return cap_.open(path);
}

bool Camera::read(cv::Mat& frame) {
    return cap_.read(frame) && !frame.empty();
}
