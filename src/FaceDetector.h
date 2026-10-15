#pragma once
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

#include "Detector.h"

// High-speed face detection using OpenCV FaceDetectorYN (YuNet).
class FaceDetector {
public:
    FaceDetector();
    explicit FaceDetector(const std::string& modelPath, float scoreThresh = 0.5f, float nmsThresh = 0.3f);

    bool load(const std::string& modelPath, float scoreThresh = 0.5f, float nmsThresh = 0.3f);
    bool isLoaded() const { return static_cast<bool>(net_); }

    std::vector<Detection> detect(const cv::Mat& frame, float confThreshold, int classId = 0);

private:
    cv::Ptr<cv::FaceDetectorYN> net_;
    float scoreThresh_ = 0.5f;
    float nmsThresh_ = 0.3f;
    cv::Size lastSize_{0, 0};
};
