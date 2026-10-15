#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

#include "Detector.h"

// High-speed hand/palm detection using MediaPipe Palm Detection via OpenCV DNN.
class HandDetector {
public:
    HandDetector();
    explicit HandDetector(const std::string& modelPath, float scoreThresh = 0.5f, float nmsThresh = 0.3f);

    bool load(const std::string& modelPath, float scoreThresh = 0.5f, float nmsThresh = 0.3f);
    bool isLoaded() const { return !net_.empty(); }

    std::vector<Detection> detect(const cv::Mat& frame, float confThreshold, int classId);

private:
    static std::vector<cv::Point2f> generateAnchors();

    cv::dnn::Net net_;
    float scoreThresh_ = 0.5f;
    float nmsThresh_ = 0.3f;
    std::vector<cv::Point2f> anchors_;
};
