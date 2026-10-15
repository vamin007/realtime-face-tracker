#include "FaceDetector.h"

#include <iostream>

FaceDetector::FaceDetector() = default;

FaceDetector::FaceDetector(const std::string& modelPath, float scoreThresh, float nmsThresh) {
    load(modelPath, scoreThresh, nmsThresh);
}

bool FaceDetector::load(const std::string& modelPath, float scoreThresh, float nmsThresh) {
    scoreThresh_ = scoreThresh;
    nmsThresh_ = nmsThresh;
    try {
        net_ = cv::FaceDetectorYN::create(modelPath, "", cv::Size(320, 320), scoreThresh_, nmsThresh_);
        return static_cast<bool>(net_);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to initialize FaceDetectorYN from " << modelPath << ": " << e.what() << '\n';
        net_.release();
        return false;
    }
}

std::vector<Detection> FaceDetector::detect(const cv::Mat& frame, float confThreshold, int classId) {
    std::vector<Detection> detections;
    if (!net_ || frame.empty()) return detections;

    if (frame.size() != lastSize_) {
        net_->setInputSize(frame.size());
        lastSize_ = frame.size();
    }
    net_->setScoreThreshold(confThreshold > 0.0f ? confThreshold : scoreThresh_);

    cv::Mat faces;
    net_->detect(frame, faces);

    const cv::Rect2f bounds(0, 0, static_cast<float>(frame.cols), static_cast<float>(frame.rows));
    for (int i = 0; i < faces.rows; ++i) {
        const float x = faces.at<float>(i, 0);
        const float y = faces.at<float>(i, 1);
        const float w = faces.at<float>(i, 2);
        const float h = faces.at<float>(i, 3);
        const float conf = faces.at<float>(i, 14);

        if (conf < confThreshold) continue;
        cv::Rect2f box = cv::Rect2f(x, y, w, h) & bounds;
        if (box.area() <= 0) continue;

        detections.push_back({box, classId, conf});
    }
    return detections;
}
