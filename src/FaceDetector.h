#pragma once
#include <opencv2/objdetect.hpp>
#include <string>

#include "IDetector.h"

// Face detection with OpenCV's FaceDetectorYN (YuNet). Each detection carries
// the five face landmarks, which face recognition needs.
class FaceDetector : public IDetector {
public:
    // maxWidth: frames wider than this are scaled down first (faster; faces near
    // the camera are large anyway). 0 keeps the full frame.
    FaceDetector(const std::string& modelPath, ClassId faceClass, DetectorThresholds thresholds,
                 int maxWidth = 640);
    const char* name() const override { return "face"; }
    bool canDetect(ClassId id) const override { return id == faceClass_; }
    std::vector<Detection> detect(const cv::Mat& frame) override;
    std::string info() const override;

private:
    cv::Ptr<cv::FaceDetectorYN> net_;
    ClassId faceClass_;
    DetectorThresholds th_;
    int maxWidth_;
    cv::Size inputSize_{0, 0};
};
