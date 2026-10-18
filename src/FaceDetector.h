#pragma once
#include <opencv2/objdetect.hpp>
#include <string>

#include "IDetector.h"

// Face detection with OpenCV's FaceDetectorYN (YuNet). Each detection carries
// the five face landmarks, which face recognition needs.
class FaceDetector : public IDetector {
public:
    FaceDetector(const std::string& modelPath, ClassId faceClass, DetectorThresholds thresholds);
    const char* name() const override { return "face"; }
    bool canDetect(ClassId id) const override { return id == faceClass_; }
    std::vector<Detection> detect(const cv::Mat& frame) override;

private:
    cv::Ptr<cv::FaceDetectorYN> net_;
    ClassId faceClass_;
    DetectorThresholds th_;
    cv::Size inputSize_{0, 0};
};
