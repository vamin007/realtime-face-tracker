#pragma once
#include <opencv2/dnn.hpp>
#include <string>

#include "IDetector.h"

// Hand detection with MediaPipe's palm detector. The model finds palms; each
// palm box is extended toward the fingers to cover the whole hand.
class HandDetector : public IDetector {
public:
    HandDetector(const std::string& modelPath, ClassId handClass, DetectorThresholds thresholds);
    const char* name() const override { return "hand"; }
    bool canDetect(ClassId id) const override { return id == handClass_; }
    std::vector<Detection> detect(const cv::Mat& frame) override;

private:
    cv::dnn::Net net_;
    ClassId handClass_;
    DetectorThresholds th_;
    std::vector<cv::Point2f> anchors_;
};

// Whole-hand box from a palm box and the 7 palm keypoints (wrist first, then the
// finger bases): the palm is extended from the wrist toward the fingertips.
cv::Rect2f handBoxFromPalm(const cv::Rect2f& palm, const std::vector<cv::Point2f>& keypoints);
