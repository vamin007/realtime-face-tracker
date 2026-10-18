#pragma once
#include <memory>
#include <string>

#include "IDetector.h"
#include "OrtModel.h"

// Hand detection with MediaPipe's palm detector. The model finds palms; each
// palm box is extended toward the fingers to cover the whole hand.
class HandDetector : public IDetector {
public:
    HandDetector(const std::string& modelPath, ClassId handClass, DetectorThresholds thresholds,
                 Backend backend, const std::string& cacheDir);
    const char* name() const override { return "hand"; }
    bool canDetect(ClassId id) const override { return id == handClass_; }
    std::vector<Detection> detect(const cv::Mat& frame) override;
    std::string info() const override;

private:
    std::unique_ptr<OrtModel> model_;
    ClassId handClass_;
    DetectorThresholds th_;
    std::vector<cv::Point2f> anchors_;
};

// Whole-hand box from a palm box and the 7 palm keypoints (wrist first, then the
// finger bases): the palm is extended from the wrist toward the fingertips.
cv::Rect2f handBoxFromPalm(const cv::Rect2f& palm, const std::vector<cv::Point2f>& keypoints);
