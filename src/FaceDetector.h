// The face detector: YuNet, a small face-detection network run by OpenCV's own DNN module on
// the CPU. It runs in parallel with the YOLO person detector (which uses the GPU / Neural
// Engine), and its boxes feed the tracker. Each face also comes with five landmarks (eyes, nose,
// mouth corners): face recognition uses them to straighten the face before embedding it, and
// the laser targeting uses them to outline the face's skin.
#pragma once
#include <opencv2/objdetect.hpp>
#include <string>

#include "IDetector.h"

// Face detection with OpenCV's FaceDetectorYN (YuNet). Each detection carries
// the five face landmarks, which face recognition needs.
// detect() keeps state (the network's input size), so call it from one thread at a time.
class FaceDetector : public IDetector {
public:
    // maxWidth: frames wider than this are scaled down first (faster; faces near
    // the camera are large anyway). 0 keeps the full frame.
    // faceClass: the ClassId given to every detection (ClassRegistry::face()).
    // Throws std::runtime_error if the model file is missing or can't be loaded.
    FaceDetector(const std::string& modelPath, ClassId faceClass, DetectorThresholds thresholds,
                 int maxWidth = 640);
    const char* name() const override { return "face"; }
    bool canDetect(ClassId id) const override { return id == faceClass_; }
    // Returns faces (with 5 keypoints each) in the original frame's pixels, however much the
    // frame was scaled down for the network.
    std::vector<Detection> detect(const cv::Mat& frame) override;
    // E.g. "up to 640 px wide, OpenCV CPU".
    std::string info() const override;

private:
    cv::Ptr<cv::FaceDetectorYN> net_;
    ClassId faceClass_;
    DetectorThresholds th_;
    int maxWidth_;
    cv::Size inputSize_{0, 0};
};
