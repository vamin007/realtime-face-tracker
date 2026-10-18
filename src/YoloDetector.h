#pragma once
#include <opencv2/dnn.hpp>
#include <string>

#include "IDetector.h"

// YOLOv8 (ONNX) for the 80 COCO classes; class ids are YOLO's output indices.
class YoloDetector : public IDetector {
public:
    YoloDetector(const std::string& modelPath, int numClasses, DetectorThresholds thresholds,
                 cv::Size inputSize = {640, 640});
    const char* name() const override { return "yolo"; }
    bool canDetect(ClassId id) const override { return id >= 0 && id < numClasses_; }
    std::vector<Detection> detect(const cv::Mat& frame) override;

private:
    cv::dnn::Net net_;
    int numClasses_;
    DetectorThresholds th_;
    cv::Size inputSize_;
};

// Turns YOLOv8 output [1, 4 + classes, boxes] into detections in frame pixels.
// scale: input pixels per frame pixel (the letterboxed image sits at the top-left).
std::vector<Detection> decodeYolo(const cv::Mat& output, float scale, cv::Size frameSize,
                                  const DetectorThresholds& th);
