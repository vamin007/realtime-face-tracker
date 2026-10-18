#pragma once
#include <memory>
#include <string>

#include "IDetector.h"
#include "OrtModel.h"

// YOLOv8 (ONNX) for the 80 COCO classes; class ids are YOLO's output indices.
// The input size comes from the model (e.g. 640x640, or 640x384 for 16:9 video).
class YoloDetector : public IDetector {
public:
    YoloDetector(const std::string& modelPath, int numClasses, DetectorThresholds thresholds,
                 Backend backend, const std::string& cacheDir);
    const char* name() const override { return "yolo"; }
    bool canDetect(ClassId id) const override { return id >= 0 && id < numClasses_; }
    std::vector<Detection> detect(const cv::Mat& frame) override;
    std::string info() const override;

private:
    std::unique_ptr<OrtModel> model_;
    int numClasses_;
    DetectorThresholds th_;
    cv::Size inputSize_;
};

// Turns YOLOv8 output [1, 4 + classes, boxes] into detections in frame pixels.
// scale: input pixels per frame pixel (the letterboxed image sits at the top-left).
std::vector<Detection> decodeYolo(const cv::Mat& output, float scale, cv::Size frameSize,
                                  const DetectorThresholds& th);
