#pragma once
#include <memory>
#include <string>

#include "IDetector.h"
#include "OrtModel.h"

// YOLOv8 is trained on the 80 COCO classes; person is the first.
constexpr int kCocoClasses = 80;
constexpr ClassId kCocoPerson = 0;

// YOLOv8 (ONNX), for people only: the other COCO classes' scores aren't read.
// People get class id kCocoPerson, which is also ClassRegistry's person.
// The input size comes from the model (e.g. 640x640, or 640x384 for 16:9 video).
class YoloDetector : public IDetector {
public:
    // numClasses: class scores per box in the model's output (kCocoClasses). A
    // model with another number is refused: its first class may not be person.
    YoloDetector(const std::string& modelPath, int numClasses, DetectorThresholds thresholds,
                 Backend backend, const std::string& cacheDir);
    const char* name() const override { return "yolo"; }
    bool canDetect(ClassId id) const override { return id == kCocoPerson; }
    std::vector<Detection> detect(const cv::Mat& frame) override;
    std::string info() const override;

private:
    std::unique_ptr<OrtModel> model_;
    int numClasses_;
    DetectorThresholds th_;
    cv::Size inputSize_;
};

// Turns YOLOv8 output [1, 4 + classes, boxes] into person detections in frame pixels.
// scale: input pixels per frame pixel (the letterboxed image sits at the top-left).
std::vector<Detection> decodeYolo(const cv::Mat& output, float scale, cv::Size frameSize,
                                  const DetectorThresholds& th);
