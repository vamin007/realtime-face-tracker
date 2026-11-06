// The people detector: YOLOv8n exported to ONNX and run through OrtModel (ONNX Runtime, on the
// GPU / Neural Engine via CoreML when the whole graph fits there). It runs in parallel with the
// face detector, and its output feeds the tracker. The key ideas:
//  - the model knows 80 COCO classes, but only the "person" score row of its output is read, so
//    non-person boxes never become candidates (and never suppress a person in NMS);
//  - frames are letterboxed (scaled to fit, padded) to the model's fixed input size;
//  - decodeYolo() is a free function so the post-processing can be unit-tested without a model.
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
// Owns its OrtModel; detect() is meant to be called by one thread at a time.
class YoloDetector : public IDetector {
public:
    // numClasses: class scores per box in the model's output (kCocoClasses). A
    // model with another number is refused: its first class may not be person.
    // (The check happens on the first detect(), when the output shape is known.)
    // Throws std::runtime_error if the file is missing or its input isn't a fixed [1,3,H,W].
    // backend/cacheDir: see OrtModel.
    YoloDetector(const std::string& modelPath, int numClasses, DetectorThresholds thresholds,
                 Backend backend, const std::string& cacheDir);
    const char* name() const override { return "yolo"; }
    bool canDetect(ClassId id) const override { return id == kCocoPerson; }
    // Letterbox -> model -> decodeYolo(). Returns person detections in frame pixels.
    std::vector<Detection> detect(const cv::Mat& frame) override;
    // "<width>x<height>, <backend>", e.g. "640x384, CoreML MLProgram (315/315 nodes)".
    std::string info() const override;

private:
    std::unique_ptr<OrtModel> model_;
    int numClasses_;
    DetectorThresholds th_;
    // The model's fixed input, width x height (cv::Size order), read from the model at load.
    cv::Size inputSize_;
};

// Turns YOLOv8 output [1, 4 + classes, boxes] into person detections in frame pixels.
// scale: input pixels per frame pixel (the letterboxed image sits at the top-left).
// output: the raw float tensor from the model (box centers/sizes in input pixels).
// frameSize: used to clip boxes to the frame. th.low drops weak candidates before NMS;
// th.conf sets which survivors are marked weak.
std::vector<Detection> decodeYolo(const cv::Mat& output, float scale, cv::Size frameSize,
                                  const DetectorThresholds& th);
