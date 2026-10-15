#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <set>
#include <string>
#include <vector>

struct Detection {
    cv::Rect2f box;  // in original frame pixels
    int classId;
    float conf;
};

class FaceDetector;
class HandDetector;

// Unified detector: YOLOv8 ONNX + FaceDetector (YuNet) + HandDetector (MediaPipe Palm).
class Detector {
public:
    Detector(const std::string& modelPath, const std::string& namesPath, int inputSize = 640,
             const std::string& faceModelPath = "", const std::string& handModelPath = "");
    ~Detector();

    // allowed: class ids to keep; empty means keep everything.
    std::vector<Detection> detect(const cv::Mat& frame, float confThreshold,
                                  float nmsThreshold, const std::set<int>& allowed);

    const std::vector<std::string>& classNames() const { return names_; }
    int classIdFor(const std::string& name) const;  // -1 if unknown

    void setPersonByFace(bool enabled) { personByFace_ = enabled; }
    bool personByFace() const { return personByFace_; }

    void setEnableHands(bool enabled) { enableHands_ = enabled; }
    bool enableHands() const { return enableHands_; }

private:
    cv::dnn::Net net_;
    std::vector<std::string> names_;
    int inputSize_;
    bool personByFace_ = true;
    bool enableHands_ = true;

    int personClassId_ = 0;
    int handClassId_ = -1;
    int faceClassId_ = -1;

    std::unique_ptr<FaceDetector> faceDetector_;
    std::unique_ptr<HandDetector> handDetector_;
};
