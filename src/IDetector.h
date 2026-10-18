#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "Detection.h"

struct DetectorThresholds {
    float conf;         // a detection at least this confident can start a track
    float low = 0.1f;   // weaker ones down to this only keep existing tracks going
    float nms = 0.45f;  // boxes overlapping more than this are duplicates
};

// Common interface of the YOLO, face and hand detectors.
class IDetector {
public:
    virtual ~IDetector() = default;
    virtual const char* name() const = 0;
    virtual bool canDetect(ClassId id) const = 0;
    // Finds objects in a BGR frame. Returns everything at or above the low
    // threshold; detections below the main threshold are marked weak.
    virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;
    // What it runs on, for the startup summary (e.g. "640x384, CoreML").
    virtual std::string info() const { return ""; }
};

// Throws a readable error if a model file is missing.
void requireModelFile(const std::string& path);
