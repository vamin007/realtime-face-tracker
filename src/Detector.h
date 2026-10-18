#pragma once
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "IDetector.h"

// Runs every enabled detector on a frame and merges their detections.
class Detector {
public:
    struct Timing {
        std::string name;
        double ms;
    };

    void add(std::unique_ptr<IDetector> detector);
    bool empty() const { return detectors_.empty(); }

    // allowed: class ids to keep; empty keeps everything. Detectors that can't
    // produce any allowed class are skipped.
    std::vector<Detection> detect(const cv::Mat& frame, const std::set<ClassId>& allowed);

    // Time each detector took on the last frame.
    const std::vector<Timing>& timings() const { return timings_; }

private:
    std::vector<std::unique_ptr<IDetector>> detectors_;
    std::vector<Timing> timings_;
};
