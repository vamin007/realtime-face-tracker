#pragma once
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "IDetector.h"

// Runs every enabled detector on a frame (side by side) and merges their detections.
class Detector {
public:
    struct Timing {
        std::string name;
        double ms;
    };

    void add(std::unique_ptr<IDetector> detector);
    bool empty() const { return detectors_.empty(); }
    void setParallel(bool parallel) { parallel_ = parallel; }

    // allowed: class ids to keep; empty keeps everything. Detectors that can't
    // produce any allowed class are skipped.
    std::vector<Detection> detect(const cv::Mat& frame, const std::set<ClassId>& allowed);

    // Time each detector took on the last frame.
    const std::vector<Timing>& timings() const { return timings_; }

    // One line per detector: what it runs on.
    std::vector<std::string> summary() const;
    // The detectors that loaded, by name ("yolo", "face").
    std::vector<std::string> names() const;

private:
    std::vector<std::unique_ptr<IDetector>> detectors_;
    std::vector<Timing> timings_;
    bool parallel_ = true;
};
