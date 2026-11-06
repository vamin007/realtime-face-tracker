// The detection stage of the pipeline (camera thread -> detectors in parallel -> tracker ->
// targeting/recognition -> overlay/log). main() builds one Detector, adds the YOLO and face
// detectors to it, and calls detect() once per frame; the merged list goes to the tracker.
// Because the two models run on different hardware (YOLO on the GPU / Neural Engine through
// CoreML, YuNet on the CPU), running them at the same time makes a frame cost about the slower
// of the two instead of their sum. Also used by label enrollment to find faces in photos.
#pragma once
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "IDetector.h"

// Runs every enabled detector on a frame (side by side) and merges their detections.
// Not thread-safe itself: call it from the one processing thread (it keeps the last timings).
class Detector {
public:
    // How long one detector took on the last frame: its IDetector::name() and wall-clock
    // milliseconds, including its own pre- and post-processing.
    struct Timing {
        std::string name;
        double ms;
    };

    // Takes ownership of a loaded detector. Results are merged in the order detectors were added.
    void add(std::unique_ptr<IDetector> detector);
    bool empty() const { return detectors_.empty(); }
    // true (default): run the detectors on separate threads; false (--sequential, a developer
    // option): one after another on the calling thread.
    void setParallel(bool parallel) { parallel_ = parallel; }

    // allowed: class ids to keep; empty keeps everything. Detectors that can't
    // produce any allowed class are skipped.
    // frame: BGR, only read. Returns detections of all detectors in frame pixels.
    std::vector<Detection> detect(const cv::Mat& frame, const std::set<ClassId>& allowed);

    // Time each detector took on the last frame (only detectors that actually ran).
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
