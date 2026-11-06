// The interface every detector implements, so Detector can run any mix of them without knowing
// which model is behind each. Pipeline role: detectors take a BGR camera frame and return
// Detections (see Detection.h); Detector (Detector.h) runs them side by side and merges the
// results for the tracker. Also holds the confidence thresholds the detectors share, and the
// check for a missing model file.
#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "Detection.h"

// Two confidence levels give the tracker "strong" and "weak" detections (ByteTrack-style):
// strong ones may start tracks, weak ones may only keep a track alive through a bad frame.
struct DetectorThresholds {
    float conf;         // a detection at least this confident can start a track
    float low = 0.1f;   // weaker ones down to this only keep existing tracks going
    // NMS (non-maximum suppression) keeps only the most confident of several boxes on one object.
    // Two boxes are duplicates when their IoU (overlap area / combined area) exceeds this.
    float nms = 0.45f;  // boxes overlapping more than this are duplicates
};

// Common interface of the YOLO and face detectors.
class IDetector {
public:
    virtual ~IDetector() = default;
    // Short stable name ("yolo", "face"): shown in the startup summary and the timing table.
    virtual const char* name() const = 0;
    // Whether this detector can ever produce class `id`; Detector skips detectors that can't
    // produce any class the user asked for (--classes).
    virtual bool canDetect(ClassId id) const = 0;
    // Finds objects in a BGR frame. Returns everything at or above the low
    // threshold; detections below the main threshold are marked weak. Box and keypoint
    // coordinates are in the frame's own pixels, whatever size the model works at internally.
    // The frame is only read, never modified, because Detector may hand the same frame to
    // several detectors on different threads; each detector is called by one thread at a time.
    virtual std::vector<Detection> detect(const cv::Mat& frame) = 0;
    // What it runs on, for the startup summary (e.g. "640x384, CoreML").
    virtual std::string info() const { return ""; }
};

// Throws a readable error if a model file is missing.
void requireModelFile(const std::string& path);
