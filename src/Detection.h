// The data type that flows between pipeline stages: a detector (YOLO for people, YuNet for faces)
// produces a list of Detections per frame, and the tracker consumes them. Keeping it a plain
// struct means the detectors, the tracker and the CSV dump/replay tools share one vocabulary.
#pragma once
#include <opencv2/core.hpp>
#include <vector>

// Index into ClassRegistry (person, face, then custom labels). A plain int so it can be -1 for
// "none"; ClassRegistry is the only place that hands out valid values.
using ClassId = int;

// One object found in one frame.
struct Detection {
    cv::Rect2f box;  // frame pixels
    ClassId classId = -1;
    // The detector's confidence, 0..1.
    float conf = 0.0f;
    // Below its detector's main threshold: it may keep an existing track going
    // but never starts a new one.
    bool weak = false;
    // Model keypoints in frame pixels. Faces: 5 landmarks (right eye, left eye,
    // nose, right mouth corner, left mouth corner).
    std::vector<cv::Point2f> keypoints;
};
