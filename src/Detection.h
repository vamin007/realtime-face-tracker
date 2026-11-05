#pragma once
#include <opencv2/core.hpp>
#include <vector>

using ClassId = int;

// One object found in one frame.
struct Detection {
    cv::Rect2f box;  // frame pixels
    ClassId classId = -1;
    float conf = 0.0f;
    // Below its detector's main threshold: it may keep an existing track going
    // but never starts a new one.
    bool weak = false;
    // Model keypoints in frame pixels. Faces: 5 landmarks (right eye, left eye,
    // nose, right mouth corner, left mouth corner).
    std::vector<cv::Point2f> keypoints;
};
