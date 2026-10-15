#pragma once
#include <deque>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <vector>

#include "Detector.h"

struct Track {
    int id;
    int classId;
    float conf;
    cv::Rect2f box;
    cv::KalmanFilter kf;
    int hits = 1;         // frames matched to a detection
    int misses = 0;       // consecutive frames without a match
    bool confirmed = false;
    std::deque<cv::Point2f> trail;  // recent centers, newest at back
};

struct TrackerParams {
    float iouThreshold = 0.3f;  // min overlap to match a detection to a track
    int minHits = 3;            // matches needed before a track is shown
    int maxMisses = 30;         // frames a track may go unmatched (~1s at 30fps)
    size_t trailLength = 40;
};

// SORT-style tracker: Kalman constant-velocity prediction + greedy IoU matching.
class Tracker {
public:
    explicit Tracker(TrackerParams params = {}) : params_(params) {}

    void update(const std::vector<Detection>& detections);
    const std::vector<Track>& tracks() const { return tracks_; }

private:
    Track createTrack(const Detection& det);

    TrackerParams params_;
    std::vector<Track> tracks_;
    int nextId_ = 1;
};
