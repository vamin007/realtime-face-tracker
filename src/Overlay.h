#pragma once
#include <opencv2/core.hpp>
#include <vector>

#include "ClassRegistry.h"
#include "Detector.h"
#include "Recognizer.h"
#include "Targeting.h"
#include "Tracker.h"

struct HudInfo {
    double fps;
    std::vector<Detector::Timing> timings;  // per detector, this frame
    int visibleTracks;
};

// Draws every visible track except the target (drawn by drawTarget). Tracks
// the recognizer gave a custom label show that label instead of "class #n".
void drawTracks(cv::Mat& frame, const std::vector<Track>& tracks, const ClassRegistry& classes,
                bool showTrail = false, int targetId = -1, const Recognizer* recognizer = nullptr);
// The target: a thick red box labeled "TARGET".
void drawTarget(cv::Mat& frame, const Target& target);
void drawHud(cv::Mat& frame, const HudInfo& info);

// A track is drawn and logged once confirmed, while seen recently (see TrackerParams::coastTime).
inline bool isVisible(const Track& t) { return t.visible; }
