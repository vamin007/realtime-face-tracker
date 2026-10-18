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

struct DrawOptions {
    bool trails = false;                        // motion trail behind each track
    bool debug = false;                         // brightness and recognition details
    const TargetSelector* targeting = nullptr;  // the TARGET; null when targeting is off
    const Recognizer* recognizer = nullptr;     // custom labels; null when there are none
};

// Draws every visible track: "class #n", a custom label instead when the
// recognizer gave it one, or a thick red "TARGET" box for the target.
void drawTracks(cv::Mat& frame, const std::vector<Track>& tracks, const ClassRegistry& classes,
                const DrawOptions& options);
void drawHud(cv::Mat& frame, const HudInfo& info);

// A track is drawn and logged once confirmed, while seen recently (see TrackerParams::coastTime).
inline bool isVisible(const Track& t) { return t.visible; }
