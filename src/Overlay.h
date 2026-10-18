#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "ClassRegistry.h"
#include "Tracker.h"

struct HudInfo {
    double fps;
    double inferenceMs;
    int visibleTracks;
};

void drawTracks(cv::Mat& frame, const std::vector<Track>& tracks, const ClassRegistry& classes,
                bool showTrail = false);
void drawHud(cv::Mat& frame, const HudInfo& info);

// A track is drawn/logged only once confirmed and while currently matched.
inline bool isVisible(const Track& t) { return t.confirmed && t.misses == 0; }
