#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

#include "ClassRegistry.h"
#include "Detector.h"
#include "Recognizer.h"
#include "TagLayout.h"
#include "Tracker.h"

// What the overlay shows about the current frame besides the tracks.
struct FrameInfo {
    double fps = 0;
    std::vector<Detector::Timing> timings;  // per detector, this frame (debug view)
    double drawMs = 0;                      // drawing the last frame's overlay (debug view)
    double uptime = 0;                      // s since start, for animations and the "h: help" hint
    bool paused = false;
    bool recording = false;
};

struct DrawOptions {
    bool trails = false;                     // motion trail behind each track
    bool debug = false;                      // confidence and recognition details
    bool help = false;                       // the list of keys
    const Recognizer* recognizer = nullptr;  // custom labels; null when there are none
};

// Draws the tracks and the HUD on the frame. Each track gets a box and a tag
// with "class #n", or its custom label when the recognizer gave it one. Tags
// stay inside the frame and off each other, and keep their places from frame
// to frame.
class Overlay {
public:
    void draw(cv::Mat& frame, const std::vector<Track>& tracks, const ClassRegistry& classes,
              const DrawOptions& options, const FrameInfo& info);

private:
    TagLayout tags_;
};

// Applies a key that changes the view: t trails, d debug view, h help. Returns
// false for any other key.
bool applyViewKey(int key, DrawOptions& options);

// Visible tracks per class, sorted by name: {"face", 1}, {"hand", 2}.
std::vector<std::pair<std::string, int>> countByClass(const std::vector<Track>& tracks,
                                                      const ClassRegistry& classes);

// A track's color: stable per id, from a band of hues that leaves out red, orange and pink.
cv::Scalar trackColor(int id);

// A track is drawn and logged once confirmed, while seen recently (see TrackerParams::coastTime).
inline bool isVisible(const Track& t) { return t.visible; }
