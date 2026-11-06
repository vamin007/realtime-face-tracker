// On-screen drawing: boxes, name tags, motion trails, the HUD (frame rate, modes, counts), the
// "TARGET LOCKED" badge, the help panel and the debug view. Everything is drawn in place on the
// BGR frame and scales with the frame's height.
//
// Color rule: red is reserved for the TARGET. trackColor() never returns red and nothing else in
// the overlay is red, so a red box always means "a laser dot is on this face".
// Tags are positioned by TagLayout so they stay inside the frame and off each other.
#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <utility>
#include <vector>

#include "ClassRegistry.h"
#include "Detector.h"
#include "Recognizer.h"
#include "TagLayout.h"
#include "Targeting.h"
#include "Tracker.h"

// What the overlay shows about the current frame besides the tracks (mostly for the HUD, the
// panel in the top-left corner).
struct FrameInfo {
    double fps = 0;
    std::vector<Detector::Timing> timings;  // per detector, this frame (debug view)
    double drawMs = 0;                      // drawing the last frame's overlay (debug view)
    double uptime = 0;                      // s since start, for animations and the "h: help" hint
    bool paused = false;
    bool recording = false;
};

// View switches (see applyViewKey) and the optional data the overlay can show. The pointers are
// not owned and may be null.
struct DrawOptions {
    bool trails = false;                        // motion trail behind each track
    bool debug = false;                         // confidence, laser and recognition details
    bool help = false;                          // the list of keys
    const TargetSelector* targeting = nullptr;  // the TARGET; null when targeting is off
    const Recognizer* recognizer = nullptr;     // custom labels; null when there are none
};

// Draws the tracks and the HUD on the frame. Each track gets a box and a tag
// with "class #n", or its custom label when the recognizer gave it one; the
// target gets a thick red box tagged "TARGET". Tags stay inside the frame and
// off each other, and keep their places from frame to frame.
class Overlay {
public:
    // Draws on `frame` (BGR) in place. tracks: all tracks; only those with Track::visible are
    // drawn. classes: gives the class names for the tags.
    void draw(cv::Mat& frame, const std::vector<Track>& tracks, const ClassRegistry& classes,
              const DrawOptions& options, const FrameInfo& info);

private:
    // Remembers where each tag was last frame, so tags keep their places: keep one Overlay
    // for the whole run rather than making a new one per frame.
    TagLayout tags_;
};

// Applies a key that changes the view: t trails, d debug view, h help. Returns
// false for any other key.
bool applyViewKey(int key, DrawOptions& options);

// Visible tracks per class, sorted by name: {"face", 1}, {"person", 2}.
std::vector<std::pair<std::string, int>> countByClass(const std::vector<Track>& tracks,
                                                      const ClassRegistry& classes);

// A track's color (BGR): stable per id and never red (red is reserved for the TARGET).
cv::Scalar trackColor(int id);

// A track is drawn and logged once confirmed, while seen recently (see TrackerParams::coastTime).
inline bool isVisible(const Track& t) { return t.visible; }
