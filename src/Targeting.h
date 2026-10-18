#pragma once
#include <map>
#include <opencv2/core.hpp>
#include <vector>

#include "ClassRegistry.h"
#include "Tracker.h"

// Picks a TARGET: the face that a bright light has just started shining on.
//
// For every face: ratio = face brightness / background brightness (so room
// lighting doesn't matter), and its "normal" ratio, learned while it isn't lit.
// A face becomes the target when its ratio is both high (brighter than the
// room) and has risen well above its normal (a light just hit it). Only faces
// are ever measured; a bright phone, lamp or window can't become the target.
// Defaults come from the flashlight reference clip: unlit, the face/room ratio
// was 0.72-0.78 (a bright room); lit, 1.2-1.8 (a rise of 1.5-2x); right after
// the light went off it dipped to ~0.5 for a second while the camera's exposure
// recovered.
struct TargetParams {
    float minRatio = 1.0f;      // a lit face is at least as bright as the room
    float minRise = 1.35f;      // ... and its ratio at least this much above normal
    float releaseRise = 1.2f;   // the target is released below this rise
    double enterTime = 0.1;     // s the lit condition must hold to become the target
    double releaseTime = 0.25;  // s below the release level before letting go
    double learnTime = 1.0;     // s a face must be watched before it can become a target
    // The normal ratio follows the face quickly while it's near normal, and only
    // slowly otherwise, so a lit period (or the dark dip after one) doesn't drag
    // it along but a lasting change in the room is still absorbed.
    float normalBand = 0.15f;         // "near normal": rise within 1 +- this
    double normalTimeConstant = 3.0;  // s, near normal
    double driftTimeConstant = 20.0;  // s, outside the band
    float switchMargin = 0.2f;   // another face takes over only if its ratio is this much higher
    double coastTime = 0.5;      // s the target stays on while its face can't be measured
    float maxHandCover = 0.25f;  // a face covered more than this by a hand isn't measured
};

// Brightness state of one face track.
struct FaceLight {
    double firstSeen = 0;
    double lastUpdate = 0;
    double measuredAt = -1;  // last time its brightness was measured
    bool blocked = false;    // not measured this frame (not detected, or a hand in front)
    float faceLuma = 0;      // mean brightness of the middle of the face, 0..255
    float ratio = 0;         // faceLuma / background brightness, this frame
    float normalRatio = 0;   // its usual value
    double litSince = -1;    // when the lit condition started holding; -1 if it isn't
    float rise() const { return normalRatio > 0 ? ratio / normalRatio : 0; }
};

struct Target {
    int trackId = -1;  // -1: no target
    cv::Rect2f box;
    float ratio = 0;
    float rise = 0;
    bool active() const { return trackId >= 0; }
};

class TargetSelector {
public:
    explicit TargetSelector(const ClassRegistry& classes, TargetParams params = {});

    // Measures the faces in this frame (before anything is drawn on it) and
    // updates the target.
    const Target& update(const cv::Mat& frame, const std::vector<Track>& tracks, double now);

    const Target& target() const { return target_; }
    const std::map<int, FaceLight>& faces() const { return faces_; }  // by track id
    float background() const { return background_; }

private:
    bool isFace(const Track& t) const;

    const ClassRegistry& classes_;
    TargetParams params_;
    std::map<int, FaceLight> faces_;
    Target target_;
    double releaseSince_ = -1;  // when the target's rise first fell below the release level
    double targetSeen_ = 0;     // when the target's face was last detected
    float background_ = 0;
};

// Mean brightness (luma, 0..255) of the middle `keep` fraction of a box.
float meanLuma(const cv::Mat& frame, const cv::Rect2f& box, float keep = 0.6f);

// Median brightness of the frame outside the given boxes, on a small copy of the frame.
float backgroundLuma(const cv::Mat& frame, const std::vector<cv::Rect2f>& exclude);
