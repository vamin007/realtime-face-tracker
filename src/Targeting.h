#pragma once
#include <map>
#include <opencv2/core.hpp>
#include <vector>

#include "ClassRegistry.h"
#include "Tracker.h"

// Picks a TARGET: the face a red laser dot is on.
//
// The dot must be on the face itself, not just inside its box: the face's
// outline is an oval built from its landmarks (eyes, nose, mouth), the dot's
// center must be inside it, and the pixels around the dot must look like that
// face's skin. A dot on the hair, an ear or the wall behind the head doesn't
// count. (Hands aren't detected, so a dot on a hand held in front of the face
// can't be told apart from one on the face.)
//
// A laser dot on a camera image is small, far redder and brighter than the skin
// around it, and often overexposed to white in the middle with a red rim. Red
// is judged against the face's own skin (its median color), so skin tone,
// lighting and white balance don't matter, and red lips or a lit face don't
// pass for a dot.
struct TargetParams {
    // A dot pixel: red at least this much above the skin's (red minus the larger
    // of green and blue) ...
    float minRednessAbove = 60;
    float minRed = 180;  // ... with a red channel at least this bright, 0..255
    float coreLevel = 235;  // overexposed core: every channel at least this (next to red)
    // The dot: at least this many pixels, at most this share of the face's oval,
    // and roughly round.
    int minArea = 3;
    float maxAreaShare = 0.02f;
    float minFill = 0.3f;       // its pixels / its bounding box
    float maxElongation = 3.0f;  // longer side / shorter side of its bounding box
    float minSkinAround = 0.6f;  // share of the ring around it that is this face's skin
    float skinTolerance = 50;    // a ring pixel is skin within this of the skin's median, per channel

    double enterTime = 0.1;    // s the dot must stay on the face to make it the target
    double dropTime = 0.15;    // s without the dot before it counts as gone (camera misses)
    double releaseTime = 0.3;  // s without the dot before the target is released
    double coastTime = 0.75;   // s the target stays on while its face isn't detected
};

// What the laser search found on one face.
struct LaserDot {
    bool found = false;
    cv::Point2f center;    // frame pixels
    int area = 0;          // pixels
    float redness = 0;     // mean redness of the dot minus the skin's
    float skinAround = 0;  // share of the ring around it that is skin
};

// Laser state of one face track.
struct FaceLaser {
    bool checked = false;  // looked at this frame (detected now, so its landmarks are fresh)
    LaserDot dot;          // this frame's result
    double onSince = -1;   // the dot has been on the face since; -1 if it isn't
    double lastDot = -1;   // last time a dot was seen on it
};

struct Target {
    int trackId = -1;  // -1: no target
    cv::Rect2f box;
    cv::Point2f dot;  // where the laser was last seen on it
    bool active() const { return trackId >= 0; }
};

class TargetSelector {
public:
    explicit TargetSelector(const ClassRegistry& classes, TargetParams params = {});

    // Looks for the laser on each face in this frame (before anything is drawn
    // on it) and updates the target.
    const Target& update(const cv::Mat& frame, const std::vector<Track>& tracks, double now);

    const Target& target() const { return target_; }
    const std::map<int, FaceLaser>& faces() const { return faces_; }  // by track id

private:
    bool isFace(const Track& t) const;
    bool dotOn(const FaceLaser& f, double now) const;

    const ClassRegistry& classes_;
    TargetParams params_;
    std::map<int, FaceLaser> faces_;
    Target target_;
    double targetSeen_ = 0;  // when the target's face was last detected
};

// The face's outline from its box and 5 landmarks (right eye, left eye, nose,
// right and left mouth corner): forehead to chin, cheek to cheek, tilted with
// the eyes. Without landmarks, an oval inside the middle of the box.
cv::RotatedRect faceOval(const cv::Rect2f& box, const std::vector<cv::Point2f>& landmarks);

// Looks for a laser dot on the face inside `oval`.
LaserDot findLaserDot(const cv::Mat& frame, const cv::RotatedRect& oval,
                      const TargetParams& params = {});
