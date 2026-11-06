#pragma once
#include <map>
#include <opencv2/core.hpp>
#include <vector>

#include "ClassRegistry.h"
#include "Tracker.h"

// Picks a TARGET: the face a red laser dot is on.
//
// The dot must be on the face itself, not just inside its box: the face's
// outline is an oval built from its landmarks (eyes, nose, mouth) and cut to
// its detection box, the dot's center must be inside it, and the pixels around
// the dot must look like that face's skin. A dot on the hair, an ear or the
// wall behind the head doesn't count. (Hands aren't detected, so a dot on a
// hand held in front of the face can't be told apart from one on the face.)
//
// A laser adds red light. On a camera image its dot is small, its red rises
// well above the skin's (as far as the skin's red has room to rise below 255)
// and much more than its green and blue, and it is often overexposed to white
// in the middle. It glows: its red is brighter than the skin right around it,
// and none of its colors is darker. Color is judged against the face's own
// skin right around each pixel, so skin tone, lighting, a flashlight's bright
// spot and white balance don't matter, and red lips, a red sticker or a lit
// face don't pass for a dot.
struct TargetParams {
    // A dot pixel: its red at least this much above the skin's around it (on
    // skin too bright for that: as far as the red can still rise) ...
    float minRedRise = 25;
    // ... and risen at least this much more than its green and blue (on bright
    // skin: half the room the red has left, but at least 15).
    float minRedGain = 35;
    float saturated = 250;   // red at least this is overexposed
    float whiteLevel = 235;  // every channel at least this: white (a bright dot's middle)
    // The dot glows: its brightest red at least this above the ring around it
    // (or overexposed), and its green and blue at most this below the ring's.
    // Pigment (a red sticker, a bindi) is lit like the skin around it and
    // absorbs green and blue.
    float minGlow = 30;
    float maxDarker = 15;
    // The dot: at least this many pixels, at most this share of the face's oval,
    // and roughly round.
    int minArea = 3;
    float maxAreaShare = 0.01f;
    float minFill = 0.3f;       // its pixels / its bounding box
    float maxElongation = 3.0f;  // longer side / shorter side of its bounding box
    float minSkinAround = 0.6f;  // share of the ring around it that is this face's skin
    float skinTolerance = 50;    // a ring pixel is skin within this of the skin's median, per channel

    double enterTime = 0.1;    // s the dot must stay on the face to make it the target
    double dropTime = 0.15;    // s without the dot before it counts as gone (camera misses)
    // s of looking at the target's face (detected) without the dot before it is
    // released; time the face isn't detected doesn't count ...
    double releaseTime = 0.3;
    double coastTime = 0.75;  // ... up to this many s, then it is released too
};

// What the laser search found on one face.
struct LaserDot {
    bool found = false;
    cv::Point2f center;    // frame pixels
    int area = 0;          // pixels
    float redness = 0;     // its red pixels' mean red rise minus their green/blue rise
    float glow = 0;        // its brightest red minus the ring's median red
    float skinAround = 0;  // share of the ring around it that is skin
};

// Laser state of one face track.
struct FaceLaser {
    bool checked = false;  // looked at this frame (detected now, so its landmarks are fresh)
    LaserDot dot;          // this frame's result
    double onSince = -1;   // the dot has been on the face since; -1 if it isn't
    double lastDot = -1;   // last time a dot was seen on it
    double missing = 0;    // s it was looked at without the dot since the dot was last seen
};

struct Target {
    int trackId = -1;  // -1: no target
    cv::Rect2f box;
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
    double targetSeen_ = 0;   // when the target's face was last detected
    double lastUpdate_ = -1;  // the previous frame's time
};

// The face's outline from its box and 5 landmarks (right eye, left eye, nose,
// right and left mouth corner): forehead to chin, cheek to cheek, tilted with
// the eyes. Without landmarks, or with landmarks that aren't shaped like a face
// (YuNet's guess when hands cover it), an oval inside the middle of the box.
cv::RotatedRect faceOval(const cv::Rect2f& box, const std::vector<cv::Point2f>& landmarks);

// Looks for a laser dot on the face inside both `oval` and its detection `box`.
LaserDot findLaserDot(const cv::Mat& frame, const cv::RotatedRect& oval, const cv::Rect2f& box,
                      const TargetParams& params = {});
