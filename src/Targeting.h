// Laser-dot TARGET selection: decides which face, if any, a red laser dot is on right now.
// Two layers:
//   findLaserDot(): per frame and face, is there a laser dot on this face?
//   TargetSelector: over time, turns those answers into one steady target.
// Tracker supplies the face tracks (with their landmarks); Overlay draws the target as a red
// box tagged TARGET.
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
//
// In short, for one face in one frame (details in Targeting.cpp): build the face oval; take the
// skin's typical color and, for each pixel, the skin color right around it; mark the pixels
// whose red rose like a laser's; group them into blobs; keep a blob that is small, round,
// glowing and surrounded by skin. TargetSelector then asks for the dot to last a short while
// before it makes the face the target, and for it to be absent a while before letting go.
//
// Pixel values are 8-bit channel levels (0-255) of OpenCV's BGR frames, so "red" is channel 2.
struct TargetParams {
    // A dot pixel: its red at least this much above the skin's around it (on
    // skin too bright for that: as far as the red can still rise) ...
    float minRedRise = 25;
    // ... and risen at least this much more than its green and blue (on bright
    // skin: half the room the red has left, but at least 15).
    float minRedGain = 35;
    // A bright dot overexposes the camera: its red clips at 255 and, in the middle, so do green and
    // blue (the dot turns white).
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

    // Timing, in seconds of frame time. Different times to become and to stop being the target
    // make this a hysteresis: a flickering dot or a missed frame doesn't make the target blink.
    double enterTime = 0.1;    // s the dot must stay on the face to make it the target
    double dropTime = 0.15;    // s without the dot before it counts as gone (camera misses)
    // s of looking at the target's face (detected) without the dot before it is
    // released; time the face isn't detected doesn't count ...
    double releaseTime = 0.3;
    double coastTime = 0.75;  // ... up to this many s, then it is released too
};

// What the laser search found on one face. found is false when no candidate passed every check;
// the other fields are then unset. Coordinates are frame pixels.
struct LaserDot {
    bool found = false;
    cv::Point2f center;    // frame pixels
    int area = 0;          // pixels
    float redness = 0;     // its red pixels' mean red rise minus their green/blue rise
    float glow = 0;        // its brightest red minus the ring's median red
    float skinAround = 0;  // share of the ring around it that is skin
};

// Laser state of one face track, kept across frames. Times are frame times in seconds; -1 means
// none.
struct FaceLaser {
    bool checked = false;  // looked at this frame (detected now, so its landmarks are fresh)
    LaserDot dot;          // this frame's result
    double onSince = -1;   // the dot has been on the face since; -1 if it isn't
    double lastDot = -1;   // last time a dot was seen on it
    double missing = 0;    // s it was looked at without the dot since the dot was last seen
};

// The current target: the face track the dot is on, and that track's box (frame pixels).
struct Target {
    int trackId = -1;  // -1: no target
    cv::Rect2f box;
    bool active() const { return trackId >= 0; }
};

// Looks for the laser on each face and keeps the target steady over time. Stateful: call
// update() once per frame, in time order.
class TargetSelector {
public:
    explicit TargetSelector(const ClassRegistry& classes, TargetParams params = {});

    // Looks for the laser on each face in this frame (before anything is drawn
    // on it) and updates the target.
    // frame: the BGR frame, untouched by the overlay (its own red would look like a dot).
    // tracks: the tracker's tracks; only visible faces with a fresh detection are searched.
    // now: the frame's time in seconds. Returns the target (trackId -1 if there is none).
    const Target& update(const cv::Mat& frame, const std::vector<Track>& tracks, double now);

    const Target& target() const { return target_; }
    // What was found on each face (for the debug view, the log and --dump-laser).
    const std::map<int, FaceLaser>& faces() const { return faces_; }  // by track id

private:
    // Whether the track is a face (a custom label such as "me" counts through its base class), and
    // whether a dot has been on the face long enough, and recently enough, to count.
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
// box and landmarks are in frame pixels. The result is a rotated rectangle whose inscribed
// ellipse is the outline: center, full width and height, and the tilt in degrees.
cv::RotatedRect faceOval(const cv::Rect2f& box, const std::vector<cv::Point2f>& landmarks);

// Looks for a laser dot on the face inside both `oval` and its detection `box`.
// frame: BGR image. Returns the best candidate (the one whose red rose most beyond its green and
// blue), or found == false. The dot's position is in frame pixels.
LaserDot findLaserDot(const cv::Mat& frame, const cv::RotatedRect& oval, const cv::Rect2f& box,
                      const TargetParams& params = {});
