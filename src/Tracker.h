#pragma once
#include <deque>
#include <map>
#include <opencv2/core.hpp>
#include <vector>

#include "ClassRegistry.h"
#include "Detection.h"
#include "KalmanBox.h"

struct Track {
    int id;                  // unique for the whole run, across all classes (used in logs)
    int number = 0;          // per-class number shown on screen ("hand #2"); 0 until confirmed
    ClassId classId;         // person, hand or face: the class of its detections
    float conf;              // smoothed detection confidence
    cv::Rect2f box;          // current estimate, frame pixels
    bool confirmed = false;  // matched often enough to be shown
    bool visible = false;    // drawn and logged this frame
    bool matched = false;    // matched to a detection this frame
    bool coasting = false;   // missing for more than a brief drop-out: drawn thin, as a prediction
    int hits = 1;            // detections matched so far
    double lastSeen = 0;     // time of the last matched detection, seconds
    double lastStrong = 0;   // ... of the last confident (not weak) one
    cv::Rect2f detected;     // box of the last matched detection
    std::vector<cv::Point2f> keypoints;  // from the last matched detection
    std::deque<cv::Point2f> trail;       // recent centers, newest at the back

    // Tracker internals
    KalmanBox kf;
    double kfTime = 0;    // time the filter was last advanced to
    float refSize = 0;    // typical size from detections not cut off by the frame edge
    bool atEdge = false;  // the last detection touched the frame edge (it may have left)
};

struct TrackerParams {
    int minHits = 3;                // matches needed before a track is shown
    double tentativeTimeout = 0.1;  // s: an unconfirmed track missing this long is dropped
    double lostTimeout = 2.0;       // s: a confirmed track missing this long is dropped
                                    //    (hands often vanish for over a second, e.g. in hair)
    double exitTimeout = 3.0;       // s: ... or this long if it was last seen at the frame edge
    double coastTime = 0.2;         // s: a missing confirmed track stays visible this long
    double thinAfter = 0.1;   // s: ... and is drawn thin (a prediction) once missing this long
    double weakBridge = 1.0;  // s: weak detections alone keep a person (YOLO) visible this long
    double weakBridgeFaceHand = 2.0;  // s: ... and a face or hand
    double weakTimeout = 5.0;  // s: any track with no confident detection this long is dropped
    double weakReacquire = 0.5;  // s: weak detections match only tracks seen this recently
    float weakNoise = 1.5f;      // weak detections count as this much noisier: they move a box less
    // s: while unseen, a track's velocity halves this often. Short: a hand the detector
    // misses mostly turns or changes shape, and holding its box predicted it better
    // than carrying on at its last speed.
    double velocityHalfLife = 0.05;

    // Matching. Distances are center distances in units of object size (the
    // square root of the box area; the geometric mean of the track and the detection).
    float maxSizeRatio = 3.0f;      // a detection may be at most this much bigger or smaller
    float minIou = 0.1f;            // pairs overlapping less must be within the distance limit
    float maxDistance = 1.0f;       // distance limit for a track seen in the last frame
    float lostSearchGrowth = 2.0f;  // the limit grows this much per second a track is unseen
    float maxLostDistance = 3.0f;   // ... up to this
    float weakMinIou = 0.3f;        // weak detections must overlap the track this much
    float duplicateIou =
        0.7f;  // a new detection overlapping a matched track this much is a duplicate

    size_t trailLength = 40;
    KalmanBox::Params kalman;
};

// Multi-object tracker:
// - one Kalman filter per track, advanced by the real time between frames;
// - optimal (Hungarian) matching on box overlap and center distance;
// - confident detections first, then weak ones to keep existing tracks going;
// - tracks survive short gaps, slowing down so they don't drift away.
class Tracker {
public:
    explicit Tracker(const ClassRegistry& classes, TrackerParams params = {});

    // Lets the tracker recognize boxes cut off by the frame edge (objects leaving).
    void setFrameSize(cv::Size size) { frameSize_ = size; }

    // timeSeconds: the frame's time; it must not go backwards.
    void update(const std::vector<Detection>& detections, double timeSeconds);
    const std::vector<Track>& tracks() const { return tracks_; }
    const TrackerParams& params() const { return params_; }

private:
    ClassKind kindOf(ClassId id) const;
    bool touchesEdge(const cv::Rect2f& box) const;
    cv::Rect2f inFrame(const cv::Rect2f& box) const;
    float pairCost(const Track& t, const Detection& d, double now, bool weak) const;
    void match(const std::vector<size_t>& trackIdx, const std::vector<size_t>& detIdx,
               const std::vector<Detection>& detections, double now, bool weak,
               std::vector<bool>& detUsed);
    void applyMatch(Track& t, const Detection& d, double now);
    Track createTrack(const Detection& d, double now);
    void confirm(Track& t);

    const ClassRegistry& classes_;
    TrackerParams params_;
    cv::Size frameSize_;  // empty: edges unknown
    std::vector<Track> tracks_;
    int nextId_ = 1;
    std::map<ClassId, int> lastNumber_;  // classId -> last number handed out (never reused)
};
