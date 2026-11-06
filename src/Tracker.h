// Multi-object tracker. Turns each frame's detections (people from YOLO, faces from YuNet, each
// with a box and a confidence) into tracks: objects that keep the same id and on-screen number
// from frame to frame and have smooth boxes, even when the detector misses them for a moment.
//
// Per frame, Tracker::update():
//   1. Predict: advance every track's Kalman filter (KalmanBox.h) to the frame's time.
//   2. Match detections to tracks with the Hungarian algorithm (Assignment.h) in ByteTrack-style
//      stages: confident detections first; then weak (low-confidence) ones, which may only
//      continue tracks that already exist and never start one.
//   3. Start new tracks from the confident detections left over; drop tracks missing too long.
//
// A track's life: tentative (new, not shown) -> confirmed after minHits matches (shown, and
// given its per-class number) -> coasting while the detector misses it (still shown, drawn as a
// prediction) -> dropped after a timeout. Numbers are per class and never reused.
#pragma once
#include <deque>
#include <map>
#include <opencv2/core.hpp>
#include <vector>

#include "ClassRegistry.h"
#include "Detection.h"
#include "KalmanBox.h"

// One tracked object. The fields above "Tracker internals" are what the rest of the program
// reads (overlay, logger, targeting). Boxes are in frame pixels: x, y of the top-left corner,
// then width and height. Some fields in more detail:
//   box       the filtered estimate (smoothed, and predicted while the object is unseen), cut to
//             the part inside the frame; detected is the detector's own raw box from the last
//             match
//   visible   confirmed, in view, detected within coastTime and not kept alive by weak
//             detections alone for too long
//   coasting  not detected this frame and unseen for more than thinAfter (a brief miss doesn't
//             count)
//   keypoints for a face, its 5 landmarks (right eye, left eye, nose, right and left mouth
//             corner) in frame pixels; the Targeting code builds the face outline from them
//   trail     one box center per matched frame, at most TrackerParams::trailLength of them
struct Track {
    int id;                  // unique for the whole run, across all classes (used in logs)
    int number = 0;          // per-class number shown on screen ("face #2"); 0 until confirmed
    ClassId classId;         // person or face: the class of its detections
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

    // Tracker internals. refSize is a smoothed sqrt(width * height) and atEdge marks a detection
    // that touches the frame edge; together they let matching accept the smaller box of an
    // object that is leaving.
    KalmanBox kf;
    double kfTime = 0;    // time the filter was last advanced to
    float refSize = 0;    // typical size from detections not cut off by the frame edge
    bool atEdge = false;  // the last detection touched the frame edge (it may have left)
};

// Tuning knobs for Tracker. All times are seconds of video time (frame timestamps), not frame
// counts, so behavior does not depend on the frame rate. tools/replay.cpp re-runs the tracker on
// recorded detections to try other values.
struct TrackerParams {
    // Lifecycle: a new track is tentative until it has been detected minHits times (its first
    // detection counts).
    int minHits = 3;                // matches needed before a track is shown
    double tentativeTimeout = 0.1;  // s: an unconfirmed track missing this long is dropped
    double lostTimeout = 2.0;       // s: a confirmed track missing this long is dropped
                                    //    (set for hands, which often vanished for over a second)
    double exitTimeout = 3.0;       // s: ... or this long if it was last seen at the frame edge
    double coastTime = 0.2;         // s: a missing confirmed track stays visible this long
    double thinAfter = 0.1;   // s: ... and is drawn thin (a prediction) once missing this long
    // Weak detections (below their detector's main confidence threshold, so often noise) may keep
    // a track going but never start one.
    double weakBridge = 1.0;  // s: weak detections alone keep a person (YOLO) visible this long
    double weakBridgeFace = 2.0;  // s: ... and a face
    double weakTimeout = 5.0;  // s: any track with no confident detection this long is dropped
    double weakReacquire = 0.5;  // s: weak detections match only tracks seen this recently
    float weakNoise = 1.5f;      // weak detections count as this much noisier: they move a box less
    // s: while unseen, a track's velocity halves this often. Short: an object the
    // detector misses mostly turns or changes shape (hands did, when they were tracked),
    // and holding its box predicted it better than carrying on at its last speed.
    // Applied once per frame as velocity *= 0.5^(dt / velocityHalfLife).
    double velocityHalfLife = 0.05;

    // Matching. Distances are center distances in units of object size (the
    // square root of the box area; the geometric mean of the track and the detection).
    // IoU (intersection over union) is the overlap area of two boxes divided by the area they
    // cover together: 1 for identical boxes, 0 for boxes that don't touch.
    float maxSizeRatio = 3.0f;      // a detection may be at most this much bigger or smaller
    float minIou = 0.1f;            // pairs overlapping less must be within the distance limit
    float maxDistance = 1.0f;       // distance limit for a track seen in the last frame
    float lostSearchGrowth = 2.0f;  // the limit grows this much per second a track is unseen
    float maxLostDistance = 3.0f;   // ... up to this
    float weakMinIou = 0.3f;        // weak detections must overlap the track this much
    float duplicateIou =
        0.7f;  // a new detection overlapping a matched track this much is a duplicate

    // Points kept in Track::trail.
    size_t trailLength = 40;
    // Noise model of the per-track Kalman filter (see KalmanBox.h).
    KalmanBox::Params kalman;
};

// Multi-object tracker:
// - one Kalman filter per track, advanced by the real time between frames;
// - optimal (Hungarian) matching on box overlap and center distance;
// - confident detections first, then weak ones to keep existing tracks going;
// - tracks survive short gaps, slowing down so they don't drift away.
//
// Use: construct with the class registry (kept by reference, so it must outlive the tracker), call
// setFrameSize() once, then update() once per frame with all of that frame's detections and read
// the result from tracks(); Track::visible says which ones to draw and log.
class Tracker {
public:
    explicit Tracker(const ClassRegistry& classes, TrackerParams params = {});

    // Lets the tracker recognize boxes cut off by the frame edge (objects leaving). size: the
    // frame in pixels. Left empty, the edge logic is off: no box counts as cut off or outside.
    void setFrameSize(cv::Size size) { frameSize_ = size; }

    // detections: everything the detectors found in this frame, people and faces, confident and
    // weak. timeSeconds: the frame's time; it must not go backwards. The Kalman filters are
    // advanced by the real time since the last call, so uneven frame rates are fine.
    void update(const std::vector<Detection>& detections, double timeSeconds);
    // All live tracks, including unconfirmed and hidden ones; filter on Track::visible to show.
    const std::vector<Track>& tracks() const { return tracks_; }
    const TrackerParams& params() const { return params_; }

private:
    // What kind of class this is (person, face); a custom label counts as its base class.
    ClassKind kindOf(ClassId id) const;
    // Frame-edge helpers (no-ops while the frame size is unknown): whether a box reaches the edge,
    // and the part of a box inside the frame.
    bool touchesEdge(const cv::Rect2f& box) const;
    cv::Rect2f inFrame(const cv::Rect2f& box) const;
    // Matching: the cost of one track/detection pair, and one stage of Hungarian matching over a
    // subset of tracks and detections (given as indices).
    float pairCost(const Track& t, const Detection& d, double now, bool weak) const;
    void match(const std::vector<size_t>& trackIdx, const std::vector<size_t>& detIdx,
               const std::vector<Detection>& detections, double now, bool weak,
               std::vector<bool>& detUsed);
    // Bookkeeping: fold a matched detection into a track; start a track; confirm a tentative one.
    void applyMatch(Track& t, const Detection& d, double now);
    Track createTrack(const Detection& d, double now);
    void confirm(Track& t);

    const ClassRegistry& classes_;
    TrackerParams params_;
    cv::Size frameSize_;  // empty: edges unknown
    std::vector<Track> tracks_;
    // Next Track::id (unique across classes for the whole run).
    int nextId_ = 1;
    std::map<ClassId, int> lastNumber_;  // classId -> last number handed out (never reused)
};
