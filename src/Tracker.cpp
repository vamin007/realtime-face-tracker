#include "Tracker.h"

#include <algorithm>
#include <cmath>

#include "Assignment.h"

namespace {

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = (a & b).area();
    const float uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

cv::Point2f center(const cv::Rect2f& r) { return {r.x + r.width / 2, r.y + r.height / 2}; }

float sizeOf(const cv::Rect2f& r) { return std::sqrt(std::max(r.area(), 1.0f)); }

// Weight of the center distance next to (1 - IoU) in a pair's cost.
constexpr float kDistanceWeight = 0.5f;
// Class votes fade by this factor on every match, so recent detections count most.
constexpr float kVoteDecay = 0.9f;
// A different class takes over only once its votes beat the current class by this factor.
constexpr float kClassSwitchMargin = 1.5f;
// Smoothing of the displayed confidence (weight of the newest detection).
constexpr float kConfSmoothing = 0.4f;
// Smoothing of a track's typical size; hand boxes change a lot with the hand's pose.
constexpr float kSizeSmoothing = 0.3f;

}  // namespace

Tracker::Tracker(const ClassRegistry& classes, TrackerParams params)
    : classes_(classes), params_(params) {}

ClassKind Tracker::kindOf(ClassId id) const { return classes_.kind(classes_.base(id)); }

// The part of a box inside the frame (the whole box if it's entirely outside).
cv::Rect2f Tracker::inFrame(const cv::Rect2f& box) const {
    if (frameSize_.empty()) return box;
    const cv::Rect2f inside = box & cv::Rect2f(0, 0, static_cast<float>(frameSize_.width),
                                               static_cast<float>(frameSize_.height));
    return inside.area() > 0 ? inside : box;
}

bool Tracker::touchesEdge(const cv::Rect2f& box) const {
    if (frameSize_.empty()) return false;
    constexpr float margin = 2;
    return box.x <= margin || box.y <= margin || box.x + box.width >= frameSize_.width - margin ||
           box.y + box.height >= frameSize_.height - margin;
}

// Cost of pairing track t with detection d (lower is better), or -1 if they can't match.
float Tracker::pairCost(const Track& t, const Detection& d, double now, bool weak) const {
    if (kindOf(t.classId) != kindOf(d.classId)) return -1;
    // Sizes: the track's typical size (not shrunk by the frame edge) and the detection's.
    const float trackSize = t.refSize > 0 ? t.refSize : sizeOf(t.box);
    const float detSize = sizeOf(d.box);
    const float ratio = detSize / trackSize;
    if (ratio > params_.maxSizeRatio) return -1;
    // Much smaller is fine only for a box cut off by the frame edge (an object
    // leaving): the edge can make a box smaller, never bigger.
    if (ratio * params_.maxSizeRatio < 1 && (weak || !touchesEdge(d.box))) return -1;

    const float overlap = iou(t.box, d.box);
    const cv::Point2f offset = center(t.box) - center(d.box);
    const float distance = std::hypot(offset.x, offset.y) / std::sqrt(trackSize * detSize);
    if (weak) {
        // Weak detections are often noise, so they must overlap the track itself.
        if (overlap < params_.weakMinIou) return -1;
    } else {
        // The longer a track has been unseen, the further away it may turn up.
        const float unseen = static_cast<float>(std::max(0.0, now - t.lastSeen));
        const float limit = std::min(params_.maxLostDistance,
                                     params_.maxDistance + params_.lostSearchGrowth * unseen);
        if (overlap < params_.minIou && distance > limit) return -1;
    }
    float cost = (1 - overlap) + kDistanceWeight * distance;
    if (t.classId != d.classId) {
        // Another class is the same object only if the boxes nearly coincide (tv
        // vs laptop); otherwise a track would swallow neighboring objects.
        if (overlap < params_.classChangeMinIou) return -1;
        cost += params_.classChangeCost;
    }
    return cost;
}

void Tracker::match(const std::vector<size_t>& trackIdx, const std::vector<size_t>& detIdx,
                    const std::vector<Detection>& detections, double now, bool weak,
                    std::vector<bool>& detUsed) {
    if (trackIdx.empty() || detIdx.empty()) return;
    std::vector<std::vector<float>> cost(trackIdx.size(), std::vector<float>(detIdx.size()));
    for (size_t r = 0; r < trackIdx.size(); ++r)
        for (size_t c = 0; c < detIdx.size(); ++c)
            cost[r][c] = pairCost(tracks_[trackIdx[r]], detections[detIdx[c]], now, weak);

    const std::vector<int> pick = assign(cost);
    for (size_t r = 0; r < trackIdx.size(); ++r) {
        if (pick[r] < 0) continue;
        const size_t d = detIdx[pick[r]];
        applyMatch(tracks_[trackIdx[r]], detections[d], now);
        detUsed[d] = true;
    }
}

void Tracker::applyMatch(Track& t, const Detection& d, double now) {
    t.atEdge = touchesEdge(d.box);
    t.kf.correct(d.box);
    // A box cut off by the edge shrinks as the object leaves; that isn't the
    // object getting smaller, so don't let it set a shrinking trend.
    if (t.atEdge) t.kf.stopResizing();
    t.box = inFrame(t.kf.box());
    t.conf = (1 - kConfSmoothing) * t.conf + kConfSmoothing * d.conf;
    t.lastSeen = now;
    if (!d.weak) t.lastStrong = now;
    t.matched = true;
    t.keypoints = d.keypoints;
    if (!t.atEdge) t.refSize = (1 - kSizeSmoothing) * t.refSize + kSizeSmoothing * sizeOf(d.box);
    ++t.hits;
    voteClass(t, d);
    if (!t.confirmed && t.hits >= params_.minHits) confirm(t);
    t.trail.push_back(center(t.box));
    if (t.trail.size() > params_.trailLength) t.trail.pop_front();
}

// Objects can be reported under different classes from frame to frame (tv and
// laptop); the track keeps one identity and shows the class seen most lately.
void Tracker::voteClass(Track& t, const Detection& d) {
    for (auto& [cls, votes] : t.classVotes) votes *= kVoteDecay;
    t.classVotes[d.classId] += d.conf;
    if (d.classId == t.classId) return;
    if (t.classVotes[d.classId] < kClassSwitchMargin * t.classVotes[t.classId]) return;
    t.classId = d.classId;
    if (t.confirmed) t.number = ++lastNumber_[t.classId];  // now a different kind of thing
}

Track Tracker::createTrack(const Detection& d, double now) {
    Track t;
    t.id = nextId_++;
    t.classId = d.classId;
    t.conf = d.conf;
    t.kf = KalmanBox(d.box, params_.kalman);
    t.kfTime = now;
    t.box = inFrame(t.kf.box());
    t.lastSeen = now;
    t.lastStrong = now;  // new tracks only start from confident detections
    t.matched = true;
    t.keypoints = d.keypoints;
    t.atEdge = touchesEdge(d.box);
    t.refSize = sizeOf(d.box);
    t.classVotes[d.classId] = d.conf;
    t.trail.push_back(center(t.box));
    if (params_.minHits <= 1) confirm(t);
    return t;
}

// Numbers are handed out only on confirmation, so brief false detections that
// never get shown don't use them up.
void Tracker::confirm(Track& t) {
    t.confirmed = true;
    t.number = ++lastNumber_[t.classId];
}

void Tracker::update(const std::vector<Detection>& detections, double now) {
    // 1. Advance every track to now. A track that went unseen last frame slows
    //    down, so it waits near where it was lost instead of drifting off.
    for (Track& t : tracks_) {
        const double dt = now - t.kfTime;
        if (!t.matched && dt > 0)
            t.kf.dampVelocity(static_cast<float>(std::pow(0.5, dt / params_.velocityHalfLife)));
        t.kf.predict(dt);
        t.kfTime = now;
        t.box = inFrame(t.kf.box());
        t.matched = false;
    }

    std::vector<size_t> strong, weak;
    for (size_t i = 0; i < detections.size(); ++i)
        (detections[i].weak ? weak : strong).push_back(i);
    std::vector<bool> detUsed(detections.size(), false);
    auto unused = [&](const std::vector<size_t>& idx) {
        std::vector<size_t> out;
        for (size_t i : idx)
            if (!detUsed[i]) out.push_back(i);
        return out;
    };
    auto tracksWhere = [&](auto pred) {
        std::vector<size_t> out;
        for (size_t i = 0; i < tracks_.size(); ++i)
            if (pred(tracks_[i])) out.push_back(i);
        return out;
    };

    // 2. Confident detections against confirmed tracks, including recently lost ones.
    match(tracksWhere([](const Track& t) { return t.confirmed; }), strong, detections, now, false,
          detUsed);
    // 3. Weak detections keep the remaining confirmed tracks going (never start new ones).
    match(tracksWhere([](const Track& t) { return t.confirmed && !t.matched; }), unused(weak),
          detections, now, true, detUsed);
    // 4. The remaining confident detections against new, unconfirmed tracks.
    match(tracksWhere([](const Track& t) { return !t.confirmed; }), unused(strong), detections, now,
          false, detUsed);

    // 5. Confident detections still unmatched start new tracks, unless they
    //    duplicate a track that was just matched (two boxes on one object).
    for (size_t i : unused(strong)) {
        const Detection& d = detections[i];
        const bool duplicate = std::any_of(tracks_.begin(), tracks_.end(), [&](const Track& t) {
            return t.matched && kindOf(t.classId) == kindOf(d.classId) &&
                   iou(t.box, d.box) > params_.duplicateIou;
        });
        if (!duplicate) tracks_.push_back(createTrack(d, now));
    }

    // 6. Drop tracks missing too long, or kept going by weak detections alone for
    //    too long, and show confirmed ones while recently seen.
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) {
                                     const double timeout = !t.confirmed ? params_.tentativeTimeout
                                                            : t.atEdge   ? params_.exitTimeout
                                                                         : params_.lostTimeout;
                                     return now - t.lastSeen > timeout ||
                                            now - t.lastStrong > params_.weakTimeout;
                                 }),
                  tracks_.end());
    for (Track& t : tracks_) {
        const bool inView =
            frameSize_.empty() ||
            (t.box & cv::Rect2f(0, 0, frameSize_.width, frameSize_.height)).area() > 0;
        // A track seen only weakly for a while is hidden (it may be a false
        // detection). YOLO's weak detections (80 classes) are wrong more often than
        // the face and hand models', so objects are hidden sooner.
        const double bridge = kindOf(t.classId) == ClassKind::Object ? params_.weakBridge
                                                                     : params_.weakBridgeFaceHand;
        const bool weakOnlyTooLong = now - t.lastStrong > bridge;
        t.visible =
            t.confirmed && inView && now - t.lastSeen <= params_.coastTime && !weakOnlyTooLong;
    }
}
