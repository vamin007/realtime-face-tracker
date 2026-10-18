#include "Targeting.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int kSmallWidth = 160;       // background brightness is measured on a 160x90 copy
constexpr float kMeasuredPart = 0.6f;  // the middle of the face box: skin, not hair or background

cv::Rect frameRect(const cv::Mat& frame) { return {0, 0, frame.cols, frame.rows}; }

cv::Rect2f middleOf(const cv::Rect2f& box, float keep) {
    return {box.x + box.width * (1 - keep) / 2, box.y + box.height * (1 - keep) / 2,
            box.width * keep, box.height * keep};
}

}  // namespace

float meanLuma(const cv::Mat& frame, const cv::Rect2f& box, float keep) {
    const cv::Rect inner = cv::Rect(middleOf(box, keep)) & frameRect(frame);
    if (inner.area() <= 0) return 0;
    cv::Mat gray;
    cv::cvtColor(frame(inner), gray, cv::COLOR_BGR2GRAY);
    return static_cast<float>(cv::mean(gray)[0]);
}

float backgroundLuma(const cv::Mat& frame, const std::vector<cv::Rect2f>& exclude) {
    const float scale = static_cast<float>(kSmallWidth) / frame.cols;
    cv::Mat small, gray;
    cv::resize(frame, small, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    cv::Mat mask(gray.size(), CV_8U, cv::Scalar(255));
    for (const cv::Rect2f& b : exclude) {
        const cv::Rect r =
            cv::Rect(cv::Rect2f(b.x * scale, b.y * scale, b.width * scale, b.height * scale)) &
            cv::Rect(0, 0, gray.cols, gray.rows);
        if (r.area() > 0) mask(r).setTo(0);
    }
    // Median of the unmasked pixels from their histogram: a small, very bright
    // spot (the flashlight itself, a lamp) barely moves it.
    int counts[256] = {0};
    int total = 0;
    for (int y = 0; y < gray.rows; ++y) {
        const uchar* g = gray.ptr<uchar>(y);
        const uchar* m = mask.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x)
            if (m[x]) {
                ++counts[g[x]];
                ++total;
            }
    }
    if (total == 0) return static_cast<float>(cv::mean(gray)[0]);
    for (int v = 0, seen = 0; v < 256; ++v) {
        seen += counts[v];
        if (seen * 2 >= total) return static_cast<float>(v);
    }
    return 255;
}

TargetSelector::TargetSelector(const ClassRegistry& classes, TargetParams params)
    : classes_(classes), params_(params) {}

bool TargetSelector::isFace(const Track& t) const {
    return classes_.kind(classes_.base(t.classId)) == ClassKind::Face;
}

const Target& TargetSelector::update(const cv::Mat& frame, const std::vector<Track>& tracks,
                                     double now) {
    std::vector<cv::Rect2f> faceBoxes, handBoxes;
    for (const Track& t : tracks) {
        if (!t.visible) continue;
        if (isFace(t)) faceBoxes.push_back(t.box);
        // Only a hand detected confidently just now counts as being in front of a
        // face: the palm model sometimes weakly mistakes a face for a palm.
        if (classes_.kind(classes_.base(t.classId)) == ClassKind::Hand &&
            now - t.lastStrong <= params_.handRecent)
            handBoxes.push_back(t.box);
    }
    background_ = std::max(backgroundLuma(frame, faceBoxes), 1.0f);

    // 1. Measure every visible face and update its normal ratio.
    for (const Track& t : tracks) {
        if (!t.visible || !isFace(t)) continue;
        auto [it, isNew] = faces_.try_emplace(t.id);
        FaceLight& f = it->second;
        if (isNew) f.firstSeen = now;
        const double dt = isNew ? 0 : now - f.lastUpdate;
        f.lastUpdate = now;

        // Measure only a face detected in this frame with no hand in front of
        // it: a palm held up to the face is brighter than the face behind it.
        const cv::Rect2f measured = middleOf(t.box, kMeasuredPart);
        float handCover = 0;
        for (const cv::Rect2f& h : handBoxes)
            handCover = std::max(handCover, (h & measured).area() / measured.area());
        f.blocked = !t.matched || handCover > params_.maxHandCover;
        if (f.blocked) {
            f.litSince = -1;
            continue;
        }
        f.measuredAt = now;
        f.faceLuma = meanLuma(frame, t.box, kMeasuredPart);
        f.ratio = f.faceLuma / background_;

        const bool learning = now - f.firstSeen < params_.learnTime;
        const bool lit =
            !learning && f.ratio >= params_.minRatio && f.ratio >= params_.minRise * f.normalRatio;
        if (lit) {
            if (f.litSince < 0) f.litSince = now;
        } else {
            f.litSince = -1;
        }
        // The normal ratio: learned fast in the first second, then it follows the
        // face quickly while it's near normal and slowly otherwise. It's frozen
        // while the face is lit or targeted.
        if (f.normalRatio <= 0) {
            f.normalRatio = f.ratio;
        } else if (!lit && t.id != target_.trackId) {
            const bool nearNormal = std::abs(f.rise() - 1) <= params_.normalBand;
            const double tau = learning     ? params_.learnTime / 4
                               : nearNormal ? params_.normalTimeConstant
                                            : params_.driftTimeConstant;
            const float a = static_cast<float>(1 - std::exp(-dt / tau));
            f.normalRatio += a * (f.ratio - f.normalRatio);
        }
    }
    // Forget faces whose tracks are gone.
    for (auto it = faces_.begin(); it != faces_.end();) {
        const bool alive = std::any_of(tracks.begin(), tracks.end(),
                                       [&](const Track& t) { return t.id == it->first; });
        it = alive ? std::next(it) : faces_.erase(it);
    }

    // 2. Keep or release the current target.
    const Track* current = nullptr;
    for (const Track& t : tracks)
        if (t.id == target_.trackId) current = &t;
    if (current) {
        const FaceLight& f = faces_[current->id];
        if (f.measuredAt == now) {
            targetSeen_ = now;
            if (f.rise() < params_.releaseRise) {
                if (releaseSince_ < 0) releaseSince_ = now;
            } else {
                releaseSince_ = -1;
            }
        }
        const bool released = releaseSince_ >= 0 && now - releaseSince_ >= params_.releaseTime;
        const bool lostTooLong = now - targetSeen_ > params_.coastTime;
        if (released || lostTooLong) current = nullptr;
    }

    // 3. The brightest face that has been lit long enough can become (or take
    //    over as) the target.
    const Track* best = nullptr;
    for (const Track& t : tracks) {
        if (!t.visible || !isFace(t)) continue;
        const FaceLight& f = faces_[t.id];
        if (f.litSince < 0 || now - f.litSince < params_.enterTime) continue;
        if (!best || f.ratio > faces_[best->id].ratio) best = &t;
    }
    if (best && best != current &&
        (!current || faces_[best->id].ratio > faces_[current->id].ratio + params_.switchMargin)) {
        current = best;
        targetSeen_ = now;
        releaseSince_ = -1;
    }

    target_ = Target{};
    if (current) {
        const FaceLight& f = faces_[current->id];
        target_ = {current->id, current->box, f.ratio, f.rise()};
    }
    return target_;
}
