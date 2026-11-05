#include "Targeting.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int kMinSkinPixels = 50;  // fewer face pixels than this: too small to judge

cv::Rect frameRect(const cv::Mat& frame) { return {0, 0, frame.cols, frame.rows}; }

float length(cv::Point2f v) { return std::sqrt(v.x * v.x + v.y * v.y); }

// The median of values counted in a histogram.
int medianOf(const int* counts, int bins, int total) {
    for (int v = 0, seen = 0; v < bins; ++v) {
        seen += counts[v];
        if (seen * 2 >= total) return v;
    }
    return bins - 1;
}

// Red minus the larger of green and blue: high for a laser, moderate for skin
// and lips, about zero for grey.
int redness(const cv::Vec3b& p) { return int(p[2]) - std::max(int(p[1]), int(p[0])); }

}  // namespace

cv::RotatedRect faceOval(const cv::Rect2f& box, const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() >= 5) {
        cv::Point2f eyes = landmarks[1] - landmarks[0];
        if (eyes.x < 0) eyes = -eyes;  // left to right in the image, mirrored or not
        const cv::Point2f eyeMid = (landmarks[0] + landmarks[1]) * 0.5f;
        const cv::Point2f mouthMid = (landmarks[3] + landmarks[4]) * 0.5f;
        const float eyeDist = length(eyes);
        const float eyesToMouth = length(mouthMid - eyeMid);
        if (eyeDist >= 2 && eyesToMouth >= 2) {
            const cv::Point2f up = (eyeMid - mouthMid) * (1 / eyesToMouth);
            // Forehead: about one eyes-to-mouth distance above the eyes; chin:
            // 0.8 of it below the mouth; cheeks: one eye distance out from the middle.
            const cv::Point2f top = eyeMid + up * eyesToMouth;
            const cv::Point2f bottom = mouthMid - up * (0.8f * eyesToMouth);
            const float angle = std::atan2(eyes.y, eyes.x) * 180 / static_cast<float>(CV_PI);
            return {(top + bottom) * 0.5f, cv::Size2f(2 * eyeDist, length(top - bottom)), angle};
        }
    }
    return {cv::Point2f(box.x + box.width / 2, box.y + box.height / 2),
            cv::Size2f(box.width * 0.75f, box.height * 0.9f), 0};
}

LaserDot findLaserDot(const cv::Mat& frame, const cv::RotatedRect& oval,
                      const TargetParams& params) {
    const cv::Rect roi = oval.boundingRect() & frameRect(frame);
    if (roi.area() <= 0) return {};
    const cv::Mat img = frame(roi);

    // The face: inside the oval.
    cv::Mat face(roi.size(), CV_8U, cv::Scalar(0));
    cv::RotatedRect local = oval;
    local.center -= cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
    cv::ellipse(face, local, cv::Scalar(255), cv::FILLED);

    // The skin's typical color: the median of each channel and of redness. A dot
    // is a handful of pixels, so it barely moves a median.
    std::array<std::array<int, 256>, 3> channel{};
    std::array<int, 511> red{};  // redness + 255
    int total = 0;
    for (int y = 0; y < img.rows; ++y) {
        const cv::Vec3b* p = img.ptr<cv::Vec3b>(y);
        const uchar* m = face.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x) {
            if (!m[x]) continue;
            for (int c = 0; c < 3; ++c) ++channel[c][p[x][c]];
            ++red[redness(p[x]) + 255];
            ++total;
        }
    }
    if (total < kMinSkinPixels) return {};
    const cv::Vec3i skin(medianOf(channel[0].data(), 256, total),
                         medianOf(channel[1].data(), 256, total),
                         medianOf(channel[2].data(), 256, total));
    const int skinRedness = medianOf(red.data(), 511, total) - 255;

    // Dot pixels: far redder than the skin and bright, plus an overexposed white
    // core where it touches them.
    const float minRedness = skinRedness + params.minRednessAbove;
    cv::Mat redMask(roi.size(), CV_8U, cv::Scalar(0)), coreMask(roi.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < img.rows; ++y) {
        const cv::Vec3b* p = img.ptr<cv::Vec3b>(y);
        const uchar* m = face.ptr<uchar>(y);
        uchar* r = redMask.ptr<uchar>(y);
        uchar* c = coreMask.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x) {
            if (!m[x]) continue;
            if (p[x][2] >= params.minRed && redness(p[x]) >= minRedness) r[x] = 255;
            else if (std::min({p[x][0], p[x][1], p[x][2]}) >= params.coreLevel) c[x] = 255;
        }
    }
    if (cv::countNonZero(redMask) == 0) return {};
    cv::Mat nearRed;
    cv::dilate(redMask, nearRed, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
    cv::Mat dotMask = redMask | (coreMask & nearRed);

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(dotMask, labels, stats, centroids, 8, CV_32S);
    const float maxArea = params.maxAreaShare * static_cast<float>(total);
    LaserDot best;
    for (int i = 1; i < n; ++i) {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (area < params.minArea || area > maxArea) continue;
        if (static_cast<float>(area) / (w * h) < params.minFill) continue;
        if (static_cast<float>(std::max(w, h)) / std::min(w, h) > params.maxElongation) continue;

        // Its red pixels' redness (the white core has none to speak of).
        const cv::Rect bb(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), w, h);
        double rednessSum = 0;
        int redCount = 0;
        for (int y = bb.y; y < bb.y + bb.height; ++y)
            for (int x = bb.x; x < bb.x + bb.width; ++x)
                if (labels.at<int>(y, x) == i && redMask.at<uchar>(y, x)) {
                    rednessSum += redness(img.at<cv::Vec3b>(y, x));
                    ++redCount;
                }
        if (redCount < 2) continue;

        // On the face itself: the ring around the dot (past its glow) must be
        // mostly this face's skin, inside the oval.
        const cv::Point2f c(static_cast<float>(centroids.at<double>(i, 0)),
                            static_cast<float>(centroids.at<double>(i, 1)));
        const float radius = std::max(w, h) / 2.0f;
        const float inner = 2 * radius + 1, outer = std::max(4 * radius, inner + 4);
        int ring = 0, skinRing = 0;
        for (int y = static_cast<int>(c.y - outer); y <= static_cast<int>(c.y + outer); ++y)
            for (int x = static_cast<int>(c.x - outer); x <= static_cast<int>(c.x + outer); ++x) {
                const float d = length(cv::Point2f(x - c.x, y - c.y));
                if (d < inner || d > outer) continue;
                ++ring;
                if (x < 0 || y < 0 || x >= img.cols || y >= img.rows || !face.at<uchar>(y, x))
                    continue;
                const cv::Vec3b p = img.at<cv::Vec3b>(y, x);
                bool like = true;
                for (int ch = 0; ch < 3; ++ch)
                    like = like && std::abs(int(p[ch]) - skin[ch]) <= params.skinTolerance;
                skinRing += like;
            }
        const float skinAround = ring > 0 ? static_cast<float>(skinRing) / ring : 0;
        if (skinAround < params.minSkinAround) continue;

        const float r = static_cast<float>(rednessSum / redCount) - skinRedness;
        if (!best.found || r > best.redness)
            best = {true, c + cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y)),
                    area, r, skinAround};
    }
    return best;
}

TargetSelector::TargetSelector(const ClassRegistry& classes, TargetParams params)
    : classes_(classes), params_(params) {}

bool TargetSelector::isFace(const Track& t) const {
    return classes_.kind(classes_.base(t.classId)) == ClassKind::Face;
}

// The dot was actually seen over at least enterTime (not just first seen that
// long ago) and is still there.
bool TargetSelector::dotOn(const FaceLaser& f, double now) const {
    constexpr double kSlack = 1e-6;  // frame times are sums of floating-point steps
    return f.onSince >= 0 && f.lastDot - f.onSince >= params_.enterTime - kSlack &&
           now - f.lastDot <= params_.dropTime;
}

const Target& TargetSelector::update(const cv::Mat& frame, const std::vector<Track>& tracks,
                                     double now) {
    // 1. Look for the laser on every face detected in this frame (its landmarks
    //    are fresh then; a predicted box may have drifted off the face).
    for (const Track& t : tracks) {
        if (!t.visible || !isFace(t)) continue;
        FaceLaser& f = faces_[t.id];
        f.checked = t.matched;
        f.dot = {};
        if (!t.matched) continue;
        f.dot = findLaserDot(frame, faceOval(t.box, t.keypoints), params_);
        if (f.dot.found) {
            if (f.onSince < 0 || now - f.lastDot > params_.dropTime) f.onSince = now;
            f.lastDot = now;
        } else if (f.onSince >= 0 && now - f.lastDot > params_.dropTime) {
            f.onSince = -1;
        }
    }
    // Forget faces whose tracks are gone.
    for (auto it = faces_.begin(); it != faces_.end();) {
        const bool alive = std::any_of(tracks.begin(), tracks.end(),
                                       [&](const Track& t) { return t.id == it->first; });
        it = alive ? std::next(it) : faces_.erase(it);
    }

    // 2. Keep the current target while the dot was seen on it recently.
    const Track* current = nullptr;
    for (const Track& t : tracks)
        if (t.id == target_.trackId) current = &t;
    if (current) {
        const FaceLaser& f = faces_[current->id];
        if (f.checked) targetSeen_ = now;
        const bool released = now - f.lastDot > params_.releaseTime;
        const bool lostTooLong = now - targetSeen_ > params_.coastTime;
        if (released || lostTooLong) current = nullptr;
    }

    // 3. A face the dot has stayed on becomes the target; it takes over from the
    //    current one once the dot has left that.
    const Track* best = nullptr;
    for (const Track& t : tracks) {
        if (!t.visible || !isFace(t) || !dotOn(faces_[t.id], now)) continue;
        if (!best || faces_[t.id].dot.redness > faces_[best->id].dot.redness) best = &t;
    }
    if (best && best != current && (!current || !dotOn(faces_[current->id], now))) {
        current = best;
        targetSeen_ = now;
    }

    const Target previous = target_;
    target_ = Target{};
    if (current) {
        const FaceLaser& f = faces_[current->id];
        const bool same = previous.trackId == current->id;
        target_ = {current->id, current->box,
                   f.dot.found ? f.dot.center : same ? previous.dot : cv::Point2f()};
    }
    return target_;
}
