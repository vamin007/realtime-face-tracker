#include "Targeting.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int kMinSkinPixels = 50;  // fewer face pixels than this: too small to judge
// Eyes-to-mouth distance over eye distance. A face measures 0.9-1.3; when hands
// cover it, YuNet's landmarks go astray (2-10 measured) and an oval built from
// them is a sliver that can reach off the face onto the wall.
constexpr float kMinFaceShape = 0.6f, kMaxFaceShape = 1.7f;
// Bright skin leaves a laser's red little room to rise below 255. The rise and
// gain a dot pixel needs shrink with that room: the rise to the room up to this
// (a saturated dot can come out a few levels lower after compression) ...
constexpr int kSaturationSlack = 5;
// ... and the gain to half the room, but at least this ...
constexpr int kMinGain = 15;
// ... and with less room than this the skin's red is nearly saturated: a laser
// can't be told apart from it by its red (lit lips would pass).
constexpr int kMinRoom = 15;
constexpr float kMinWindow = 15;  // px: the smallest window the skin around a pixel is taken from
constexpr int kCoarse = 4;        // ... at this fraction of the resolution
constexpr double kSlack = 1e-6;   // frame times are sums of floating-point steps

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

int median(std::vector<int>& values) {
    std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
    return values[values.size() / 2];
}

}  // namespace

cv::RotatedRect faceOval(const cv::Rect2f& box, const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() >= 5) {
        cv::Point2f eyes = landmarks[1] - landmarks[0];
        if (eyes.x < 0) eyes = -eyes;  // left to right in the image, mirrored or not
        const cv::Point2f eyeMid = (landmarks[0] + landmarks[1]) * 0.5f;
        const cv::Point2f mouthMid = (landmarks[3] + landmarks[4]) * 0.5f;
        const float eyeDist = length(eyes);
        const float eyesToMouth = length(mouthMid - eyeMid);
        if (eyeDist >= 2 && eyesToMouth >= 2 && eyesToMouth >= kMinFaceShape * eyeDist &&
            eyesToMouth <= kMaxFaceShape * eyeDist) {
            const cv::Point2f up = (eyeMid - mouthMid) * (1 / eyesToMouth);
            // Forehead: about one eyes-to-mouth distance above the eyes; chin:
            // 0.6 of it below the mouth; cheeks: one eye distance out from the middle.
            const cv::Point2f top = eyeMid + up * eyesToMouth;
            const cv::Point2f bottom = mouthMid - up * (0.6f * eyesToMouth);
            const float angle = std::atan2(eyes.y, eyes.x) * 180 / static_cast<float>(CV_PI);
            return {(top + bottom) * 0.5f, cv::Size2f(2 * eyeDist, length(top - bottom)), angle};
        }
    }
    return {cv::Point2f(box.x + box.width / 2, box.y + box.height / 2),
            cv::Size2f(box.width * 0.75f, box.height * 0.9f), 0};
}

LaserDot findLaserDot(const cv::Mat& frame, const cv::RotatedRect& oval, const cv::Rect2f& box,
                      const TargetParams& params) {
    const cv::Rect roi = oval.boundingRect() & cv::Rect(box) & frameRect(frame);
    if (roi.area() <= 0) return {};
    const cv::Mat img = frame(roi);

    // The face: inside the oval and the box (the ROI is inside the box).
    cv::Mat face(roi.size(), CV_8U, cv::Scalar(0));
    cv::RotatedRect local = oval;
    local.center -= cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
    cv::ellipse(face, local, cv::Scalar(255), cv::FILLED);

    // The skin's typical color: the median of each channel. A dot is a handful
    // of pixels, so it barely moves a median.
    std::array<std::array<int, 256>, 3> channel{};
    int total = 0;
    for (int y = 0; y < img.rows; ++y) {
        const cv::Vec3b* p = img.ptr<cv::Vec3b>(y);
        const uchar* m = face.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x) {
            if (!m[x]) continue;
            for (int c = 0; c < 3; ++c) ++channel[c][p[x][c]];
            ++total;
        }
    }
    if (total < kMinSkinPixels) return {};
    const cv::Vec3i skin(medianOf(channel[0].data(), 256, total),
                         medianOf(channel[1].data(), 256, total),
                         medianOf(channel[2].data(), 256, total));

    // The skin right around each pixel: the median over a window a fifth of the
    // face wide (a dot covers well under half of it), anything outside the face
    // counted as its typical skin. It follows uneven light (a flashlight's
    // bright spot), and a broad redder patch (the nose tip, fingers in front of
    // the face) is its own surroundings. It only has to follow the light, so it
    // is worked out at a quarter of the resolution.
    cv::Mat filled = img.clone(), small, nearby;
    filled.setTo(cv::Scalar(skin[0], skin[1], skin[2]), face == 0);
    cv::resize(filled, small, cv::Size(), 1.0 / kCoarse, 1.0 / kCoarse, cv::INTER_AREA);
    const float window = std::max(kMinWindow, 0.2f * oval.size.width) / kCoarse;
    cv::medianBlur(small, small, std::max(3, static_cast<int>(window)) | 1);
    cv::resize(small, nearby, img.size(), 0, 0, cv::INTER_LINEAR);

    // Dot pixels: red risen well above the skin's around it, and much more than
    // green and blue (on bright skin: as far as there is room).
    auto gain = [](const cv::Vec3b& p, const cv::Vec3b& s) {
        return (p[2] - s[2]) - std::max(p[1] - s[1], p[0] - s[0]);
    };
    // Plus the overexposed middle: red saturated (green and blue risen too).
    cv::Mat redMask(roi.size(), CV_8U, cv::Scalar(0)), coreMask(roi.size(), CV_8U, cv::Scalar(0));
    for (int y = 0; y < img.rows; ++y) {
        const cv::Vec3b* p = img.ptr<cv::Vec3b>(y);
        const cv::Vec3b* s = nearby.ptr<cv::Vec3b>(y);
        const uchar* m = face.ptr<uchar>(y);
        uchar* r = redMask.ptr<uchar>(y);
        uchar* c = coreMask.ptr<uchar>(y);
        for (int x = 0; x < img.cols; ++x) {
            if (!m[x]) continue;
            const int room = 255 - kSaturationSlack - s[x][2];
            if (room >= kMinRoom && p[x][2] - s[x][2] >= std::min<int>(params.minRedRise, room) &&
                gain(p[x], s[x]) >= std::min<int>(params.minRedGain, std::max(room / 2, kMinGain)))
                r[x] = 255;
            else if (p[x][2] >= params.saturated) c[x] = 255;
        }
    }
    if (cv::countNonZero(redMask) == 0) return {};
    // Overexposed patches touching red pixels join them, whole: a dot's white
    // middle can be wider than its red rim.
    cv::Mat coreLabels, nearRed;
    const int cores = cv::connectedComponents(coreMask, coreLabels, 8, CV_32S);
    cv::dilate(redMask, nearRed, cv::Mat());
    std::vector<bool> joins(cores, false);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            if (nearRed.at<uchar>(y, x)) joins[coreLabels.at<int>(y, x)] = true;
    joins[0] = false;  // not overexposed
    cv::Mat dotMask = redMask.clone();
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            if (joins[coreLabels.at<int>(y, x)]) dotMask.at<uchar>(y, x) = 255;

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

        // Its red pixels' gain (the white middle has none to speak of), its
        // brightest red, and its mean green and blue.
        const cv::Rect bb(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), w, h);
        double gainSum = 0;
        int redCount = 0, peakRed = 0, white = 0;
        cv::Vec2d greenBlue(0, 0);
        for (int y = bb.y; y < bb.y + bb.height; ++y)
            for (int x = bb.x; x < bb.x + bb.width; ++x) {
                if (labels.at<int>(y, x) != i) continue;
                const cv::Vec3b p = img.at<cv::Vec3b>(y, x);
                peakRed = std::max(peakRed, int(p[2]));
                greenBlue += cv::Vec2d(p[1], p[0]);
                white += std::min({p[0], p[1], p[2]}) >= params.whiteLevel;
                if (redMask.at<uchar>(y, x)) {
                    gainSum += gain(p, nearby.at<cv::Vec3b>(y, x));
                    ++redCount;
                }
            }
        if (redCount < 2) continue;
        const cv::Point2f c(static_cast<float>(centroids.at<double>(i, 0)),
                            static_cast<float>(centroids.at<double>(i, 1)));
        // On skin whose red has little room left, a laser shows mostly as an
        // overexposed spot: green and blue saturate in its middle too. Lit lips
        // don't.
        const int room = 255 - kSaturationSlack - nearby.at<cv::Vec3b>(cv::Point(c))[2];
        if (room < params.minRedRise && white == 0) continue;

        // On the face itself: the ring around the dot (past its glow) must be
        // mostly this face's skin, inside the oval and the box.
        const float radius = std::max(w, h) / 2.0f;
        const float inner = 2 * radius + 1, outer = std::max(4 * radius, inner + 4);
        int ring = 0, skinRing = 0;
        std::array<std::vector<int>, 3> around;  // the ring's face pixels, per channel
        for (int y = static_cast<int>(c.y - outer); y <= static_cast<int>(c.y + outer); ++y)
            for (int x = static_cast<int>(c.x - outer); x <= static_cast<int>(c.x + outer); ++x) {
                const float d = length(cv::Point2f(x - c.x, y - c.y));
                if (d < inner || d > outer) continue;
                ++ring;
                if (x < 0 || y < 0 || x >= img.cols || y >= img.rows || !face.at<uchar>(y, x))
                    continue;
                const cv::Vec3b p = img.at<cv::Vec3b>(y, x);
                bool like = true;
                for (int ch = 0; ch < 3; ++ch) {
                    like = like && std::abs(int(p[ch]) - skin[ch]) <= params.skinTolerance;
                    around[ch].push_back(p[ch]);
                }
                skinRing += like;
            }
        const float skinAround = ring > 0 ? static_cast<float>(skinRing) / ring : 0;
        if (skinAround < params.minSkinAround) continue;

        // It glows: light only adds. Its red is brighter than the skin right
        // around it, and its green and blue aren't darker (as a red mark's are).
        const int ringBlue = median(around[0]), ringGreen = median(around[1]),
                  ringRed = median(around[2]);
        const float glow = static_cast<float>(peakRed - ringRed);
        if (glow < params.minGlow && peakRed < params.saturated) continue;
        greenBlue *= 1.0 / area;
        if (ringGreen - greenBlue[0] > params.maxDarker ||
            ringBlue - greenBlue[1] > params.maxDarker)
            continue;

        const float r = static_cast<float>(gainSum / redCount);
        const cv::Point2f offset(static_cast<float>(roi.x), static_cast<float>(roi.y));
        if (!best.found || r > best.redness) best = {true, c + offset, area, r, glow, skinAround};
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
    return f.onSince >= 0 && f.lastDot - f.onSince >= params_.enterTime - kSlack &&
           now - f.lastDot <= params_.dropTime;
}

const Target& TargetSelector::update(const cv::Mat& frame, const std::vector<Track>& tracks,
                                     double now) {
    const double dt = lastUpdate_ >= 0 ? now - lastUpdate_ : 0;
    lastUpdate_ = now;
    for (auto& [id, f] : faces_) {
        f.checked = false;
        f.dot = {};
    }

    // 1. Look for the laser on every face detected in this frame (its landmarks
    //    are fresh then; a predicted box may have drifted off the face).
    for (const Track& t : tracks) {
        if (!t.visible || !isFace(t)) continue;
        FaceLaser& f = faces_[t.id];
        f.checked = t.matched;
        if (!t.matched) continue;
        f.dot = findLaserDot(frame, faceOval(t.detected, t.keypoints), t.detected, params_);
        if (f.dot.found) {
            if (f.onSince < 0 || now - f.lastDot > params_.dropTime) f.onSince = now;
            f.lastDot = now;
            f.missing = 0;
        } else {
            f.missing += dt;  // looked at, and it wasn't there
            if (f.onSince >= 0 && now - f.lastDot > params_.dropTime) f.onSince = -1;
        }
    }
    // Forget faces whose tracks are gone.
    for (auto it = faces_.begin(); it != faces_.end();) {
        const bool alive = std::any_of(tracks.begin(), tracks.end(),
                                       [&](const Track& t) { return t.id == it->first; });
        it = alive ? std::next(it) : faces_.erase(it);
    }

    // 2. Keep the current target until it has been looked at without the dot
    //    for releaseTime, or gone undetected for coastTime.
    const Track* current = nullptr;
    for (const Track& t : tracks)
        if (t.id == target_.trackId) current = &t;
    if (current) {
        const FaceLaser& f = faces_[current->id];
        if (f.checked) targetSeen_ = now;
        const bool released = f.missing > params_.releaseTime + kSlack;
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

    target_ = current ? Target{current->id, current->box} : Target{};
    return target_;
}
