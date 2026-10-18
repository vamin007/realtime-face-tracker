#include "Overlay.h"

#include <algorithm>
#include <cstdio>
#include <opencv2/imgproc.hpp>

namespace {

const cv::Scalar kTargetRed(0, 0, 255);

// Stable, well-spread color per track id: hue steps of 0.618 of the usable range
// (the golden ratio), so consecutive ids look clearly different. Red hues are
// skipped: red is reserved for the TARGET.
cv::Scalar colorForId(int id) {
    const int hue = 15 + (id * 93) % 150;  // OpenCV hue is 0..179; red sits at both ends
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 220, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return {static_cast<double>(c[0]), static_cast<double>(c[1]), static_cast<double>(c[2])};
}

// Black text on light backgrounds, white on dark ones.
cv::Scalar textColorOn(const cv::Scalar& bg) {
    const double luma = 0.114 * bg[0] + 0.587 * bg[1] + 0.299 * bg[2];
    return luma > 140 ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
}

void drawLabel(cv::Mat& frame, const std::string& text, cv::Point origin, const cv::Scalar& bg) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    origin.y = std::max(origin.y, size.height + 4);
    cv::rectangle(frame, {origin.x, origin.y - size.height - 4},
                  {origin.x + size.width + 4, origin.y + 2}, bg, cv::FILLED);
    cv::putText(frame, text, {origin.x + 2, origin.y - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                textColorOn(bg), 2, cv::LINE_AA);
}

}  // namespace

void drawTracks(cv::Mat& frame, const std::vector<Track>& tracks, const ClassRegistry& classes,
                bool showTrail, int targetId, const Recognizer* recognizer) {
    for (const Track& t : tracks) {
        if (!isVisible(t) || t.id == targetId) continue;
        const cv::Scalar color = colorForId(t.id);
        if (showTrail) {
            for (size_t i = 1; i < t.trail.size(); ++i) {
                const int thickness = 1 + static_cast<int>(3 * i / t.trail.size());
                cv::line(frame, t.trail[i - 1], t.trail[i], color, thickness, cv::LINE_AA);
            }
        }
        cv::rectangle(frame, t.box, color, 2);
        char label[96];
        // A custom label is one specific item, so it needs no number.
        const std::string* custom = recognizer ? recognizer->labelName(t.id) : nullptr;
        if (custom) std::snprintf(label, sizeof(label), "%s %.2f", custom->c_str(), t.conf);
        else
            std::snprintf(label, sizeof(label), "%s #%d %.2f", classes.name(t.classId).c_str(),
                          t.number, t.conf);
        drawLabel(frame, label, {static_cast<int>(t.box.x), static_cast<int>(t.box.y)}, color);
    }
}

void drawTarget(cv::Mat& frame, const Target& target) {
    if (!target.active()) return;
    cv::rectangle(frame, target.box, kTargetRed, 3);
    drawLabel(frame, "TARGET", {static_cast<int>(target.box.x), static_cast<int>(target.box.y)},
              kTargetRed);
}

void drawHud(cv::Mat& frame, const HudInfo& info) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "FPS %.1f |", info.fps);
    std::string text = buf;
    for (const auto& t : info.timings) {
        std::snprintf(buf, sizeof(buf), " %s %.0f", t.name.c_str(), t.ms);
        text += buf;
    }
    std::snprintf(buf, sizeof(buf), " ms | tracks %d", info.visibleTracks);
    text += buf;
    drawLabel(frame, text, {8, 28}, cv::Scalar(255, 255, 255));
}
