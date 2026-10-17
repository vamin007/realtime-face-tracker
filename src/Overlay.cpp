#include "Overlay.h"

#include <algorithm>
#include <cstdio>
#include <opencv2/imgproc.hpp>

namespace {

// Stable, well-spread color per track id (golden-angle hue steps).
cv::Scalar colorForId(int id) {
    const int hue = (id * 137) % 180;  // OpenCV hue range is 0..179
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 220, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return {static_cast<double>(c[0]), static_cast<double>(c[1]), static_cast<double>(c[2])};
}

void drawLabel(cv::Mat& frame, const std::string& text, cv::Point origin, const cv::Scalar& bg) {
    int baseline = 0;
    const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    origin.y = std::max(origin.y, size.height + 4);
    cv::rectangle(frame, {origin.x, origin.y - size.height - 4}, {origin.x + size.width + 4, origin.y + 2},
                  bg, cv::FILLED);
    cv::putText(frame, text, {origin.x + 2, origin.y - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
}

}  // namespace

void drawTracks(cv::Mat& frame, const std::vector<Track>& tracks,
                const std::vector<std::string>& classNames,
                bool showTrail) {
    for (const Track& t : tracks) {
        if (!isVisible(t)) continue;
        const cv::Scalar color = colorForId(t.id);

        // Motion trail dot/line is eliminated by default.
        if (showTrail) {
            for (size_t i = 1; i < t.trail.size(); ++i) {
                const int thickness = 1 + static_cast<int>(3 * i / t.trail.size());
                cv::line(frame, t.trail[i - 1], t.trail[i], color, thickness, cv::LINE_AA);
            }
        }

        cv::rectangle(frame, t.box, color, 2);
        char label[96];
        const std::string name = (t.classId >= 0 && t.classId < static_cast<int>(classNames.size()))
                                     ? classNames[t.classId]
                                     : "obj";
        std::snprintf(label, sizeof(label), "%s #%d %.2f", name.c_str(), t.number, t.conf);
        drawLabel(frame, label, {static_cast<int>(t.box.x), static_cast<int>(t.box.y)}, color);
    }
}

void drawHud(cv::Mat& frame, const HudInfo& info) {
    char text[96];
    std::snprintf(text, sizeof(text), "FPS %.1f | detect %.0f ms | tracks %d", info.fps,
                  info.inferenceMs, info.visibleTracks);
    drawLabel(frame, text, {8, 28}, cv::Scalar(255, 255, 255));
}
