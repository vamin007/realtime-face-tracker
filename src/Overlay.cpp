// Overlay drawing. Everything is drawn straight onto the frame with OpenCV primitives:
// translucent rounded panels (blended through small masks, so only their own region is touched),
// boxes with corner marks, fading trails, name tags, the HUD, the lock badge, the help panel and
// the startup hint. Sizes are designed for 720p and scaled by Style. Overlay::draw() at the end
// decides what is shown and in which order; TagLayout decides where the tags go.
#include "Overlay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <opencv2/imgproc.hpp>

namespace {

// Colors are BGR (OpenCV's order). kTargetRed is the only red in the overlay: it means TARGET.
const cv::Scalar kTargetRed(0, 0, 255);
const cv::Scalar kWhite(255, 255, 255);
const cv::Scalar kInk(30, 30, 30);      // dark text on light chips and key caps
const cv::Scalar kPanel(28, 26, 24);    // background of tags and panels
const cv::Scalar kSoft(225, 225, 225);  // body text on panels
const cv::Scalar kDim(175, 175, 175);   // secondary text
const cv::Scalar kAmber(0, 190, 255);   // PAUSED
const double kPi = 3.14159265358979;

// Sizes for a frame. Everything scales with the frame's height (1 at 720p), so
// the overlay reads the same at any resolution.
struct Style {
    explicit Style(cv::Size frame) : scale(std::clamp(frame.height / 720.0, 0.5, 3.0)) {}
    // px: a length designed for 720p, scaled to this frame (never below 1 pixel). font: the same
    // for a text size, never below 10 pixels, which is about the smallest that stays readable.
    int px(double v) const { return std::max(1, static_cast<int>(std::lround(v * scale))); }
    int font(double v) const { return std::max(10, px(v)); }
    double scale;
};

cv::FontFace& font() {
    static cv::FontFace face("sans");  // OpenCV's built-in sans-serif, with variable weight
    return face;
}

// Width in pixels of text drawn at this pixel size and weight (400 regular, 700 bold).
int textWidth(const std::string& text, int size, int weight) {
    return cv::getTextSize(cv::Size(), text, cv::Point(), font(), size, weight).width;
}

// Text whose row starts at top: the baseline sits so that letters of this
// font look centered in a row `size` pixels high.
void drawText(cv::Mat& img, const std::string& text, cv::Point topLeft, const cv::Scalar& color,
              int size, int weight) {
    cv::putText(img, text, {topLeft.x, topLeft.y + static_cast<int>(std::lround(0.78 * size))},
                color, font(), size, weight);
}

// Moves the pixels of roi toward color by alpha * mask / 255. Translucent shapes
// are blended through this, so they touch only their own small region. roi shares its pixels
// with the frame (a cv::Mat copy is only a new header), so this edits the frame in place. The
// arithmetic is integer (alpha in 1/256 steps), which is fast enough to do every frame.
void blend(cv::Mat roi, const cv::Mat& mask, const cv::Scalar& color, double alpha) {
    if (roi.type() != CV_8UC3) return;
    const int a = static_cast<int>(alpha * 256);
    const int c[3] = {static_cast<int>(color[0]), static_cast<int>(color[1]),
                      static_cast<int>(color[2])};
    for (int y = 0; y < roi.rows; ++y) {
        auto* p = roi.ptr<cv::Vec3b>(y);
        const uchar* m = mask.ptr<uchar>(y);
        for (int x = 0; x < roi.cols; ++x) {
            if (!m[x]) continue;
            const int w = a * m[x] / 255;
            for (int k = 0; k < 3; ++k)
                p[x][k] = static_cast<uchar>(p[x][k] + (c[k] - p[x][k]) * w / 256);
        }
    }
}

// Fills r with rounded corners (top, bottom or both), blended at alpha (1 = opaque).
// r may reach past the image; only the visible part is drawn. The shape is built as an
// antialiased mask (a rectangle whose corners are cleared and redrawn as filled circles) and
// then blended through blend().
void fillRounded(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, double alpha, int radius,
                 bool roundTop = true, bool roundBottom = true) {
    const cv::Rect clip = r & cv::Rect(0, 0, img.cols, img.rows);
    if (clip.empty()) return;
    radius = std::max(0, std::min({radius, r.width / 2, r.height / 2}));
    const cv::Rect s = r - clip.tl();  // r in the mask's coordinates
    cv::Mat mask(clip.size(), CV_8U, cv::Scalar(0));
    cv::rectangle(mask, s, cv::Scalar(255), cv::FILLED);
    // round: clear the radius x radius square in a corner, then fill a circle there, which leaves
    // that corner rounded.
    auto round = [&](int cornerX, int cornerY, int centerX, int centerY) {
        cv::rectangle(mask, cv::Rect(cornerX, cornerY, radius, radius), cv::Scalar(0), cv::FILLED);
        cv::circle(mask, {centerX, centerY}, radius, cv::Scalar(255), cv::FILLED, cv::LINE_AA);
    };
    const int x0 = s.x, y0 = s.y, x1 = s.x + s.width - radius, y1 = s.y + s.height - radius;
    if (radius > 0 && roundTop) {
        round(x0, y0, x0 + radius, y0 + radius);
        round(x1, y0, x1 - 1, y0 + radius);
    }
    if (radius > 0 && roundBottom) {
        round(x0, y1, x0 + radius, y1 - 1);
        round(x1, y1, x1 - 1, y1 - 1);
    }
    blend(img(clip), mask, color, alpha);
}

// A float rectangle as a pixel rectangle (each value rounded).
cv::Rect toRect(const cv::Rect2f& r) {
    return {static_cast<int>(std::lround(r.x)), static_cast<int>(std::lround(r.y)),
            static_cast<int>(std::lround(r.width)), static_cast<int>(std::lround(r.height))};
}

// r with `by` pixels added on every side.
cv::Rect grown(const cv::Rect& r, int by) {
    return {r.x - by, r.y - by, r.width + 2 * by, r.height + 2 * by};
}

// An L in each corner of r, drawn inside it.
void drawCorners(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, int length,
                 int thickness) {
    length = std::min({length, r.width / 2, r.height / 2});
    thickness = std::min(thickness, length);
    if (length <= 0) return;
    auto bar = [&](int x, int y, int w, int h) {
        cv::rectangle(img, cv::Rect(x, y, w, h), color, cv::FILLED);
    };
    const int x1 = r.x + r.width, y1 = r.y + r.height;
    bar(r.x, r.y, length, thickness);
    bar(r.x, r.y, thickness, length);
    bar(x1 - length, r.y, length, thickness);
    bar(x1 - thickness, r.y, thickness, length);
    bar(r.x, y1 - thickness, length, thickness);
    bar(r.x, y1 - length, thickness, length);
    bar(x1 - length, y1 - thickness, length, thickness);
    bar(x1 - thickness, y1 - length, thickness, length);
}

// A track's box: a thin outline with heavier corners. A prediction (the object
// hasn't been detected for a few frames) keeps only the corners; a single missed
// frame doesn't change the look, so boxes don't flicker.
void drawBox(cv::Mat& img, const cv::Rect& box, const cv::Scalar& color, bool predicted,
             const Style& st) {
    const int length =
        std::clamp(static_cast<int>(0.2 * std::min(box.width, box.height)), st.px(8), st.px(26));
    if (!predicted) cv::rectangle(img, box, color, st.px(1));
    drawCorners(img, box, color, length, st.px(3));
}

// The TARGET: a thick red box, with corner marks around it that pulse gently.
// Its tag goes above the marks (placed against the box grown by targetMargin), not on them.
int targetMargin(const Style& st) { return st.px(14); }

void drawTargetBox(cv::Mat& img, const cv::Rect& box, double uptime, const Style& st) {
    cv::rectangle(img, box, kTargetRed, st.px(3));
    // The marks move in and out 1.2 times a second. The pulse is a function of uptime, so it
    // needs no state kept between frames.
    const double pulse = 0.5 + 0.5 * std::sin(2 * kPi * 1.2 * uptime);
    const int gap = targetMargin(st) - static_cast<int>(std::lround(st.px(5) * pulse));
    const int length =
        std::clamp(static_cast<int>(0.25 * std::min(box.width, box.height)), st.px(12), st.px(34));
    drawCorners(img, grown(box, gap), kTargetRed, length, st.px(3));
}

// A fading trail through the track's recent centers: older parts are fainter
// and thinner. It's drawn into a mask, then blended in one pass over its area.
void drawTrail(cv::Mat& img, const Track& t, const cv::Scalar& color, const Style& st) {
    const auto& pts = t.trail;
    if (pts.size() < 2) return;
    // The trail's bounding box: the mask and the blend only cover that area, not the whole frame.
    float x0 = pts[0].x, y0 = pts[0].y, x1 = x0, y1 = y0;
    for (const cv::Point2f& p : pts) {
        x0 = std::min(x0, p.x), y0 = std::min(y0, p.y);
        x1 = std::max(x1, p.x), y1 = std::max(y1, p.y);
    }
    const int thickness = st.px(3);
    const cv::Rect bounds =
        grown(cv::Rect(cv::Point(static_cast<int>(x0), static_cast<int>(y0)),
                       cv::Point(static_cast<int>(x1) + 1, static_cast<int>(y1) + 1)),
              thickness) &
        cv::Rect(0, 0, img.cols, img.rows);
    if (bounds.empty()) return;
    cv::Mat mask(bounds.size(), CV_8U, cv::Scalar(0));
    // A step longer than the object is a jump (found again after a gap), not motion.
    const float jump = std::max(t.box.width, t.box.height);
    constexpr int kShift = 3;  // sub-pixel line ends, for smooth curves
    auto fixed = [&](const cv::Point2f& p) {
        return cv::Point(static_cast<int>(std::lround((p.x - bounds.x) * (1 << kShift))),
                         static_cast<int>(std::lround((p.y - bounds.y) * (1 << kShift))));
    };
    for (size_t i = 1; i < pts.size(); ++i) {
        if (cv::norm(pts[i] - pts[i - 1]) > jump) continue;
        const double age = static_cast<double>(i) / (pts.size() - 1);  // 1 = newest
        const int w = std::max(1, static_cast<int>(std::lround(thickness * (0.4 + 0.6 * age))));
        // Older segments are thinner (40% to 100% of the width) and fainter (mask 30 to 255).
        cv::line(mask, fixed(pts[i - 1]), fixed(pts[i]), cv::Scalar(30 + 225 * age), w, cv::LINE_AA,
                 kShift);
    }
    blend(img(bounds), mask, color, 0.9);
}

// A piece of text with its style (size in pixels, weight, BGR color).
struct Span {
    std::string text;
    int size, weight;
    cv::Scalar color;
};

// A tag: its name on a dark plate with a stripe of the track's color (the
// TARGET's on solid red), and in the debug view smaller detail lines below.
struct Tag {
    Span name;
    std::vector<Span> details;
    cv::Scalar color;
    bool target = false;
};

// Paddings of a tag, scaled to the frame: inside the plate (padX, padY), the width of the color
// stripe, and the gap between detail lines.
struct TagMetrics {
    int padX, padY, stripe, gap;
    explicit TagMetrics(const Style& st)
        : padX(st.px(7)), padY(st.px(4)), stripe(st.px(4)), gap(st.px(3)) {}
};

int nameHeight(const Tag& tag, const TagMetrics& m) { return tag.name.size + 2 * m.padY; }

// The tag's size: the widest line plus padding (and the stripe, except on the TARGET's tag),
// and the name's height plus the detail lines. TagLayout needs it to place the tag.
cv::Size measure(const Tag& tag, const TagMetrics& m) {
    int w = textWidth(tag.name.text, tag.name.size, tag.name.weight), h = nameHeight(tag, m);
    for (const Span& d : tag.details) {
        w = std::max(w, textWidth(d.text, d.size, d.weight));
        h += d.size + m.gap;
    }
    if (!tag.details.empty()) h += m.padY;
    return {w + 2 * m.padX + (tag.target ? 0 : m.stripe), h};
}

// Draws a tag in r (a spot from TagLayout, as big as measure() said). The TARGET's tag is solid
// red behind its name, with any detail lines on a dark panel below; other tags are a translucent
// dark plate with a stripe in the track's color.
void drawTag(cv::Mat& img, const Tag& tag, const cv::Rect& r, const TagMetrics& m,
             const Style& st) {
    const int radius = st.px(4), head = nameHeight(tag, m);
    const bool details = !tag.details.empty();
    int x = r.x + m.padX;
    if (tag.target) {
        fillRounded(img, {r.x, r.y, r.width, head}, kTargetRed, 1.0, radius, true, !details);
        if (details)
            fillRounded(img, {r.x, r.y + head, r.width, r.height - head}, kPanel, 0.8, radius,
                        false, true);
    } else {
        fillRounded(img, r, kPanel, 0.8, radius);
        cv::rectangle(img, cv::Rect(r.x, r.y, m.stripe, r.height), tag.color, cv::FILLED);
        x += m.stripe;
    }
    drawText(img, tag.name.text, {x, r.y + m.padY}, tag.name.color, tag.name.size, tag.name.weight);
    int y = r.y + head;
    for (const Span& d : tag.details) {
        drawText(img, d.text, {x, y}, d.color, d.size, d.weight);
        y += d.size + m.gap;
    }
}

// Debug view: the detection confidence, why a face is or isn't the TARGET, how
// close the track is to a label, what the face network says, and whether the
// latest check took a photo for the label.
std::vector<std::string> debugLines(const Track& t, bool predicted, const DrawOptions& options) {
    std::vector<std::string> lines;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "conf %.2f%s", t.conf, predicted ? ", predicted" : "");
    lines.push_back(buf);
    if (options.targeting) {
        const auto it = options.targeting->faces().find(t.id);
        if (it != options.targeting->faces().end()) {
            const FaceLaser& f = it->second;
            if (!f.checked) lines.push_back("laser: not checked");
            else if (!f.dot.found) lines.push_back("laser: none");
            else {
                std::snprintf(buf, sizeof(buf), "laser on face, +%.0f red, %d px", f.dot.redness,
                              f.dot.area);
                lines.push_back(buf);
            }
        }
    }
    if (options.recognizer) {
        if (const TrackIdentity* id = options.recognizer->identity(t.id);
            id && !id->closest.empty()) {
            std::snprintf(buf, sizeof(buf), "closest label %s %.2f", id->closest.c_str(),
                          id->closestScore);
            lines.push_back(buf);
            // The face network's answer, or why it's off.
            if (id->netProb >= 0) {
                std::snprintf(buf, sizeof(buf), "network %s %.2f", id->netLabel.c_str(),
                              id->netProb);
                lines.push_back(buf);
            } else if (!options.recognizer->netStatus().empty()) {
                lines.push_back("network " + options.recognizer->netStatus());
            }
            if (!id->photo.empty()) lines.push_back("photo: " + id->photo);
        }
    }
    return lines;
}

// A small rounded chip with a word in capitals, e.g. PAUSED.
struct Chip {
    std::string text;
    cv::Scalar fill, ink;
    double alpha;
    bool dot = false;  // a blinking dot before the text (REC)
};

// The panel in the top-left corner: frame rate and modes, how many of each
// class are tracked, and in the debug view how long each stage took. Returns
// the panel's rect; draws it only when paint is set (the tags are placed
// around it first).
cv::Rect hud(cv::Mat& img, const FrameInfo& info,
             const std::vector<std::pair<std::string, int>>& counts, const DrawOptions& options,
             const Style& st, bool paint) {
    const int big = st.font(18), body = st.font(14), small = st.font(12), chipText = st.font(11);
    const int pad = st.px(10), gap = st.px(6), chipPad = st.px(7), chipH = chipText + st.px(9);
    const int dot = st.px(6);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.0f FPS", info.fps);
    const std::string fps = buf;
    std::vector<Chip> chips;
    if (info.paused) chips.push_back({"PAUSED", kAmber, kInk, 1.0});
    if (info.recording) chips.push_back({"REC", kWhite, kWhite, 0.2, true});
    if (options.debug) chips.push_back({"DEBUG", kWhite, kWhite, 0.2});
    if (options.trails) chips.push_back({"TRAILS", kWhite, kWhite, 0.2});
    // What is tracked, as "face 1  person 2" with the numbers stronger than the names.
    std::vector<Span> tracked;
    for (const auto& [name, n] : counts) {
        tracked.push_back({name, body, 400, kDim});
        tracked.push_back({std::to_string(n), body, 700, kWhite});
    }
    if (tracked.empty()) tracked.push_back({"nothing tracked", body, 400, kDim});
    const int nameGap = st.px(4), pairGap = st.px(12);
    int trackedWidth = 0;
    for (size_t i = 0; i < tracked.size(); ++i) {
        trackedWidth += textWidth(tracked[i].text, body, tracked[i].weight);
        if (i + 1 < tracked.size()) trackedWidth += i % 2 ? pairGap : nameGap;
    }
    std::string timing;
    if (options.debug) {
        for (const auto& t : info.timings) {
            std::snprintf(buf, sizeof(buf), "%s %.0f   ", t.name.c_str(), t.ms);
            timing += buf;
        }
        std::snprintf(buf, sizeof(buf), "draw %.1f ms", info.drawMs);
        timing += buf;
    }

    auto chipWidth = [&](const Chip& c) {
        return textWidth(c.text, chipText, 700) + 2 * chipPad + (c.dot ? dot + st.px(5) : 0);
    };
    int row1 = textWidth(fps, big, 700);
    for (const Chip& c : chips) row1 += gap + chipWidth(c);
    int w = std::max(row1, trackedWidth);
    int h = big + gap + body;
    if (!timing.empty()) {
        w = std::max(w, textWidth(timing, small, 400));
        h += gap + small;
    }
    const cv::Rect panel(st.px(12), st.px(12), w + 2 * pad, h + 2 * pad);
    if (!paint) return panel;

    fillRounded(img, panel, kPanel, 0.62, st.px(8));
    int x = panel.x + pad, y = panel.y + pad;
    drawText(img, fps, {x, y}, kWhite, big, 700);
    x += textWidth(fps, big, 700);
    for (const Chip& c : chips) {
        x += gap;
        const cv::Rect r(x, y + (big - chipH) / 2, chipWidth(c), chipH);
        fillRounded(img, r, c.fill, c.alpha, chipH / 2);
        int tx = r.x + chipPad;
        if (c.dot) {
            if (std::fmod(info.uptime, 1.0) < 0.6)
                cv::circle(img, {tx + dot / 2, r.y + chipH / 2}, dot / 2, c.ink, cv::FILLED,
                           cv::LINE_AA);
            tx += dot + st.px(5);
        }
        drawText(img, c.text, {tx, r.y + (chipH - chipText) / 2 + st.px(1)}, c.ink, chipText, 700);
        x += r.width;
    }
    y += big + gap;
    x = panel.x + pad;
    for (size_t i = 0; i < tracked.size(); ++i) {
        drawText(img, tracked[i].text, {x, y}, tracked[i].color, body, tracked[i].weight);
        x += textWidth(tracked[i].text, body, tracked[i].weight) + (i % 2 ? pairGap : nameGap);
    }
    if (!timing.empty()) drawText(img, timing, {panel.x + pad, y + body + gap}, kDim, small, 400);
    return panel;
}

// "TARGET LOCKED" at the top center while there is a target.
cv::Rect lockBadge(cv::Mat& img, const Style& st, bool paint) {
    const std::string text = "TARGET LOCKED";
    const int size = st.font(15), padX = st.px(12), h = size + st.px(14), icon = size;
    const int w = 2 * padX + icon + st.px(8) + textWidth(text, size, 700);
    const cv::Rect r((img.cols - w) / 2, st.px(12), w, h);
    if (!paint) return r;
    fillRounded(img, r, kTargetRed, 1.0, h / 2);
    // A small bull's-eye: a ring and a dot.
    const cv::Point c(r.x + padX + icon / 2, r.y + h / 2);
    cv::circle(img, c, icon / 2 - st.px(1), kWhite, st.px(2), cv::LINE_AA);
    cv::circle(img, c, st.px(2), kWhite, cv::FILLED, cv::LINE_AA);
    drawText(img, text, {r.x + padX + icon + st.px(8), r.y + (h - size) / 2 + st.px(1)}, kWhite,
             size, 700);
    return r;
}

// A key cap with a letter on it, size x size.
void drawKey(cv::Mat& img, const std::string& key, cv::Point topLeft, int size, const Style& st) {
    fillRounded(img, {topLeft.x, topLeft.y, size, size}, cv::Scalar(235, 235, 235), 1.0, st.px(5));
    const int text = static_cast<int>(size * 0.62);
    drawText(img, key,
             {topLeft.x + (size - textWidth(key, text, 700)) / 2,
              topLeft.y + (size - text) / 2 - st.px(1)},
             kInk, text, 700);
}

// The help: the keys, and what the TARGET and prediction boxes mean.
void drawHelp(cv::Mat& img, const Style& st) {
    struct Line {
        const char* key;
        const char* what;
    };
    const Line keys[] = {{"p", "Pause / resume"},
                         {"t", "Trails on / off"},
                         {"d", "Debug view on / off"},
                         {"h", "Show / hide this help"},
                         {"q", "Quit (Esc also quits)"}};
    const char* target = "TARGET: a face a red laser dot is on";
    const char* predicted = "Corners only: predicted, not detected just now";
    const int title = st.font(18), body = st.font(15), heading = st.font(12);
    const int pad = st.px(22), cap = st.px(26), row = st.px(34), sample = st.px(34);
    const int textX = cap + st.px(14);
    int w =
        std::max({textWidth("Keys", title, 700), sample + st.px(14) + textWidth(target, body, 400),
                  sample + st.px(14) + textWidth(predicted, body, 400)});
    for (const Line& k : keys) w = std::max(w, textX + textWidth(k.what, body, 400));
    const int h = title + st.px(14) + 5 * row + st.px(14) + heading + st.px(10) + 2 * row;
    const cv::Rect panel((img.cols - w) / 2 - pad, (img.rows - h) / 2 - pad, w + 2 * pad,
                         h + 2 * pad);
    fillRounded(img, panel, kPanel, 0.86, st.px(12));
    const int x = panel.x + pad;
    int y = panel.y + pad;
    drawText(img, "Keys", {x, y}, kWhite, title, 700);
    y += title + st.px(14);
    for (const Line& k : keys) {
        drawKey(img, k.key, {x, y + (row - cap) / 2 - st.px(4)}, cap, st);
        drawText(img, k.what, {x + textX, y + (row - body) / 2 - st.px(4)}, kSoft, body, 400);
        y += row;
    }
    y += st.px(14);
    drawText(img, "ON SCREEN", {x, y}, kDim, heading, 700);
    y += heading + st.px(10);
    const int sh = st.px(22);
    const cv::Rect redBox(x + st.px(4), y + (row - sh) / 2 - st.px(4), sample - st.px(8), sh);
    cv::rectangle(img, redBox, kTargetRed, st.px(3));
    drawText(img, target, {x + sample + st.px(14), y + (row - body) / 2 - st.px(4)}, kSoft, body,
             400);
    y += row;
    drawCorners(img, {redBox.x, redBox.y + row, redBox.width, redBox.height}, kSoft, st.px(7),
                st.px(3));
    drawText(img, predicted, {x + sample + st.px(14), y + (row - body) / 2 - st.px(4)}, kSoft, body,
             400);
}

// "Press h for help" at the bottom for the first few seconds, then fading out.
void drawHint(cv::Mat& img, double uptime, const Style& st) {
    // Seconds the hint stays fully visible, then seconds it takes to fade out.
    const double show = 4.0, fade = 1.5;
    if (uptime >= show + fade) return;
    const int body = st.font(15), cap = st.px(22), padX = st.px(14), h = cap + st.px(12);
    const int gap = st.px(8);
    const int w =
        2 * padX + textWidth("Press", body, 400) + cap + textWidth("for help", body, 400) + 2 * gap;
    const cv::Rect r((img.cols - w) / 2, img.rows - h - st.px(24), w, h);
    const cv::Rect clip = r & cv::Rect(0, 0, img.cols, img.rows);
    if (clip.empty()) return;
    const double alpha = uptime <= show ? 1.0 : 1.0 - (uptime - show) / fade;
    // To fade the hint out, keep what is under it and blend that back in at the end.
    const cv::Mat before = img(clip).clone();
    fillRounded(img, r, kPanel, 0.75, h / 2);
    int x = r.x + padX;
    const int textY = r.y + (h - body) / 2;
    drawText(img, "Press", {x, textY}, kSoft, body, 400);
    x += textWidth("Press", body, 400) + gap;
    drawKey(img, "h", {x, r.y + (h - cap) / 2}, cap, st);
    x += cap + gap;
    drawText(img, "for help", {x, textY}, kSoft, body, 400);
    if (alpha < 1) cv::addWeighted(img(clip), alpha, before, 1 - alpha, 0, img(clip));
}

}  // namespace

cv::Scalar trackColor(int id) {
    // Hue steps of 0.618 of the usable range (the golden ratio), so consecutive
    // ids look clearly different. OpenCV hue is 0..179, and red sits at both
    // ends: hues near it (and orange and pink) are skipped.
    // Saturation and brightness are fixed, so every color is vivid and only the hue varies.
    const int hue = 18 + (id * 87) % 140;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 190, 255)), bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return {static_cast<double>(c[0]), static_cast<double>(c[1]), static_cast<double>(c[2])};
}

bool applyViewKey(int key, DrawOptions& options) {
    switch (key) {
        case 't':
            options.trails = !options.trails;
            return true;
        case 'd':
            options.debug = !options.debug;
            return true;
        case 'h':
            options.help = !options.help;
            return true;
        default:
            return false;
    }
}

std::vector<std::pair<std::string, int>> countByClass(const std::vector<Track>& tracks,
                                                      const ClassRegistry& classes) {
    std::map<std::string, int> counts;
    for (const Track& t : tracks)
        if (isVisible(t)) ++counts[classes.name(t.classId)];
    return {counts.begin(), counts.end()};
}

void Overlay::draw(cv::Mat& frame, const std::vector<Track>& tracks, const ClassRegistry& classes,
                   const DrawOptions& options, const FrameInfo& info) {
    const Style st(frame.size());
    const cv::Rect inFrame(0, 0, frame.cols, frame.rows);
    const Target* target = options.targeting && options.targeting->target().active()
                               ? &options.targeting->target()
                               : nullptr;

    // One thing to draw. track is null for the target's last known box (see below).
    struct Shown {
        int id;
        cv::Rect box;
        cv::Scalar color;
        bool target, predicted, labeled;
        const Track* track;  // null for the target's last box
        std::string name;
    };
    std::vector<Shown> shown;
    bool targetShown = false;
    for (const Track& t : tracks) {
        if (!isVisible(t)) continue;
        Shown s{t.id, toRect(t.box) & inFrame, trackColor(t.id), false, false, false, &t, ""};
        if (s.box.empty()) continue;
        s.target = target && t.id == target->trackId;
        s.predicted = t.coasting;
        // A custom label is one specific item, so it needs no number.
        const std::string* custom =
            options.recognizer ? options.recognizer->labelName(t.id) : nullptr;
        s.labeled = custom != nullptr;
        s.name = s.target ? "TARGET"
                 : custom ? *custom
                          : classes.name(t.classId) + " #" + std::to_string(t.number);
        if (s.target) s.color = kTargetRed;
        targetShown |= s.target;
        shown.push_back(std::move(s));
    }
    // The target can be briefly undetected (its track then isn't visible): keep
    // showing it on its last box, so the red box doesn't blink off during a drop-out.
    if (target && !targetShown) {
        const cv::Rect box = toRect(target->box) & inFrame;
        if (!box.empty())
            shown.push_back(
                {target->trackId, box, kTargetRed, true, false, false, nullptr, "TARGET"});
    }

    // Draw order: trails, then the ordinary boxes, then the target's, so the red box is on top.
    if (options.trails)
        for (const Shown& s : shown)
            if (s.track) drawTrail(frame, *s.track, s.color, st);
    for (const Shown& s : shown)
        if (!s.target) drawBox(frame, s.box, s.color, s.predicted, st);
    for (const Shown& s : shown)
        if (s.target) drawTargetBox(frame, s.box, info.uptime, st);

    // Tags: the target's first, then custom labels', then the oldest tracks'. Earlier tags get
    // the better spots in TagLayout, so this order is their priority.
    std::vector<size_t> order(shown.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const Shown &x = shown[a], &y = shown[b];
        if (x.target != y.target) return x.target;
        if (x.labeled != y.labeled) return x.labeled;
        return x.id < y.id;
    });
    const TagMetrics metrics(st);
    std::vector<Tag> tags;
    std::vector<TagLayout::Item> items;
    for (size_t i : order) {
        const Shown& s = shown[i];
        Tag tag{{s.name, st.font(16), s.target ? 700 : 600, kWhite}, {}, s.color, s.target};
        if (options.debug && s.track)
            for (const std::string& line : debugLines(*s.track, s.predicted, options))
                tag.details.push_back({line, st.font(12), 400, kSoft});
        // The target's tag is placed against its box plus the corner marks around it.
        const cv::Rect box = s.target ? grown(s.box, targetMargin(st)) & inFrame : s.box;
        items.push_back({s.id, box, measure(tag, metrics)});
        tags.push_back(std::move(tag));
    }
    const auto counts = countByClass(tracks, classes);
    // Areas the tags must keep off: the HUD panel and the lock badge. They are only measured
    // here (paint = false) and painted after the tags.
    std::vector<cv::Rect> reserved = {hud(frame, info, counts, options, st, false)};
    if (target) reserved.push_back(lockBadge(frame, st, false));
    const std::vector<cv::Rect> spots = tags_.place(items, frame.size(), reserved);
    // Least important first, so if tags must overlap the important ones stay on top.
    for (size_t i = tags.size(); i-- > 0;) drawTag(frame, tags[i], spots[i], metrics, st);

    // The HUD and the badge go on top of the tags.
    hud(frame, info, counts, options, st, true);
    if (target) lockBadge(frame, st, true);
    if (options.help) drawHelp(frame, st);
    else drawHint(frame, info.uptime, st);
}
