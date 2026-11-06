#include "Overlay.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <opencv2/imgproc.hpp>

namespace {

const cv::Scalar kWhite(255, 255, 255);
const cv::Scalar kInk(30, 30, 30);      // dark text on light chips and key caps
const cv::Scalar kPanel(28, 26, 24);    // background of tags and panels
const cv::Scalar kSoft(225, 225, 225);  // body text on panels
const cv::Scalar kDim(175, 175, 175);   // secondary text
const cv::Scalar kAmber(0, 190, 255);   // PAUSED

// Sizes for a frame. Everything scales with the frame's height (1 at 720p), so
// the overlay reads the same at any resolution.
struct Style {
    explicit Style(cv::Size frame) : scale(std::clamp(frame.height / 720.0, 0.5, 3.0)) {}
    int px(double v) const { return std::max(1, static_cast<int>(std::lround(v * scale))); }
    int font(double v) const { return std::max(10, px(v)); }
    double scale;
};

cv::FontFace& font() {
    static cv::FontFace face("sans");  // OpenCV's built-in sans-serif, with variable weight
    return face;
}

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
// are blended through this, so they touch only their own small region.
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
void fillRounded(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, double alpha, int radius,
                 bool roundTop = true, bool roundBottom = true) {
    const cv::Rect clip = r & cv::Rect(0, 0, img.cols, img.rows);
    if (clip.empty()) return;
    radius = std::max(0, std::min({radius, r.width / 2, r.height / 2}));
    const cv::Rect s = r - clip.tl();  // r in the mask's coordinates
    cv::Mat mask(clip.size(), CV_8U, cv::Scalar(0));
    cv::rectangle(mask, s, cv::Scalar(255), cv::FILLED);
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

cv::Rect toRect(const cv::Rect2f& r) {
    return {static_cast<int>(std::lround(r.x)), static_cast<int>(std::lround(r.y)),
            static_cast<int>(std::lround(r.width)), static_cast<int>(std::lround(r.height))};
}

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

// A fading trail through the track's recent centers: older parts are fainter
// and thinner. It's drawn into a mask, then blended in one pass over its area.
void drawTrail(cv::Mat& img, const Track& t, const cv::Scalar& color, const Style& st) {
    const auto& pts = t.trail;
    if (pts.size() < 2) return;
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
        cv::line(mask, fixed(pts[i - 1]), fixed(pts[i]), cv::Scalar(30 + 225 * age), w, cv::LINE_AA,
                 kShift);
    }
    blend(img(bounds), mask, color, 0.9);
}

struct Span {
    std::string text;
    int size, weight;
    cv::Scalar color;
};

// A tag: its name on a dark plate with a stripe of the track's color, and in
// the debug view smaller detail lines below.
struct Tag {
    Span name;
    std::vector<Span> details;
    cv::Scalar color;
};

struct TagMetrics {
    int padX, padY, stripe, gap;
    explicit TagMetrics(const Style& st)
        : padX(st.px(7)), padY(st.px(4)), stripe(st.px(4)), gap(st.px(3)) {}
};

int nameHeight(const Tag& tag, const TagMetrics& m) { return tag.name.size + 2 * m.padY; }

cv::Size measure(const Tag& tag, const TagMetrics& m) {
    int w = textWidth(tag.name.text, tag.name.size, tag.name.weight), h = nameHeight(tag, m);
    for (const Span& d : tag.details) {
        w = std::max(w, textWidth(d.text, d.size, d.weight));
        h += d.size + m.gap;
    }
    if (!tag.details.empty()) h += m.padY;
    return {w + 2 * m.padX + m.stripe, h};
}

void drawTag(cv::Mat& img, const Tag& tag, const cv::Rect& r, const TagMetrics& m,
             const Style& st) {
    const int radius = st.px(4), head = nameHeight(tag, m);
    fillRounded(img, r, kPanel, 0.8, radius);
    cv::rectangle(img, cv::Rect(r.x, r.y, m.stripe, r.height), tag.color, cv::FILLED);
    const int x = r.x + m.padX + m.stripe;
    drawText(img, tag.name.text, {x, r.y + m.padY}, tag.name.color, tag.name.size, tag.name.weight);
    int y = r.y + head;
    for (const Span& d : tag.details) {
        drawText(img, d.text, {x, y}, d.color, d.size, d.weight);
        y += d.size + m.gap;
    }
}

// Debug view: the detection confidence, how close the track is to a label, what
// the face network says, and whether the latest check took a photo for the label.
std::vector<std::string> debugLines(const Track& t, bool predicted, const DrawOptions& options) {
    std::vector<std::string> lines;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "conf %.2f%s", t.conf, predicted ? ", predicted" : "");
    lines.push_back(buf);
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
    // What is tracked, as "face 1  hand 2" with the numbers stronger than the names.
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

// A key cap with a letter on it, size x size.
void drawKey(cv::Mat& img, const std::string& key, cv::Point topLeft, int size, const Style& st) {
    fillRounded(img, {topLeft.x, topLeft.y, size, size}, cv::Scalar(235, 235, 235), 1.0, st.px(5));
    const int text = static_cast<int>(size * 0.62);
    drawText(img, key,
             {topLeft.x + (size - textWidth(key, text, 700)) / 2,
              topLeft.y + (size - text) / 2 - st.px(1)},
             kInk, text, 700);
}

// The help: the keys, and what the prediction boxes mean.
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
    const char* predicted = "Corners only: predicted, not detected just now";
    const int title = st.font(18), body = st.font(15), heading = st.font(12);
    const int pad = st.px(22), cap = st.px(26), row = st.px(34), sample = st.px(34);
    const int textX = cap + st.px(14);
    int w = std::max(textWidth("Keys", title, 700),
                     sample + st.px(14) + textWidth(predicted, body, 400));
    for (const Line& k : keys) w = std::max(w, textX + textWidth(k.what, body, 400));
    const int h = title + st.px(14) + 5 * row + st.px(14) + heading + st.px(10) + row;
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
    drawCorners(img, {x + st.px(4), y + (row - sh) / 2 - st.px(4), sample - st.px(8), sh}, kSoft,
                st.px(7), st.px(3));
    drawText(img, predicted, {x + sample + st.px(14), y + (row - body) / 2 - st.px(4)}, kSoft, body,
             400);
}

// "Press h for help" at the bottom for the first few seconds, then fading out.
void drawHint(cv::Mat& img, double uptime, const Style& st) {
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

    struct Shown {
        int id;
        cv::Rect box;
        cv::Scalar color;
        bool predicted, labeled;
        const Track* track;
        std::string name;
    };
    std::vector<Shown> shown;
    for (const Track& t : tracks) {
        if (!isVisible(t)) continue;
        Shown s{t.id, toRect(t.box) & inFrame, trackColor(t.id), false, false, &t, ""};
        if (s.box.empty()) continue;
        s.predicted = t.coasting;
        // A custom label is one specific item, so it needs no number.
        const std::string* custom =
            options.recognizer ? options.recognizer->labelName(t.id) : nullptr;
        s.labeled = custom != nullptr;
        s.name = custom ? *custom : classes.name(t.classId) + " #" + std::to_string(t.number);
        shown.push_back(std::move(s));
    }

    if (options.trails)
        for (const Shown& s : shown) drawTrail(frame, *s.track, s.color, st);
    for (const Shown& s : shown) drawBox(frame, s.box, s.color, s.predicted, st);

    // Tags: custom labels' first, then the oldest tracks'.
    std::vector<size_t> order(shown.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const Shown &x = shown[a], &y = shown[b];
        if (x.labeled != y.labeled) return x.labeled;
        return x.id < y.id;
    });
    const TagMetrics metrics(st);
    std::vector<Tag> tags;
    std::vector<TagLayout::Item> items;
    for (size_t i : order) {
        const Shown& s = shown[i];
        Tag tag{{s.name, st.font(16), 600, kWhite}, {}, s.color};
        if (options.debug)
            for (const std::string& line : debugLines(*s.track, s.predicted, options))
                tag.details.push_back({line, st.font(12), 400, kSoft});
        items.push_back({s.id, s.box, measure(tag, metrics)});
        tags.push_back(std::move(tag));
    }
    const auto counts = countByClass(tracks, classes);
    std::vector<cv::Rect> reserved = {hud(frame, info, counts, options, st, false)};
    const std::vector<cv::Rect> spots = tags_.place(items, frame.size(), reserved);
    // Least important first, so if tags must overlap the important ones stay on top.
    for (size_t i = tags.size(); i-- > 0;) drawTag(frame, tags[i], spots[i], metrics, st);

    hud(frame, info, counts, options, st, true);
    if (options.help) drawHelp(frame, st);
    else drawHint(frame, info.uptime, st);
}
