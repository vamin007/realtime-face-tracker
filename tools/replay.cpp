// Re-runs the tracker on detections saved with `tracker --dump-detections` and
// prints how stable the tracks were. It skips the neural networks, so trying a
// tracker setting on a whole clip takes well under a second.
//
// Usage: tracker_replay DETECTIONS.csv [--tracks] [--samples PATH] [name=value ...]
//   --tracks        list every confirmed track
//   --samples PATH  write every shown box per frame to a CSV (for analysis)
//   name=value      override a TrackerParams field, e.g. lostTimeout=1.5
//
// Besides IDs it reports, per class:
//   flicker: shown boxes that disappear and come back (gaps), switch to a thin
//            predicted box (thin), change class (cls) or number (num); wobble
//            (jitter: the second difference of the box center, median and 90th
//            percentile; slow90: only while it moves under 1 box size per second)
//            and frame-to-frame width changes; how long a box stays up after the
//            last detection when the object then stays gone for over a second
//            (ghost ms);
//   lag:     how far the shown box is from the detection it was matched to,
//            overall and while the object moves fast, and how far behind the
//            object it trails then (ms); while it isn't detected (coast), how far
//            the box is from the straight path between the detections around it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ClassRegistry.h"
#include "Tracker.h"

namespace {

struct Frame {
    double timeMs = 0;
    std::vector<Detection> detections;
};

std::vector<Frame> readDetections(const std::string& path, const ClassRegistry& classes) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Can't open " + path);
    std::string line;
    std::getline(in, line);  // header: frame,t_ms,class,conf,weak,x,y,w,h,keypoints
    std::vector<Frame> frames;
    double msPerFrame = 0;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::vector<std::string> f;
        for (std::string field; std::getline(ss, field, ',');) f.push_back(field);
        if (f.size() < 9) continue;
        const size_t frame = std::stoul(f[0]);
        if (frame >= frames.size()) frames.resize(frame + 1);
        frames[frame].timeMs = std::stod(f[1]);
        if (frame > 0) msPerFrame = frames[frame].timeMs / frame;
        Detection d;
        d.classId = classes.find(f[2]);
        if (d.classId < 0) continue;  // a class it doesn't track (in dumps made before v3)
        d.conf = std::stof(f[3]);
        d.weak = f[4] == "1";
        d.box = {std::stof(f[5]), std::stof(f[6]), std::stof(f[7]), std::stof(f[8])};
        if (f.size() > 9) {
            std::stringstream kp(f[9]);
            for (float x, y; kp >> x >> y;) d.keypoints.emplace_back(x, y);
        }
        frames[frame].detections.push_back(d);
    }
    // Frames without detections have no rows; give them their time too.
    for (size_t i = 0; i < frames.size(); ++i)
        if (frames[i].detections.empty()) frames[i].timeMs = i * msPerFrame;
    return frames;
}

bool setParam(TrackerParams& p, const std::string& assignment) {
    const auto eq = assignment.find('=');
    if (eq == std::string::npos) return false;
    const std::string name = assignment.substr(0, eq);
    const double v = std::stod(assignment.substr(eq + 1));
    const std::map<std::string, std::function<void(double)>> setters = {
        {"minHits", [&](double x) { p.minHits = static_cast<int>(x); }},
        {"tentativeTimeout", [&](double x) { p.tentativeTimeout = x; }},
        {"lostTimeout", [&](double x) { p.lostTimeout = x; }},
        {"exitTimeout", [&](double x) { p.exitTimeout = x; }},
        {"maxSizeRatio", [&](double x) { p.maxSizeRatio = static_cast<float>(x); }},
        {"coastTime", [&](double x) { p.coastTime = x; }},
        {"thinAfter", [&](double x) { p.thinAfter = x; }},
        {"weakBridge", [&](double x) { p.weakBridge = x; }},
        {"weakTimeout", [&](double x) { p.weakTimeout = x; }},
        {"weakReacquire", [&](double x) { p.weakReacquire = x; }},
        {"weakBridgeFaceHand", [&](double x) { p.weakBridgeFaceHand = x; }},
        {"velocityHalfLife", [&](double x) { p.velocityHalfLife = x; }},
        {"minIou", [&](double x) { p.minIou = static_cast<float>(x); }},
        {"maxDistance", [&](double x) { p.maxDistance = static_cast<float>(x); }},
        {"lostSearchGrowth", [&](double x) { p.lostSearchGrowth = static_cast<float>(x); }},
        {"maxLostDistance", [&](double x) { p.maxLostDistance = static_cast<float>(x); }},
        {"weakMinIou", [&](double x) { p.weakMinIou = static_cast<float>(x); }},
        {"classChangeCost", [&](double x) { p.classChangeCost = static_cast<float>(x); }},
        {"duplicateIou", [&](double x) { p.duplicateIou = static_cast<float>(x); }},
        {"voteDecay", [&](double x) { p.voteDecay = static_cast<float>(x); }},
        {"classSwitchMargin", [&](double x) { p.classSwitchMargin = static_cast<float>(x); }},
        {"accelPos", [&](double x) { p.kalman.accelPos = static_cast<float>(x); }},
        {"accelSize", [&](double x) { p.kalman.accelSize = static_cast<float>(x); }},
        {"measurePos", [&](double x) { p.kalman.measurePos = static_cast<float>(x); }},
        {"measureSize", [&](double x) { p.kalman.measureSize = static_cast<float>(x); }},
        {"snapStart", [&](double x) { p.kalman.snapStart = static_cast<float>(x); }},
        {"snapFull", [&](double x) { p.kalman.snapFull = static_cast<float>(x); }},
        {"weakNoise", [&](double x) { p.weakNoise = static_cast<float>(x); }},
        {"snapSizeFactor", [&](double x) { p.kalman.snapSizeFactor = static_cast<float>(x); }},
    };
    const auto it = setters.find(name);
    if (it == setters.end()) return false;
    it->second(v);
    return true;
}

// One frame of a visible track.
struct Sample {
    size_t frame;
    double time;  // s
    bool matched, coasting;
    cv::Rect2f box, detected;
    ClassId cls;
    int number;
};

cv::Point2f centerOf(const cv::Rect2f& r) { return {r.x + r.width / 2, r.y + r.height / 2}; }
float sizeOf(const cv::Rect2f& r) { return std::sqrt(std::max(r.area(), 1.0f)); }

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(p / 100 * v.size()))];
}

double mean(const std::vector<double>& v) {
    double sum = 0;
    for (double x : v) sum += x;
    return v.empty() ? NAN : sum / v.size();
}

// Flicker and lag of one class's tracks (see the top of the file).
struct Steadiness {
    long long gaps = 0, thin = 0, classChanges = 0, numberChanges = 0;
    std::vector<double> jitter;              // % of box size
    std::vector<double> slowJitter;          // ... while the box moves slowly (visible as wobble)
    std::vector<double> sizeJitter;          // frame-to-frame change of the width, %
    std::vector<double> slowSizeJitter;      // ... while the box moves slowly
    std::vector<double> offset, fastOffset;  // shown box vs its detection, % of box size
    std::vector<double> trailMs;             // while fast: how far behind, in time
    std::vector<double> coastError;          // predicted box vs the object's path, % of box size
    std::vector<double> ghostMs;  // shown after the last detection, before a long absence

    // Shown boxes after the last detection, before the track disappears for a while.
    void addGhosts(const std::vector<Sample>& s) {
        for (size_t i = 0; i < s.size(); ++i) {
            const bool gone = i + 1 == s.size() || s[i + 1].time - s[i].time > 1.0;
            if (!gone) continue;
            size_t last = i;  // the last frame with a detection
            while (last > 0 && !s[last].matched && s[last - 1].frame + 1 == s[last].frame) --last;
            if (s[last].matched) ghostMs.push_back(1000 * (s[i].time - s[last].time));
        }
    }

    void add(const std::vector<Sample>& s) {
        addGhosts(s);
        // A run of predicted boxes between two detections: the object moved
        // roughly in a straight line from one detection to the next.
        for (size_t i = 0; i + 1 < s.size(); ++i) {
            if (!s[i].matched) continue;
            size_t j = i + 1;
            while (j < s.size() && !s[j].matched && s[j].frame == s[j - 1].frame + 1) ++j;
            if (j == i + 1 || j >= s.size() || !s[j].matched || s[j].frame != s[j - 1].frame + 1)
                continue;
            const cv::Point2f a = centerOf(s[i].detected), b = centerOf(s[j].detected);
            for (size_t k = i + 1; k < j; ++k) {
                const float w =
                    static_cast<float>((s[k].time - s[i].time) / (s[j].time - s[i].time));
                const cv::Point2f err = a + (b - a) * w - centerOf(s[k].box);
                coastError.push_back(100 * std::hypot(err.x, err.y) / sizeOf(s[k].box));
            }
            i = j - 1;
        }
        for (size_t i = 1; i < s.size(); ++i) {
            classChanges += s[i].cls != s[i - 1].cls;
            numberChanges += s[i].number != s[i - 1].number;
            if (s[i].frame != s[i - 1].frame + 1) {
                ++gaps;
                continue;
            }
            thin += !s[i - 1].coasting && s[i].coasting;
            const float widthChange = 100 * std::abs(s[i].box.width - s[i - 1].box.width) /
                                      std::max(s[i - 1].box.width, 1.0f);
            sizeJitter.push_back(widthChange);
            const cv::Point2f d1 = centerOf(s[i].box) - centerOf(s[i - 1].box);
            if (std::hypot(d1.x, d1.y) / (s[i].time - s[i - 1].time) <
                kSlowSizesPerSecond * sizeOf(s[i].box))
                slowSizeJitter.push_back(widthChange);
            if (i >= 2 && s[i - 2].frame + 2 == s[i].frame) {
                const cv::Point2f d2 =
                    centerOf(s[i].box) - 2 * centerOf(s[i - 1].box) + centerOf(s[i - 2].box);
                jitter.push_back(100 * std::hypot(d2.x, d2.y) / sizeOf(s[i - 1].box));
                const cv::Point2f d1 = centerOf(s[i].box) - centerOf(s[i - 2].box);
                const double speed = std::hypot(d1.x, d1.y) / (s[i].time - s[i - 2].time);
                if (speed < kSlowSizesPerSecond * sizeOf(s[i - 1].box))
                    slowJitter.push_back(jitter.back());
            }
        }
        for (size_t i = 0; i < s.size(); ++i) {
            if (!s[i].matched) continue;
            const float size = sizeOf(s[i].detected);
            const cv::Point2f off = centerOf(s[i].detected) - centerOf(s[i].box);
            offset.push_back(100 * std::hypot(off.x, off.y) / size);
            // The object's speed, from this and the previous frame's detections.
            if (i == 0 || !s[i - 1].matched || s[i - 1].frame + 1 != s[i].frame) continue;
            const double dt = s[i].time - s[i - 1].time;
            if (dt <= 0) continue;
            const cv::Point2f v =
                (centerOf(s[i].detected) - centerOf(s[i - 1].detected)) / static_cast<float>(dt);
            const double speed = std::hypot(v.x, v.y);
            if (speed < kFastSizesPerSecond * size) continue;
            fastOffset.push_back(offset.back());
            trailMs.push_back(1000 * (off.x * v.x + off.y * v.y) / (speed * speed));
        }
    }
    // Moving at least this many box sizes per second counts as fast (a waved
    // hand); detection noise alone stays well below it.
    static constexpr double kFastSizesPerSecond = 4.0;
    // Moving less than this: wobble shows (jitter while slow).
    static constexpr double kSlowSizesPerSecond = 1.0;
};

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::cerr << "Usage: tracker_replay DETECTIONS.csv [--tracks] [--samples PATH] "
                     "[name=value ...]\n";
        return 1;
    }
    const ClassRegistry classes;
    TrackerParams params;
    bool listTracks = false;
    std::string samplesPath;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tracks") listTracks = true;
        else if (arg == "--samples" && i + 1 < argc) samplesPath = argv[++i];
        else if (!setParam(params, arg)) throw std::runtime_error("Unknown setting: " + arg);
    }

    const std::vector<Frame> frames = readDetections(argv[1], classes);
    Tracker tracker(classes, params);
    tracker.setFrameSize({1280, 720});  // the reference clips

    struct Life {
        ClassId cls;
        size_t first, last, visibleFrames = 0;
        cv::Rect2f firstBox, lastBox;
    };
    std::map<int, Life> lives;              // by track id
    std::map<int, std::vector<Sample>> samples;  // by track id
    std::map<ClassId, long long> detected;  // confident detections per class
    std::map<ClassId, long long> covered;   // ... that had a visible track of the class
    for (size_t f = 0; f < frames.size(); ++f) {
        tracker.update(frames[f].detections, frames[f].timeMs / 1000.0);
        std::map<ClassId, int> dets, shown;
        for (const Detection& d : frames[f].detections)
            if (!d.weak) ++dets[d.classId];
        for (const Track& t : tracker.tracks()) {
            if (!t.visible) continue;
            ++shown[t.classId];
            auto [it, isNew] = lives.try_emplace(t.id, Life{t.classId, f, f, 0, t.box, t.box});
            it->second.last = f;
            it->second.lastBox = t.box;
            ++it->second.visibleFrames;
            samples[t.id].push_back({f, frames[f].timeMs / 1000.0, t.matched, t.coasting, t.box,
                                     t.detected, t.classId, t.number});
        }
        for (const auto& [cls, n] : dets) {
            detected[cls] += n;
            covered[cls] += std::min(n, shown[cls]);
        }
    }

    std::map<ClassId, std::vector<size_t>> byClass;  // visible frames per track
    for (const auto& [id, life] : lives) byClass[life.cls].push_back(life.visibleFrames);
    std::printf("%zu frames\n%-14s%5s%7s%13s%10s\n", frames.size(), "class", "IDs", "short",
                "median life", "covered");
    for (const auto& [cls, v] : byClass) {
        std::vector<size_t> sorted = v;
        std::sort(sorted.begin(), sorted.end());
        const long long shortOnes =
            std::count_if(sorted.begin(), sorted.end(), [](size_t n) { return n <= 5; });
        const double cover = detected[cls] ? 100.0 * covered[cls] / detected[cls] : 0;
        std::printf("%-14s%5zu%7lld%13zu%9.1f%%\n", classes.name(cls).c_str(), sorted.size(),
                    shortOnes, sorted[sorted.size() / 2], cover);
    }

    // Flicker and lag, per minute of the clip where counted.
    std::map<ClassId, Steadiness> steadiness;
    for (const auto& [id, s] : samples) steadiness[lives[id].cls].add(s);
    const double minutes =
        frames.size() > 1 ? (frames.back().timeMs - frames.front().timeMs) / 60000.0 : 1.0;
    std::printf(
        "\nFlicker (counts per minute; jitter and size changes in %% of the box size)\n"
        "%-12s%6s%6s%6s%6s |%7s%7s%7s |%7s%7s |%9s\n",
        "class", "gaps", "thin", "cls", "num", "jit50", "jit90", "slow90", "size90", "slow90",
        "ghost ms");
    for (const auto& [cls, s] : steadiness)
        std::printf("%-12s%6.1f%6.1f%6.1f%6.1f |%7.1f%7.1f%7.1f |%7.1f%7.1f |%9.0f\n",
                    classes.name(cls).c_str(), s.gaps / minutes, s.thin / minutes,
                    s.classChanges / minutes, s.numberChanges / minutes, percentile(s.jitter, 50),
                    percentile(s.jitter, 90), percentile(s.slowJitter, 90),
                    percentile(s.sizeJitter, 90), percentile(s.slowSizeJitter, 90),
                    mean(s.ghostMs));
    std::printf(
        "\nLag (shown box vs detection, %% of the box size)\n"
        "%-12s%7s%7s |%7s%7s%7s%9s |%7s%7s%7s\n",
        "class", "mean", "p90", "fast", "mean", "p90", "trail ms", "coast", "mean", "p90");
    for (const auto& [cls, s] : steadiness)
        std::printf("%-12s%7.1f%7.1f |%7zu%7.1f%7.1f%9.0f |%7zu%7.1f%7.1f\n",
                    classes.name(cls).c_str(), mean(s.offset), percentile(s.offset, 90),
                    s.fastOffset.size(), mean(s.fastOffset), percentile(s.fastOffset, 90),
                    percentile(s.trailMs, 50), s.coastError.size(), mean(s.coastError),
                    percentile(s.coastError, 90));
    auto describe = [](const cv::Rect2f& b) {
        char s[64];
        std::snprintf(s, sizeof(s), "(%4.0f,%4.0f) %3.0fx%-3.0f", b.x + b.width / 2,
                      b.y + b.height / 2, b.width, b.height);
        return std::string(s);
    };
    if (listTracks)
        for (const auto& [id, life] : lives)
            std::printf("  #%-4d %-12s frames %5zu-%-5zu visible %-5zu %s -> %s\n", id,
                        classes.name(life.cls).c_str(), life.first, life.last, life.visibleFrames,
                        describe(life.firstBox).c_str(), describe(life.lastBox).c_str());
    if (!samplesPath.empty()) {
        std::ofstream out(samplesPath);
        if (!out) throw std::runtime_error("Can't write " + samplesPath);
        out << "id,class,frame,t_ms,matched,coasting,x,y,w,h,det_x,det_y,det_w,det_h\n";
        for (const auto& [id, s] : samples)
            for (const Sample& k : s)
                out << id << ',' << classes.name(lives[id].cls) << ',' << k.frame << ','
                    << k.time * 1000 << ',' << k.matched << ',' << k.coasting << ',' << k.box.x
                    << ',' << k.box.y << ',' << k.box.width << ',' << k.box.height << ','
                    << k.detected.x << ',' << k.detected.y << ',' << k.detected.width << ','
                    << k.detected.height << '\n';
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}
