// Re-runs the tracker on detections saved with `tracker --dump-detections` and
// prints how stable the tracks were. It skips the neural networks, so trying a
// tracker setting on a whole clip takes well under a second.
//
// Usage: tracker_replay DETECTIONS.csv [--tracks] [name=value ...]
//   --tracks      list every confirmed track
//   name=value    override a TrackerParams field, e.g. lostTimeout=1.5

#include <algorithm>
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
        {"weakBridge", [&](double x) { p.weakBridge = x; }},
        {"weakTimeout", [&](double x) { p.weakTimeout = x; }},
        {"velocityHalfLife", [&](double x) { p.velocityHalfLife = x; }},
        {"minIou", [&](double x) { p.minIou = static_cast<float>(x); }},
        {"maxDistance", [&](double x) { p.maxDistance = static_cast<float>(x); }},
        {"lostSearchGrowth", [&](double x) { p.lostSearchGrowth = static_cast<float>(x); }},
        {"maxLostDistance", [&](double x) { p.maxLostDistance = static_cast<float>(x); }},
        {"weakMinIou", [&](double x) { p.weakMinIou = static_cast<float>(x); }},
        {"classChangeCost", [&](double x) { p.classChangeCost = static_cast<float>(x); }},
        {"duplicateIou", [&](double x) { p.duplicateIou = static_cast<float>(x); }},
        {"accelPos", [&](double x) { p.kalman.accelPos = static_cast<float>(x); }},
        {"accelSize", [&](double x) { p.kalman.accelSize = static_cast<float>(x); }},
        {"measurePos", [&](double x) { p.kalman.measurePos = static_cast<float>(x); }},
        {"measureSize", [&](double x) { p.kalman.measureSize = static_cast<float>(x); }},
    };
    const auto it = setters.find(name);
    if (it == setters.end()) return false;
    it->second(v);
    return true;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::cerr << "Usage: tracker_replay DETECTIONS.csv [--tracks] [name=value ...]\n";
        return 1;
    }
    const ClassRegistry classes = ClassRegistry::fromFile(PROJECT_DIR "/models/coco.names");
    TrackerParams params;
    bool listTracks = false;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tracks") listTracks = true;
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
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
}
