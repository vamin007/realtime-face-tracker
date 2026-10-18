#pragma once
#include <fstream>
#include <string>
#include <vector>

#include "ClassRegistry.h"
#include "Recognizer.h"
#include "Targeting.h"
#include "Tracker.h"

// Writes one CSV row per visible track per frame.
class Logger {
public:
    explicit Logger(const std::string& logDir);
    // timeMs: the frame's time (camera: since start; video: position in the file).
    // targeting and recognizer are null when those features are off.
    void log(long long frameIndex, double timeMs, const std::vector<Track>& tracks,
             const ClassRegistry& classes, const TargetSelector* targeting,
             const Recognizer* recognizer);
    const std::string& path() const { return path_; }

private:
    std::ofstream out_;
    std::string path_;
    double lastFlushMs_ = 0;
};
