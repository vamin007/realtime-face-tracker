// The last stage of the pipeline (camera thread -> detectors -> tracker -> targeting/recognition
// -> overlay/log): a CSV record of what the tracker showed. Each run writes a new timestamped
// file under logs/, with one row per visible track per frame, so a session can be analyzed
// afterwards (scripts/log_stats.py) without re-running the models.
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
    // Creates logDir if needed and opens a new file there, run_<YYYYmmdd_HHMMSS>.csv (local
    // time), and writes the header row. Throws std::runtime_error if the file can't be opened.
    explicit Logger(const std::string& logDir);
    // timeMs: the frame's time (camera: since start; video: position in the file).
    // targeting and recognizer are null when those features are off.
    // tracks: normally the tracks being shown, so the log matches the screen. Columns are
    // written in the order of the header in Logger.cpp; box coordinates are whole frame pixels.
    // Call once per frame, from the processing thread.
    void log(long long frameIndex, double timeMs, const std::vector<Track>& tracks,
             const ClassRegistry& classes, const TargetSelector* targeting,
             const Recognizer* recognizer);
    // The CSV file being written.
    const std::string& path() const { return path_; }

private:
    std::ofstream out_;
    std::string path_;
    // Frame time (ms) of the last flush to disk; see log().
    double lastFlushMs_ = 0;
};
