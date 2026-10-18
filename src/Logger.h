#pragma once
#include <fstream>
#include <string>
#include <vector>

#include "ClassRegistry.h"
#include "Tracker.h"

// Writes one CSV row per visible track per frame.
class Logger {
public:
    explicit Logger(const std::string& logDir);
    void log(long long frameIndex, const std::vector<Track>& tracks, const ClassRegistry& classes);
    const std::string& path() const { return path_; }

private:
    std::ofstream out_;
    std::string path_;
};
