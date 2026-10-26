#include "Logger.h"

#include <ctime>
#include <filesystem>
#include <stdexcept>

#include "Overlay.h"

Logger::Logger(const std::string& logDir) {
    std::filesystem::create_directories(logDir);

    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    path_ = logDir + "/run_" + stamp + ".csv";

    out_.open(path_);
    if (!out_) throw std::runtime_error("Failed to open log file: " + path_);
    out_ << "time_ms,frame,id,number,class,label,matched,x,y,w,h,conf,target,laser,"
            "coasting\n";
}

void Logger::log(long long frameIndex, double timeMs, const std::vector<Track>& tracks,
                 const ClassRegistry& classes, const TargetSelector* targeting,
                 const Recognizer* recognizer) {
    for (const Track& t : tracks) {
        if (!isVisible(t)) continue;
        const bool isTarget = targeting && targeting->target().trackId == t.id;
        const std::string* label = recognizer ? recognizer->labelName(t.id) : nullptr;
        out_ << static_cast<long long>(timeMs) << ',' << frameIndex << ',' << t.id << ','
             << t.number << ',' << classes.name(t.classId) << ',' << (label ? *label : "") << ','
             << (t.matched ? 1 : 0) << ',' << static_cast<int>(t.box.x) << ','
             << static_cast<int>(t.box.y) << ',' << static_cast<int>(t.box.width) << ','
             << static_cast<int>(t.box.height) << ',' << t.conf << ',' << (isTarget ? 1 : 0) << ',';
        if (targeting) {
            const auto it = targeting->faces().find(t.id);
            if (it != targeting->faces().end() && it->second.dot.found)
                out_ << it->second.dot.redness;
        }
        out_ << ',' << (t.coasting ? 1 : 0) << '\n';
    }
    // Flush about once a second so a crash loses little.
    if (timeMs - lastFlushMs_ >= 1000) {
        out_.flush();
        lastFlushMs_ = timeMs;
    }
}
