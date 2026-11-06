// Logger: the CSV run log. See Logger.h for the role in the pipeline. One row per track the
// overlay would draw (isVisible), so the file records exactly what was shown.
#include "Logger.h"

#include <ctime>
#include <filesystem>
#include <stdexcept>

#include "Overlay.h"

Logger::Logger(const std::string& logDir) {
    std::filesystem::create_directories(logDir);

    // One file per run, named by the start time, so runs never overwrite each other.
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    path_ = logDir + "/run_" + stamp + ".csv";

    out_.open(path_);
    if (!out_) throw std::runtime_error("Failed to open log file: " + path_);
    // Columns: time_ms and frame identify the frame; id is unique for the run, number is the
    // per-class number shown on screen; label is the recognized person's name (empty if none);
    // matched is 1 when a detection matched the track this frame; x,y,w,h is the box;
    // target is 1 for the laser TARGET; laser is the redness of the laser dot found on this
    // face (empty if none); coasting is 1 when the box is only a prediction.
    out_ << "time_ms,frame,id,number,class,label,matched,x,y,w,h,conf,target,laser,"
            "coasting\n";
}

void Logger::log(long long frameIndex, double timeMs, const std::vector<Track>& tracks,
                 const ClassRegistry& classes, const TargetSelector* targeting,
                 const Recognizer* recognizer) {
    for (const Track& t : tracks) {
        // Unconfirmed tracks and ones missing too long aren't drawn, so they aren't logged.
        if (!isVisible(t)) continue;
        const bool isTarget = targeting && targeting->target().trackId == t.id;
        const std::string* label = recognizer ? recognizer->labelName(t.id) : nullptr;
        out_ << static_cast<long long>(timeMs) << ',' << frameIndex << ',' << t.id << ','
             << t.number << ',' << classes.name(t.classId) << ',' << (label ? *label : "") << ','
             << (t.matched ? 1 : 0) << ',' << static_cast<int>(t.box.x) << ','
             << static_cast<int>(t.box.y) << ',' << static_cast<int>(t.box.width) << ','
             << static_cast<int>(t.box.height) << ',' << t.conf << ',' << (isTarget ? 1 : 0) << ',';
        // The laser column is left empty unless a dot was found on this face in this frame.
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
