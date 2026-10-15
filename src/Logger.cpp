#include "Logger.h"

#include <chrono>
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
    out_ << "timestamp_ms,frame,id,class,x,y,w,h,conf\n";
}

void Logger::log(long long frameIndex, const std::vector<Track>& tracks,
                 const std::vector<std::string>& classNames) {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch())
                        .count();
    for (const Track& t : tracks) {
        if (!isVisible(t)) continue;
        out_ << ms << ',' << frameIndex << ',' << t.id << ',' << classNames[t.classId] << ','
             << static_cast<int>(t.box.x) << ',' << static_cast<int>(t.box.y) << ','
             << static_cast<int>(t.box.width) << ',' << static_cast<int>(t.box.height) << ','
             << t.conf << '\n';
    }
}
