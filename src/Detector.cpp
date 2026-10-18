#include "Detector.h"

#include <chrono>
#include <filesystem>
#include <stdexcept>

void requireModelFile(const std::string& path) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error("Model file not found: " + path +
                                 " (run scripts/fetch_models.sh)");
}

void Detector::add(std::unique_ptr<IDetector> detector) {
    detectors_.push_back(std::move(detector));
}

std::vector<Detection> Detector::detect(const cv::Mat& frame, const std::set<ClassId>& allowed) {
    using Clock = std::chrono::steady_clock;
    std::vector<Detection> result;
    timings_.clear();
    for (const auto& detector : detectors_) {
        bool wanted = allowed.empty();
        for (ClassId id : allowed) wanted = wanted || detector->canDetect(id);
        if (!wanted) continue;

        const auto t0 = Clock::now();
        for (Detection& d : detector->detect(frame))
            if (allowed.empty() || allowed.count(d.classId)) result.push_back(std::move(d));
        timings_.push_back({detector->name(),
                            std::chrono::duration<double, std::milli>(Clock::now() - t0).count()});
    }
    return result;
}
