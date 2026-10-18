#include "Detector.h"

#include <chrono>
#include <filesystem>
#include <future>
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
    struct Output {
        std::vector<Detection> detections;
        double ms = 0;
    };
    auto runOne = [&frame](IDetector* detector) {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        Output out{detector->detect(frame)};
        out.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return out;
    };

    std::vector<IDetector*> running;
    for (const auto& detector : detectors_) {
        bool wanted = allowed.empty();
        for (ClassId id : allowed) wanted = wanted || detector->canDetect(id);
        if (wanted) running.push_back(detector.get());
    }

    // Each detector owns its model and only reads the frame, so they can run side
    // by side: the frame then costs the slowest detector instead of the sum.
    std::vector<Output> outputs(running.size());
    if (parallel_) {
        std::vector<std::future<Output>> pending;
        for (size_t i = 1; i < running.size(); ++i)
            pending.push_back(std::async(std::launch::async, runOne, running[i]));
        if (!running.empty()) outputs[0] = runOne(running[0]);
        for (size_t i = 1; i < running.size(); ++i) outputs[i] = pending[i - 1].get();
    } else {
        for (size_t i = 0; i < running.size(); ++i) outputs[i] = runOne(running[i]);
    }

    std::vector<Detection> result;
    timings_.clear();
    for (size_t i = 0; i < running.size(); ++i) {
        for (Detection& d : outputs[i].detections)
            if (allowed.empty() || allowed.count(d.classId)) result.push_back(std::move(d));
        timings_.push_back({running[i]->name(), outputs[i].ms});
    }
    return result;
}

std::vector<std::string> Detector::summary() const {
    std::vector<std::string> lines;
    for (const auto& detector : detectors_) {
        const std::string info = detector->info();
        lines.push_back(std::string(detector->name()) + (info.empty() ? "" : ": " + info));
    }
    return lines;
}
