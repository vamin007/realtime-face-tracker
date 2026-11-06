// Detector: runs the YOLO and face detectors on each frame, in parallel by default, and merges
// their detections for the tracker. See Detector.h for the role in the pipeline.
#include "Detector.h"

#include <chrono>
#include <filesystem>
#include <future>
#include <stdexcept>

// Declared in IDetector.h so any detector can use it for a clear "run fetch_models.sh" message.
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
    // Runs one detector and times it. Safe to call from any thread: it touches only its own
    // detector and reads the shared frame.
    auto runOne = [&frame](IDetector* detector) {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        Output out{detector->detect(frame)};
        out.ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return out;
    };

    // Pick the detectors worth running: with --classes face, YOLO (people) isn't run at all.
    std::vector<IDetector*> running;
    for (const auto& detector : detectors_) {
        bool wanted = allowed.empty();
        for (ClassId id : allowed) wanted = wanted || detector->canDetect(id);
        if (wanted) running.push_back(detector.get());
    }

    // Each detector owns its model and only reads the frame, so they can run side
    // by side: the frame then costs the slowest detector instead of the sum.
    // (The first detector runs on this thread, the others on std::async threads, so with two
    // detectors only one thread is created per frame.)
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

    // Merge in detector order, keeping only the allowed classes (a detector that handles
    // several classes may produce some that weren't asked for).
    std::vector<Detection> result;
    timings_.clear();
    for (size_t i = 0; i < running.size(); ++i) {
        for (Detection& d : outputs[i].detections)
            if (allowed.empty() || allowed.count(d.classId)) result.push_back(std::move(d));
        timings_.push_back({running[i]->name(), outputs[i].ms});
    }
    return result;
}

// Feeds the "Detectors (...)" lines main() prints at startup, e.g. "yolo: 640x384, CoreML".
std::vector<std::string> Detector::summary() const {
    std::vector<std::string> lines;
    for (const auto& detector : detectors_) {
        const std::string info = detector->info();
        lines.push_back(std::string(detector->name()) + (info.empty() ? "" : ": " + info));
    }
    return lines;
}

// Lets main() check that the face detector loaded before it looks at label photos.
std::vector<std::string> Detector::names() const {
    std::vector<std::string> names;
    for (const auto& detector : detectors_) names.push_back(detector->name());
    return names;
}
