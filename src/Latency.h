// Per-stage latency bookkeeping for the main loop. main() times each pipeline stage (capture
// age, detect, track, labels+target, draw, display, capture to screen) every frame and adds the
// numbers here; the end-of-run summary shows where the time goes. The mean says what a typical
// frame costs; the 95th percentile says how bad the slow frames are (the ones that feel like
// stutter), which a mean alone hides.
#pragma once
#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// Milliseconds per stage of each frame (capture age, detection, tracking, ...),
// summarized at the end of a run as the mean and the 95th percentile.
// Keeps every sample (a few doubles per frame), so it suits runs of minutes, not days.
// Not thread-safe: only the main loop uses it.
class LatencyStats {
public:
    // Records one frame's time for `stage`. Stages are created on first use.
    void add(const std::string& stage, double ms) {
        auto it = std::find_if(stages_.begin(), stages_.end(),
                               [&](const auto& s) { return s.first == stage; });
        if (it == stages_.end()) it = stages_.insert(stages_.end(), {stage, {}});
        it->second.push_back(ms);
    }

    // "stage mean/p95, ..." in the order the stages were first added.
    std::string summary() const {
        std::string text;
        for (const auto& [stage, values] : stages_) {
            std::vector<double> sorted = values;
            std::sort(sorted.begin(), sorted.end());
            double sum = 0;
            for (double v : sorted) sum += v;
            // 95th percentile by index into the sorted samples (clamped for tiny sample counts).
            const double p95 = sorted[std::min(sorted.size() - 1, sorted.size() * 95 / 100)];
            char buf[96];
            std::snprintf(buf, sizeof(buf), "%s%s %.1f/%.1f", text.empty() ? "" : ", ",
                          stage.c_str(), sum / sorted.size(), p95);
            text += buf;
        }
        return text;
    }

private:
    std::vector<std::pair<std::string, std::vector<double>>> stages_;
};
