#pragma once
#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// Milliseconds per stage of each frame (capture age, detection, tracking, ...),
// summarized at the end of a run as the mean and the 95th percentile.
class LatencyStats {
public:
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
