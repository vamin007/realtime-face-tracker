#include "Labels.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

constexpr int kTopSamples = 3;

}  // namespace

bool validLabelName(const std::string& name) {
    if (name.empty() || name.size() > 40) return false;
    return std::all_of(name.begin(), name.end(), [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-';
    });
}

float labelScore(const CustomLabel& label, const Embedding& e) {
    std::vector<float> sims;
    for (const LabelSample& s : label.samples) sims.push_back(similarity(s.embedding, e));
    if (sims.empty()) return 0;
    const size_t k = std::min<size_t>(kTopSamples, sims.size());
    std::partial_sort(sims.begin(), sims.begin() + k, sims.end(), std::greater<float>());
    float sum = 0;
    for (size_t i = 0; i < k; ++i) sum += sims[i];
    return sum / k;
}

void LabelLibrary::load() {
    labels_.clear();
    if (!fs::exists(dir_)) return;
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir_))
        if (entry.is_directory() && fs::exists(entry.path() / "label.yml"))
            files.push_back(entry.path() / "label.yml");
    std::sort(files.begin(), files.end());
    for (const fs::path& file : files) {
        cv::FileStorage in(file.string(), cv::FileStorage::READ);
        if (!in.isOpened()) {
            std::cerr << "Warning: can't read " << file << '\n';
            continue;
        }
        CustomLabel label;
        in["name"] >> label.name;
        in["base_class"] >> label.baseClass;
        in["model"] >> label.model;
        in["threshold"] >> label.threshold;
        for (const cv::FileNode& node : in["samples"]) {
            LabelSample s;
            node["image"] >> s.image;
            node["roi"] >> s.roi;
            s.flipped = static_cast<int>(node["flipped"]) != 0;
            cv::Mat m;
            node["embedding"] >> m;
            s.embedding.assign(m.begin<float>(), m.end<float>());
            label.samples.push_back(std::move(s));
        }
        if (label.name.empty() || label.samples.empty()) {
            std::cerr << "Warning: skipping incomplete label " << file << '\n';
            continue;
        }
        labels_.push_back(std::move(label));
    }
}

void LabelLibrary::save(const CustomLabel& label) const {
    fs::create_directories(folder(label.name));
    cv::FileStorage out(folder(label.name) + "/label.yml", cv::FileStorage::WRITE);
    if (!out.isOpened()) throw std::runtime_error("Can't write label " + label.name);
    out << "name" << label.name << "base_class" << label.baseClass << "model" << label.model
        << "threshold" << label.threshold;
    out << "samples" << "[";
    for (const LabelSample& s : label.samples) {
        out << "{" << "image" << s.image << "roi" << s.roi << "flipped" << (s.flipped ? 1 : 0)
            << "embedding" << cv::Mat(s.embedding, false) << "}";
    }
    out << "]";
}

int LabelLibrary::find(const std::string& name) const {
    for (size_t i = 0; i < labels_.size(); ++i)
        if (labels_[i].name == name) return static_cast<int>(i);
    return -1;
}

bool LabelLibrary::hasLabelsFor(const std::string& baseClass) const {
    return std::any_of(labels_.begin(), labels_.end(),
                       [&](const CustomLabel& l) { return l.baseClass == baseClass; });
}

LabelMatch LabelLibrary::match(const std::string& baseClass, const Embedding& e,
                               float defaultThreshold, float minMargin) const {
    LabelMatch best;
    float bestScore = -1;
    for (size_t i = 0; i < labels_.size(); ++i) {
        if (labels_[i].baseClass != baseClass) continue;
        const float score = labelScore(labels_[i], e);
        if (score > bestScore) {
            best.runnerUp = std::max(bestScore, 0.0f);
            bestScore = score;
            best.label = static_cast<int>(i);
        } else {
            best.runnerUp = std::max(best.runnerUp, score);
        }
    }
    if (best.label < 0) return {};
    best.score = bestScore;
    const CustomLabel& label = labels_[best.label];
    const float threshold = label.threshold > 0 ? label.threshold : defaultThreshold;
    if (best.score < threshold || best.score - best.runnerUp < minMargin) best.label = -1;
    return best;
}
