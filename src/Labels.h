#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "Embedder.h"

// One photo of a labeled item.
struct LabelSample {
    std::string image;     // file name in the label's folder
    cv::Rect roi;          // where the item is in that photo
    bool flipped = false;  // the left-right mirrored copy (the live camera view is mirrored)
    Embedding embedding;
};

// A specific item the tracker was taught, e.g. "my_mug": one particular cup.
struct CustomLabel {
    std::string name;
    std::string baseClass;             // the detector class it refines ("cup", "face")
    std::string model;                 // the embedding model its samples were made with
    float threshold = 0;               // similarity needed to recognize it; 0 = the model's default
    std::vector<LabelSample> samples;  // several photos: more angles and light, better recognition
};

struct LabelMatch {
    int label = -1;      // index into LabelLibrary::labels(); -1 if nothing matched
    float score = 0;     // the best label's score
    float runnerUp = 0;  // the next best label's score
};

// Labels on disk: <dir>/<name>/label.yml next to that label's photos.
class LabelLibrary {
public:
    explicit LabelLibrary(std::string dir) : dir_(std::move(dir)) {}

    void load();  // reads every <dir>/*/label.yml
    void save(const CustomLabel& label) const;
    const std::string& dir() const { return dir_; }
    std::string folder(const std::string& name) const { return dir_ + "/" + name; }

    std::vector<CustomLabel>& labels() { return labels_; }
    const std::vector<CustomLabel>& labels() const { return labels_; }
    int find(const std::string& name) const;  // -1 if unknown
    bool hasLabelsFor(const std::string& baseClass) const;

    // The best label among those refining `baseClass`. A label's score is the
    // mean of its 3 most similar samples; the best must reach its threshold and
    // beat the runner-up by at least `minMargin`.
    LabelMatch match(const std::string& baseClass, const Embedding& e, float defaultThreshold,
                     float minMargin) const;

private:
    std::string dir_;
    std::vector<CustomLabel> labels_;
};

// Score of one label for an embedding: the mean of its 3 most similar samples.
float labelScore(const CustomLabel& label, const Embedding& e);

// Label names become class names, so they're limited to letters, digits, _ and -.
bool validLabelName(const std::string& name);
