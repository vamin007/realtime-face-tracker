#pragma once
#include <opencv2/ml.hpp>
#include <string>
#include <vector>

#include "Embedder.h"

// What the face network learns from: each label's samples, and faces of
// people who are none of the labels.
struct FaceNetData {
    std::string model;                            // the embedding model they're from
    std::vector<std::string> labels;              // label names: one output each, in this order
    std::vector<std::vector<Embedding>> samples;  // each label's samples (photos, mirrored too)
    std::vector<Embedding> others;                // other people's faces: the "other" output
    int otherPeople = 0;  // how many separate faces (tracks) the others came from
    // Changes whenever any of the above does.
    std::string fingerprint() const;
};

// A small neural network on face embeddings that learns the labeled people:
// 128 inputs (an SFace embedding), 32 hidden units, and one output per label
// plus one for "someone else". The gallery rule (LabelLibrary::match) compares
// a face with each photo on its own; the network learns from all of a label's
// photos at once, and from other people's faces, so it can be surer than the
// photos alone, either way. A label's samples count against every other
// output, so the labels are also each other's negatives.
//
// It's OpenCV's ANN_MLP, trained by least squares on one-hot targets, so each
// output estimates that answer's probability. Each label, and "other", weighs
// the same in training however many samples it has. With so little data one
// network's answers vary with its starting weights (and a run now and then
// never learns one of the outputs), so three are trained from different
// starting weights, each must fit its own training samples, and their answers
// are averaged. On synthetic people that took confident answers for people
// never seen from 12% to 0.4%. Training is seeded: the same data gives the
// same networks.
//
// It's only trained with enough data: 3 photos of every label, and 20 faces of
// at least 5 other people. Without them it would call anyone it never saw one
// of the labels (it has to answer something), so it stays off and recognition
// uses the photos alone.
class FaceNet {
public:
    static constexpr int kHidden = 32;
    static constexpr int kNets = 3;            // trained from different starting weights
    static constexpr int kMinSamples = 6;      // of each label: 3 photos, each mirrored
    static constexpr int kMinOthers = 20;      // faces of other people ...
    static constexpr int kMinOtherPeople = 5;  // ... from this many separate faces (tracks)

    // Why there's too little data to train on, e.g. "needs other faces (3 of
    // 20 seen)"; empty if there's enough.
    static std::string shortfall(const FaceNetData& data);

    // Trains new networks on `data`. Returns false (and stays off) if it's too
    // little, or if they can't learn it (two labels of the same face, say).
    bool train(const FaceNetData& data);
    bool on() const { return !nets_.empty(); }

    // The probability of each label (in labels() order), then of "other",
    // summing to 1. Empty when it's off or the embedding doesn't fit.
    std::vector<float> predict(const Embedding& e) const;

    // Trained for these labels (in this order) and embedding model: its outputs can be used.
    bool fits(const std::vector<std::string>& labels, const std::string& model) const;
    // ... and on exactly this data: it needn't be trained again.
    bool trainedOn(const FaceNetData& data) const;

    const std::vector<std::string>& labels() const { return labels_; }
    const std::string& model() const { return model_; }
    int samples() const { return samples_; }          // label samples it was trained on
    int others() const { return others_; }            // ... other faces
    int otherPeople() const { return otherPeople_; }  // ... and how many tracks they came from

    // labels/.face_net.yml: the labels, the model, what it was trained on and
    // the network. Written next to the file and renamed over it. Throws if it can't.
    void save(const std::string& file) const;
    // Returns false (and is off) if the file is missing or unreadable.
    bool load(const std::string& file);

private:
    std::vector<cv::Ptr<cv::ml::ANN_MLP>> nets_;
    std::vector<std::string> labels_;
    std::string model_;
    std::string fingerprint_;
    int samples_ = 0;
    int others_ = 0;
    int otherPeople_ = 0;
};

// In the labels directory, written by the tracker.
constexpr const char* kFaceNetFile = ".face_net.yml";
