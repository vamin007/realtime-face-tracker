// FaceNet training, prediction and saving. Why the training looks as it does (noisy copies, three
// networks, how a network is judged to have learned) is explained where it happens in train().

#include "FaceNet.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

// Version of .face_net.yml; a file of another version is ignored and the network trained again.
constexpr int kFormat = 1;
// RPROP (resilient backpropagation) updates each weight by a step that depends only on the sign
// of its gradient, so there's no learning rate to tune.
constexpr int kIterations = 300;      // RPROP steps at most; it converges in far fewer
constexpr uint64_t kSeed = 20261104;  // of the noise, so training is repeatable
// Each sample is also learned from in noisy copies (noise of this length; the
// embedding is 1 long). With tens of samples and some 4000 weights, a network
// would otherwise learn the samples by heart and answer anything near them at
// random; with the copies it learns their neighbourhood (on synthetic people,
// held-out views went from 94% to 97-100% right).
constexpr int kCopies = 4;
constexpr float kNoise = 0.5f;
constexpr float kLearned = 0.9f;  // share of an output's own samples a network must get right
constexpr int kAttempts = 3;      // tries per network (new noise each time)

// FNV-1a, 64 bits: a simple non-cryptographic hash, enough to notice that the training data
// changed.
struct Hash {
    uint64_t h = 1469598103934665603ULL;
    void bytes(const void* data, size_t n) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ULL;
    }
    void text(const std::string& s) {
        const uint64_t n = s.size();
        bytes(&n, sizeof(n));
        bytes(s.data(), s.size());
    }
    // Rounded, so an embedding read back from a file hashes the same.
    void embeddings(const std::vector<Embedding>& es) {
        const uint64_t n = es.size();
        bytes(&n, sizeof(n));
        for (const Embedding& e : es)
            for (float x : e) {
                const int32_t q = static_cast<int32_t>(std::lround(x * 1e5f));
                bytes(&q, sizeof(q));
            }
    }
};

}  // namespace

std::string FaceNetData::fingerprint() const {
    Hash hash;
    hash.text(model);
    for (const std::string& name : labels) hash.text(name);
    for (const std::vector<Embedding>& s : samples) hash.embeddings(s);
    hash.embeddings(others);
    const int64_t people = otherPeople;
    hash.bytes(&people, sizeof(people));
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash.h));
    return text;
}

std::string FaceNet::shortfall(const FaceNetData& data) {
    if (data.labels.empty()) return "needs a label";
    // Every label needs kMinSamples samples, i.e. kMinSamples / 2 photos (each is also mirrored).
    for (size_t i = 0; i < data.labels.size(); ++i)
        if (i >= data.samples.size() || static_cast<int>(data.samples[i].size()) < kMinSamples)
            return "needs " + std::to_string(kMinSamples / 2) + " usable photos of " +
                   data.labels[i];
    if (static_cast<int>(data.others.size()) < kMinOthers)
        return "needs other faces (" + std::to_string(data.others.size()) + " of " +
               std::to_string(kMinOthers) + " seen)";
    if (data.otherPeople < kMinOtherPeople)
        return "needs other faces (" + std::to_string(data.others.size()) + " seen, but from " +
               std::to_string(data.otherPeople) + " of " + std::to_string(kMinOtherPeople) +
               " people)";
    return "";
}

bool FaceNet::train(const FaceNetData& data) {
    // Start from "off": a failed training must not leave an older network behind.
    *this = FaceNet();
    if (!shortfall(data).empty()) return false;
    const int dim = static_cast<int>(data.samples[0][0].size());
    const int outputs = static_cast<int>(data.labels.size()) + 1;  // the labels, then "other"
    int faces = static_cast<int>(data.others.size());
    for (const auto& s : data.samples) faces += static_cast<int>(s.size());
    // Every face is one clean row plus kCopies rows that get noise (added per network, below).
    const int rows = faces * (1 + kCopies);

    // x: one embedding per row. y: one-hot targets (a 1 in the column of the right answer; the last
    // column is "other"). w: a weight per row.
    cv::Mat x(rows, dim, CV_32F), y(rows, outputs, CV_32F, cv::Scalar(0)), w(rows, 1, CV_32F);
    std::vector<int> plain;  // the rows without noise, to check what was learned
    int row = 0;
    auto add = [&](const std::vector<Embedding>& es, int output) {
        // Each output's samples weigh the same in all, so a label with few
        // photos counts as much as hundreds of other faces.
        const float weight = static_cast<float>(rows) / (outputs * es.size() * (1 + kCopies));
        for (const Embedding& e : es) {
            if (static_cast<int>(e.size()) != dim) return false;
            for (int copy = 0; copy <= kCopies; ++copy, ++row) {
                std::copy(e.begin(), e.end(), x.ptr<float>(row));
                if (copy == 0) plain.push_back(row);
                y.at<float>(row, output) = 1;
                w.at<float>(row) = weight;
            }
        }
        return true;
    };
    for (size_t i = 0; i < data.samples.size(); ++i)
        if (!add(data.samples[i], static_cast<int>(i))) return false;
    if (!add(data.others, outputs - 1)) return false;
    // The noisy copies: new noise for each network. ANN_MLP always starts from
    // the same weights (its own fixed seed), so the noise is what makes the
    // networks differ.
    auto withNoise = [&](uint64_t seed) {
        cv::RNG noise(seed);
        cv::Mat noisy = x.clone();
        for (int r = 0; r < rows; ++r) {
            if (std::binary_search(plain.begin(), plain.end(), r)) continue;
            cv::Mat n(1, dim, CV_32F), out = noisy.row(r);
            // Each component gets deviation kNoise / sqrt(dim), so the whole noise vector is about
            // kNoise long.
            noise.fill(n, cv::RNG::NORMAL, 0, kNoise / std::sqrt(static_cast<float>(dim)));
            out += n;
            // Back to length 1, like a real embedding.
            cv::normalize(out, out);
        }
        return cv::ml::TrainData::create(noisy, cv::ml::ROW_SAMPLE, y, cv::noArray(), cv::noArray(),
                                         w);
    };

    // A network has learned the data when, for each output, nearly all of its
    // own samples get it as the most likely answer.
    auto learned = [&](const cv::ml::ANN_MLP& net) {
        std::vector<int> right(outputs, 0), total(outputs, 0);
        cv::Mat out;
        // Judged on the clean rows only: did it learn the real samples?
        for (int r : plain) {
            net.predict(x.row(r), out);
            int truth = 0, answer = 0;
            for (int k = 0; k < outputs; ++k) {
                if (y.at<float>(r, k) > 0) truth = k;
                if (out.at<float>(0, k) > out.at<float>(0, answer)) answer = k;
            }
            ++total[truth];
            right[truth] += answer == truth;
        }
        for (int k = 0; k < outputs; ++k)
            if (right[k] < kLearned * total[k]) return false;
        return true;
    };

    // The ensemble: kNets networks, each trained on different noise (ANN_MLP always starts from the
    // same weights). A network that doesn't learn its data is thrown away and another attempt made,
    // up to kNets * kAttempts attempts in all.
    std::vector<cv::Ptr<cv::ml::ANN_MLP>> nets;
    for (int attempt = 0; attempt < kNets * kAttempts && nets.size() < kNets; ++attempt) {
        cv::Ptr<cv::ml::ANN_MLP> net = cv::ml::ANN_MLP::create();
        // Layers: the embedding as input, one hidden layer of kHidden units, one output per answer.
        net->setLayerSizes(std::vector<int>{dim, kHidden, outputs});
        // tanh-like units of slope 1 and range +-1: with OpenCV's default (1.7159
        // * tanh(2x/3)) RPROP saturated on these inputs and didn't converge.
        net->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
        net->setTrainMethod(cv::ml::ANN_MLP::RPROP);
        net->setTermCriteria(
            {cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, kIterations, 1e-5});
        // A new seed for each attempt, so a retry sees different noise (and a rerun sees the same).
        if (net->train(withNoise(kSeed + attempt)) && learned(*net)) nets.push_back(net);
    }
    // Too few networks learned the data (two labels of the same face, say): stay off.
    if (static_cast<int>(nets.size()) < kNets) return false;

    nets_ = std::move(nets);
    labels_ = data.labels;
    model_ = data.model;
    fingerprint_ = data.fingerprint();
    samples_ = faces - static_cast<int>(data.others.size());
    others_ = static_cast<int>(data.others.size());
    otherPeople_ = data.otherPeople;
    return true;
}

std::vector<float> FaceNet::predict(const Embedding& e) const {
    if (!on() || e.empty()) return {};
    if (static_cast<int>(e.size()) != nets_[0]->getLayerSizes().at<int>(0)) return {};
    // A header over the embedding's memory, no copy (cv::Mat's constructor wants a non-const
    // pointer; the network doesn't write to it).
    const cv::Mat in(1, static_cast<int>(e.size()), CV_32F, const_cast<float*>(e.data()));
    // mean: the average of the networks' answers, each first made into a probability distribution.
    std::vector<float> mean(labels_.size() + 1, 0.0f), p(mean.size());
    for (const cv::Ptr<cv::ml::ANN_MLP>& net : nets_) {
        cv::Mat out;
        net->predict(in, out);
        // The outputs estimate probabilities, a little off in places: clipped
        // to 0-1 and scaled to sum to 1.
        float sum = 0;
        for (size_t i = 0; i < p.size(); ++i)
            sum += p[i] = std::clamp(out.at<float>(0, static_cast<int>(i)), 0.0f, 1.0f);
        for (size_t i = 0; i < p.size(); ++i)
            mean[i] += (sum > 1e-6f ? p[i] / sum : 1.0f / p.size()) / nets_.size();
    }
    return mean;
}

bool FaceNet::fits(const std::vector<std::string>& labels, const std::string& model) const {
    return on() && labels == labels_ && model == model_;
}

bool FaceNet::trainedOn(const FaceNetData& data) const {
    return fits(data.labels, data.model) && data.fingerprint() == fingerprint_;
}

void FaceNet::save(const std::string& file) const {
    if (!on()) throw std::runtime_error("No face network to save to " + file);
    // Atomic write: a temp file, then a rename over the old file.
    const std::string temp = file + ".tmp.yml";
    {
        cv::FileStorage out(temp, cv::FileStorage::WRITE);
        if (!out.isOpened()) throw std::runtime_error("Can't write " + temp);
        out.writeComment("Written by the tracker: a neural network that learned the labels'");
        out.writeComment("faces. It's trained again when the labels or their photos change;");
        out.writeComment("delete it to have it trained again now.");
        out << "format" << kFormat << "model" << model_ << "labels" << labels_ << "fingerprint"
            << fingerprint_ << "samples" << samples_ << "others" << others_ << "other_people"
            << otherPeople_;
        out << "nets" << "[";
        for (const cv::Ptr<cv::ml::ANN_MLP>& net : nets_) {
            out << "{";
            net->write(out);
            out << "}";
        }
        out << "]";
    }
    fs::rename(temp, file);
}

bool FaceNet::load(const std::string& file) {
    *this = FaceNet();
    if (!fs::exists(file)) return false;
    // Read into a local object and moved in only if the whole file checks out, so a bad file leaves
    // this one off.
    FaceNet loaded;
    try {
        cv::FileStorage in(file, cv::FileStorage::READ);
        if (!in.isOpened() || static_cast<int>(in["format"]) != kFormat) return false;
        in["model"] >> loaded.model_;
        in["labels"] >> loaded.labels_;
        in["fingerprint"] >> loaded.fingerprint_;
        loaded.samples_ = static_cast<int>(in["samples"]);
        loaded.others_ = static_cast<int>(in["others"]);
        loaded.otherPeople_ = static_cast<int>(in["other_people"]);
        for (const cv::FileNode& node : in["nets"]) {
            cv::Ptr<cv::ml::ANN_MLP> net = cv::Algorithm::read<cv::ml::ANN_MLP>(node);
            if (!net || !net->isTrained()) return false;
            const cv::Mat sizes = net->getLayerSizes();
            // Expected shape: input, one hidden layer, and one output per label plus "other".
            if (sizes.total() != 3 ||
                sizes.at<int>(2) != static_cast<int>(loaded.labels_.size()) + 1)
                return false;
            loaded.nets_.push_back(net);
        }
    } catch (const cv::Exception&) {
        return false;
    }
    // A file that doesn't hold a whole ensemble is rejected.
    if (static_cast<int>(loaded.nets_.size()) != kNets) return false;
    *this = std::move(loaded);
    return true;
}
