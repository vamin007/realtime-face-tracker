#include "Reinforcement.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

constexpr int kFormat = 1;
constexpr int kJpegQuality = 95;

std::string join(const std::vector<std::string>& names) {
    std::string text;
    for (size_t i = 0; i < names.size(); ++i)
        text += (i == 0 ? "" : i + 1 < names.size() ? ", " : " and ") + names[i];
    return text;
}

// "on (alex and sam; 46 label samples, 35 other faces of 6 people)".
std::string describeNet(const FaceNet& net) {
    return "on (" + join(net.labels()) + "; trained on " + std::to_string(net.samples()) +
           " label samples and " + std::to_string(net.others()) + " other faces of " +
           std::to_string(net.otherPeople()) + " people)";
}

const char* kCantLearn = "off: it couldn't learn to tell the faces apart";

// The highest similarity of a sample of `image` to a sample of any other photo.
float redundancy(const CustomLabel& label, const std::string& image) {
    float most = -1;
    for (const LabelSample& s : label.samples) {
        if (s.image != image || s.flipped) continue;
        for (const LabelSample& o : label.samples)
            if (o.image != image) most = std::max(most, similarity(s.embedding, o.embedding));
    }
    return most;
}

}  // namespace

CaptureGate capturePolicy(const CaptureInput& in, const CaptureParams& p) {
    if (!in.enabled) return CaptureGate::Off;
    if (in.streak < p.minStreak) return CaptureGate::Unstable;
    if (!in.detected) return CaptureGate::NotDetected;
    if (in.width < p.minWidth) return CaptureGate::TooSmall;
    if (in.lastPhoto >= 0 && in.now - in.lastPhoto < p.interval) return CaptureGate::TooSoon;
    if (in.score < in.threshold + p.scoreLead) return CaptureGate::LowScore;
    if (in.score - in.runnerUp < p.minMargin) return CaptureGate::CloseRunnerUp;
    if (in.sharpness < p.minSharpness) return CaptureGate::Blurry;
    if (in.similarity > p.maxSimilarity) return CaptureGate::NotNew;
    return CaptureGate::Take;
}

const char* describe(CaptureGate gate) {
    switch (gate) {
        case CaptureGate::Take:
            return "taken";
        case CaptureGate::Off:
            return "off";
        case CaptureGate::Unstable:
            return "not steady yet";
        case CaptureGate::NotDetected:
            return "not detected";
        case CaptureGate::TooSmall:
            return "too small";
        case CaptureGate::TooSoon:
            return "too soon";
        case CaptureGate::LowScore:
            return "not a clear match";
        case CaptureGate::CloseRunnerUp:
            return "too like another label";
        case CaptureGate::Blurry:
            return "blurry";
        case CaptureGate::NotNew:
            return "nothing new";
    }
    return "";
}

float sharpness(const cv::Mat& image, const cv::Rect2f& face) {
    const cv::Rect r = cv::Rect(face) & cv::Rect(0, 0, image.cols, image.rows);
    if (r.area() <= 0) return 0;
    cv::Mat gray, small, laplacian;
    if (image.channels() == 3) cv::cvtColor(image(r), gray, cv::COLOR_BGR2GRAY);
    else gray = image(r);
    cv::resize(gray, small, {112, 112}, 0, 0, cv::INTER_AREA);
    cv::Laplacian(small, laplacian, CV_32F);
    cv::Scalar mean, deviation;
    cv::meanStdDev(laplacian(cv::Rect(22, 22, 68, 68)), mean, deviation);  // the middle 60%
    return static_cast<float>(deviation[0] * deviation[0]);
}

float maxSimilarity(const CustomLabel& label, const Embedding& e) {
    float most = -1;
    for (const LabelSample& s : label.samples) most = std::max(most, similarity(s.embedding, e));
    return most;
}

int autoPhotoCount(const CustomLabel& label) {
    std::set<std::string> names;
    for (const LabelSample& s : label.samples)
        if (isAutoPhotoName(s.image)) names.insert(s.image);
    return static_cast<int>(names.size());
}

std::string photoToReplace(const CustomLabel& label, int maxPhotos) {
    if (autoPhotoCount(label) <= maxPhotos) return "";
    std::string drop;
    float most = -2;
    for (const LabelSample& s : label.samples) {
        if (s.flipped || !isAutoPhotoName(s.image)) continue;
        const float r = redundancy(label, s.image);
        if (r > most) most = r, drop = s.image;
    }
    return drop;
}

std::string autoPhotoName(std::time_t t, const std::function<bool(const std::string&)>& taken) {
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&t));
    const std::string base = std::string("auto_") + stamp;
    std::string name = base + ".jpg";
    for (int n = 2; taken(name); ++n) name = base + "_" + std::to_string(n) + ".jpg";
    return name;
}

// --- NegativeStore ---------------------------------------------------------

void NegativeStore::add(const Embedding& e, const std::string& source) {
    faces_.push_back(e);
    sources_.push_back(source);
    if (static_cast<int>(faces_.size()) <= capacity_) return;
    // Full: the face most like another one adds the least.
    size_t drop = 0;
    float most = -2;
    for (size_t i = 0; i < faces_.size(); ++i) {
        float nearest = -2;
        for (size_t j = 0; j < faces_.size(); ++j)
            if (j != i) nearest = std::max(nearest, similarity(faces_[i], faces_[j]));
        if (nearest > most) most = nearest, drop = i;
    }
    faces_.erase(faces_.begin() + drop);
    sources_.erase(sources_.begin() + drop);
}

int NegativeStore::people() const {
    return static_cast<int>(std::set<std::string>(sources_.begin(), sources_.end()).size());
}

bool NegativeStore::load(const std::string& file, const std::string& model) {
    faces_.clear();
    sources_.clear();
    if (!fs::exists(file)) return false;
    try {
        cv::FileStorage in(file, cv::FileStorage::READ);
        if (!in.isOpened() || static_cast<int>(in["format"]) != kFormat) return false;
        std::string fileModel;
        in["model"] >> fileModel;
        if (fileModel != model) return false;
        cv::Mat faces;
        in["faces"] >> faces;
        std::vector<std::string> sources;
        in["sources"] >> sources;
        if (faces.empty() || faces.type() != CV_32F ||
            faces.rows != static_cast<int>(sources.size()))
            return false;
        for (int i = 0; i < faces.rows; ++i)
            faces_.emplace_back(faces.ptr<float>(i), faces.ptr<float>(i) + faces.cols);
        sources_ = std::move(sources);
    } catch (const cv::Exception&) {
        faces_.clear();
        sources_.clear();
        std::cerr << "Warning: can't read " << file << "; other faces are collected again\n";
        return false;
    }
    while (static_cast<int>(faces_.size()) > capacity_) {
        faces_.erase(faces_.begin());
        sources_.erase(sources_.begin());
    }
    return true;
}

void NegativeStore::save(const std::string& file, const std::string& model) const {
    const std::string temp = file + ".tmp.yml";
    {
        cv::FileStorage out(temp, cv::FileStorage::WRITE);
        if (!out.isOpened()) throw std::runtime_error("Can't write " + temp);
        out.writeComment("Written by the tracker: embeddings (not photos) of faces that matched");
        out.writeComment("no label, which teach the face network what other people look like.");
        out.writeComment("Delete it to forget them.");
        cv::Mat faces(static_cast<int>(faces_.size()),
                      faces_.empty() ? 0 : static_cast<int>(faces_[0].size()), CV_32F);
        for (size_t i = 0; i < faces_.size(); ++i)
            std::copy(faces_[i].begin(), faces_[i].end(), faces.ptr<float>(static_cast<int>(i)));
        out << "format" << kFormat << "model" << model << "faces" << faces << "sources" << sources_;
    }
    fs::rename(temp, file);
}

// --- The face network's data and state -------------------------------------

FaceNetData faceNetData(const LabelLibrary& library, const std::string& model) {
    FaceNetData data;
    data.model = model;
    for (const CustomLabel& label : library.labels()) {
        data.labels.push_back(label.name);
        data.samples.emplace_back();
        for (const LabelSample& s : label.samples) data.samples.back().push_back(s.embedding);
    }
    return data;
}

namespace {

// The labels' samples and the other faces saved in the labels directory.
FaceNetData savedData(const LabelLibrary& library, const std::string& model) {
    FaceNetData data = faceNetData(library, model);
    NegativeStore store;
    store.load((fs::path(library.dir()) / kNegativesFile).string(), model);
    data.others = store.faces();
    data.otherPeople = store.people();
    return data;
}

}  // namespace

std::string faceNetStatus(const LabelLibrary& library, const std::string& model) {
    const FaceNetData data = savedData(library, model);
    if (const std::string why = FaceNet::shortfall(data); !why.empty()) return "off: " + why;
    FaceNet net;
    if (net.load((fs::path(library.dir()) / kFaceNetFile).string()) && net.trainedOn(data))
        return describeNet(net);
    return "not trained yet (it will be at the next start or --sync-labels)";
}

std::string trainFaceNet(const LabelLibrary& library, const std::string& model) {
    const FaceNetData data = savedData(library, model);
    if (const std::string why = FaceNet::shortfall(data); !why.empty()) return "off: " + why;
    const std::string file = (fs::path(library.dir()) / kFaceNetFile).string();
    FaceNet net;
    if (net.load(file) && net.trainedOn(data)) return describeNet(net);
    if (!net.train(data)) return kCantLearn;
    net.save(file);
    return describeNet(net);
}

// --- Reinforcer ------------------------------------------------------------

Reinforcer::Reinforcer(std::string labelsDir, FaceNetData labels)
    : dir_(std::move(labelsDir)), labels_(std::move(labels)) {
    labels_.others.clear();
    const std::time_t now = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now));
    run_ = stamp;
    store_.load((fs::path(dir_) / kNegativesFile).string(), labels_.model);
    others_ = static_cast<int>(store_.faces().size());
    auto saved = std::make_shared<FaceNet>();
    if (saved->load((fs::path(dir_) / kFaceNetFile).string()) && saved->trainedOn(data()))
        publish(saved, describeNet(*saved));
    else retrain();  // before tracking starts: quick (well under a second)
    thread_ = std::thread(&Reinforcer::work, this);
}

Reinforcer::~Reinforcer() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    thread_.join();
}

void Reinforcer::savePhoto(Photo photo, FaceNetData labels) {
    Task task;
    task.photo = true;
    task.shot = std::move(photo);
    task.labels = std::move(labels);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(task));
    }
    wake_.notify_one();
}

void Reinforcer::addOthers(std::vector<Embedding> faces) {
    if (faces.empty()) return;
    Task task;
    task.others = std::move(faces);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(task));
    }
    wake_.notify_one();
}

std::shared_ptr<const FaceNet> Reinforcer::net() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return net_;
}

std::string Reinforcer::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

int Reinforcer::others() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return others_;
}

void Reinforcer::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [this] { return queue_.empty() && !busy_; });
}

void Reinforcer::work() {
    for (;;) {
        std::deque<Task> tasks;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) return;  // stopped, and nothing left to do
            tasks.swap(queue_);
            busy_ = true;
        }
        bool changed = false;
        for (Task& task : tasks) {
            try {
                changed |= run(task);
            } catch (const std::exception& e) {
                std::cerr << "Warning: reinforcement: " << e.what() << '\n';
            }
        }
        if (changed) {
            try {
                retrain();
            } catch (const std::exception& e) {
                std::cerr << "Warning: training the face network failed: " << e.what() << '\n';
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            busy_ = false;
            others_ = static_cast<int>(store_.faces().size());
        }
        idle_.notify_all();
    }
}

bool Reinforcer::run(Task& task) {
    if (!task.photo) {
        const std::string source = run_ + " #" + std::to_string(++tracks_);
        for (const Embedding& e : task.others) store_.add(e, source);
        store_.save((fs::path(dir_) / kNegativesFile).string(), labels_.model);
        return true;
    }
    // The photo: written next to its place under a hidden name (a sync doesn't
    // see it), then renamed, so a crash never leaves half a photo.
    Photo& p = task.shot;
    const fs::path folder(p.folder), path = folder / p.record.file.name;
    const fs::path temp = folder / ("." + p.record.file.name + ".tmp.jpg");
    if (!cv::imwrite(temp.string(), p.image, {cv::IMWRITE_JPEG_QUALITY, kJpegQuality}))
        throw std::runtime_error("can't write " + temp.string());
    fs::rename(temp, path);
    if (!p.replaces.empty()) {
        std::error_code ec;
        fs::remove(folder / p.replaces, ec);
    }
    if (!appendPhotoRecord(p.folder, p.record, p.replaces))
        std::cerr << "Warning: " << path.string()
                  << " isn't in the label's .embeddings.yml; the next sync looks at it\n";
    labels_ = std::move(task.labels);
    labels_.others.clear();
    return true;
}

FaceNetData Reinforcer::data() const {
    FaceNetData d = labels_;
    d.others = store_.faces();
    d.otherPeople = store_.people();
    return d;
}

void Reinforcer::retrain() {
    const FaceNetData d = data();
    if (const std::string why = FaceNet::shortfall(d); !why.empty()) {
        publish(nullptr, "off: " + why);
        return;
    }
    auto net = std::make_shared<FaceNet>();
    if (!net->train(d)) {
        publish(nullptr, kCantLearn);
        return;
    }
    try {
        net->save((fs::path(dir_) / kFaceNetFile).string());
    } catch (const std::exception& e) {
        std::cerr << "Warning: " << e.what() << " (the face network is used, but not saved)\n";
    }
    publish(net, describeNet(*net));
}

void Reinforcer::publish(std::shared_ptr<const FaceNet> net, std::string status) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        net_ = std::move(net);
        status_ = std::move(status);
    }
    ++version_;
}
