#include "Recognizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

constexpr float kCropMargin = 0.3f;  // context kept around a box when copying it for the worker

float sizeOf(const cv::Rect2f& r) { return std::sqrt(std::max(r.area(), 1.0f)); }

}  // namespace

Embedders::Embedders(std::string faceModel, std::string objectModel, Backend backend,
                     std::string cacheDir)
    : faceModel_(std::move(faceModel)),
      objectModel_(std::move(objectModel)),
      cacheDir_(std::move(cacheDir)),
      backend_(backend) {}

IEmbedder& Embedders::forKind(ClassKind kind) {
    if (kind == ClassKind::Face) {
        if (!face_) face_ = std::make_unique<FaceEmbedder>(faceModel_);
        return *face_;
    }
    // DINOv2 runs on the CPU even when CoreML is chosen: CoreML ran it in 8 ms
    // instead of 84 ms, but compiling it delayed every launch by 8-38 s, while
    // recognition only needs a few crops a second on its own thread.
    if (!object_) object_ = std::make_unique<ObjectEmbedder>(objectModel_, Backend::Cpu, cacheDir_);
    return *object_;
}

std::string Embedders::modelId(ClassKind kind) const {
    return kind == ClassKind::Face ? FaceEmbedder::kModelId : ObjectEmbedder::kModelId;
}

float Embedders::defaultThreshold(ClassKind kind) const {
    return kind == ClassKind::Face ? FaceEmbedder::kThreshold : ObjectEmbedder::kThreshold;
}

std::string Embedders::info(ClassKind kind) {
    if (kind == ClassKind::Face) return "SFace, OpenCV CPU";
    forKind(kind);
    return "DINOv2-small, " + object_->info();
}

Recognizer::Recognizer(const LabelLibrary& library, const ClassRegistry& classes,
                       Embedders& embedders, RecognizerParams params)
    : library_(library), classes_(classes), embedders_(embedders), params_(params) {
    // Load the models the labels need now, on this thread, so the worker never
    // creates them and loading errors show up at startup.
    for (const CustomLabel& label : library_.labels()) {
        const ClassId base = classes_.find(label.baseClass);
        const ClassKind kind =
            classes_.kind(base) == ClassKind::Face ? ClassKind::Face : ClassKind::Object;
        thresholds_[kind] = embedders_.forKind(kind).defaultThreshold();
    }
    thread_ = std::thread(&Recognizer::work, this);
}

Recognizer::~Recognizer() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    thread_.join();
}

std::string Recognizer::baseClassOf(const Track& t) const {
    return classes_.name(classes_.base(t.classId));
}

void Recognizer::work() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
            if (stop_) return;
            job = std::move(jobs_.front());
            jobs_.pop_front();
        }
        Embedding e;
        try {
            e = embedders_.forKind(job.kind).embed(job.crop, job.box, job.keypoints);
        } catch (const std::exception& ex) {
            std::cerr << "Warning: recognition failed: " << ex.what() << '\n';
        }
        std::lock_guard<std::mutex> lock(mutex_);
        results_.push_back({job.trackId, job.kind, std::move(e)});
        --inFlight_;
    }
}

void Recognizer::apply(const Result& result, const std::string& baseClass) {
    TrackIdentity& id = identities_[result.trackId];
    id.pending = false;
    if (result.embedding.empty()) return;
    const LabelMatch m =
        library_.match(baseClass, result.embedding, thresholds_[result.kind], params_.minMargin);
    if (m.label >= 0) ++id.votes[m.label];
    // The closest label, matched or not (for the debug view and dump).
    id.closest.clear();
    for (const CustomLabel& label : library_.labels()) {
        if (label.baseClass != baseClass) continue;
        const float s = labelScore(label, result.embedding);
        if (id.closest.empty() || s > id.closestScore) id.closestScore = s, id.closest = label.name;
    }
    if (debug_)
        *debug_ << now_ << ',' << result.trackId << ',' << baseClass << ',' << id.closest << ','
                << id.closestScore << ',' << m.runnerUp << ',' << (m.label >= 0 ? 1 : 0) << ','
                << (id.label >= 0 ? library_.labels()[id.label].name : "") << '\n';

    if (id.label >= 0) {
        if (m.label == id.label) {
            id.misses = 0;
            id.score = m.score;
        } else if (++id.misses >= params_.missesToDrop) {
            id.votes.erase(id.label);  // it stopped looking like that item
            id.label = -1;
            id.misses = 0;
        }
        return;
    }
    if (m.label < 0 || id.votes[m.label] < params_.votesToAssign) return;

    // Each label is one specific item: if another track has it, the one with
    // more matching results keeps it.
    for (auto& [otherId, other] : identities_) {
        if (otherId == result.trackId || other.label != m.label) continue;
        if (other.votes[m.label] >= id.votes[m.label]) return;
        other.label = -1;
        other.votes.erase(m.label);
    }
    id.label = m.label;
    id.score = m.score;
    id.misses = 0;
}

void Recognizer::update(const cv::Mat& frame, const std::vector<Track>& tracks, double now) {
    now_ = now;
    // 1. Use finished embeddings.
    std::vector<Result> done;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        done.swap(results_);
    }
    for (const Result& r : done) {
        const auto base = baseClasses_.find(r.trackId);
        if (base != baseClasses_.end()) apply(r, base->second);
    }

    // 2. Forget tracks that are gone.
    for (auto it = identities_.begin(); it != identities_.end();) {
        const bool alive = std::any_of(tracks.begin(), tracks.end(),
                                       [&](const Track& t) { return t.id == it->first; });
        if (!alive) baseClasses_.erase(it->first);
        it = alive ? std::next(it) : identities_.erase(it);
    }

    // 3. Send crops of tracks that are due a (new) look: unlabeled ones first,
    //    then the ones looked at longest ago.
    std::vector<const Track*> due;
    for (const Track& t : tracks) {
        if (!t.visible || !t.matched) continue;
        const std::string base = baseClassOf(t);
        if (!library_.hasLabelsFor(base)) continue;
        baseClasses_[t.id] = base;
        const TrackIdentity& id = identities_[t.id];
        if (id.pending) continue;
        const bool resized =
            id.askedSize > 0 && std::abs(sizeOf(t.box) / id.askedSize - 1) > params_.resizeTrigger;
        const double interval = id.label >= 0 ? params_.labeledRecheck : params_.recheckInterval;
        if (id.askedAt < 0 || now - id.askedAt >= interval || resized) due.push_back(&t);
    }
    std::sort(due.begin(), due.end(), [&](const Track* a, const Track* b) {
        const TrackIdentity& ia = identities_[a->id];
        const TrackIdentity& ib = identities_[b->id];
        if ((ia.label < 0) != (ib.label < 0)) return ia.label < 0;
        return ia.askedAt < ib.askedAt;
    });

    std::lock_guard<std::mutex> lock(mutex_);
    for (const Track* t : due) {
        if (inFlight_ >= params_.maxPending) break;
        const cv::Rect2f grown(
            t->box.x - t->box.width * kCropMargin, t->box.y - t->box.height * kCropMargin,
            t->box.width * (1 + 2 * kCropMargin), t->box.height * (1 + 2 * kCropMargin));
        const cv::Rect area = cv::Rect(grown) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (area.area() <= 0) continue;
        Job job;
        job.trackId = t->id;
        job.kind = classes_.kind(classes_.base(t->classId)) == ClassKind::Face ? ClassKind::Face
                                                                               : ClassKind::Object;
        job.crop = frame(area).clone();
        const cv::Point2f origin(static_cast<float>(area.x), static_cast<float>(area.y));
        job.box = cv::Rect2f(t->box.tl() - origin, t->box.size());
        for (const cv::Point2f& p : t->keypoints) job.keypoints.push_back(p - origin);
        jobs_.push_back(std::move(job));
        ++inFlight_;
        TrackIdentity& id = identities_[t->id];
        id.pending = true;
        id.askedAt = now;
        id.askedSize = sizeOf(t->box);
    }
    wake_.notify_one();
}

int Recognizer::labelOf(int trackId) const {
    const auto it = identities_.find(trackId);
    return it == identities_.end() ? -1 : it->second.label;
}

const std::string* Recognizer::labelName(int trackId) const {
    const int label = labelOf(trackId);
    return label < 0 ? nullptr : &library_.labels()[label].name;
}

const TrackIdentity* Recognizer::identity(int trackId) const {
    const auto it = identities_.find(trackId);
    return it == identities_.end() ? nullptr : &it->second;
}
