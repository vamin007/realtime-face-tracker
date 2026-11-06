#include "Recognizer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace {

constexpr float kCropMargin = 0.3f;  // context kept around a box when copying it for the worker

float sizeOf(const cv::Rect2f& r) { return std::sqrt(std::max(r.area(), 1.0f)); }

}  // namespace

LabelVerdict combineMatch(const LabelMatch& m, float threshold, const std::vector<float>& probs,
                          const RecognizerParams& p) {
    LabelVerdict v;
    v.label = m.label;
    if (probs.empty() || m.best < 0 || m.best + 1 >= static_cast<int>(probs.size())) return v;
    if (m.label >= 0) {
        if (probs.back() >= p.netSure) v.label = -1, v.vetoed = true;
    } else if (m.score >= threshold - p.relax && m.score - m.runnerUp >= p.minMargin &&
               probs[m.best] >= p.netSure) {
        // Just below the threshold, but the network is sure (the runner-up still
        // has to be clearly behind).
        v.label = m.best;
        v.relaxed = true;
    }
    return v;
}

Recognizer::Recognizer(LabelLibrary& library, const ClassRegistry& classes, IEmbedder& embedder,
                       RecognizerParams params, Reinforcer* reinforcer)
    : library_(library),
      classes_(classes),
      embedder_(embedder),
      params_(params),
      reinforcer_(reinforcer),
      threshold_(embedder.defaultThreshold()) {
    // Load the model now, on this thread, so the worker never loads it and
    // loading errors show up at startup.
    if (!library_.labels().empty()) embedder_.load();
    useLatestNet();
    thread_ = std::thread(&Recognizer::work, this);
}

Recognizer::~Recognizer() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    thread_.join();
    for (auto& [trackId, id] : identities_) handOver(id);
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
        Result result;
        try {
            result.embedding = embedder_.embed(job.crop, job.box, job.keypoints);
            if (!result.embedding.empty()) {
                result.sharpness = sharpness(job.crop, job.box);
                // A face that may be photographed: its mirrored embedding too, as
                // a sync makes for every photo.
                if (job.photo && result.sharpness >= params_.capture.minSharpness) {
                    cv::Mat flipped;
                    cv::flip(job.crop, flipped, 1);
                    cv::Rect2f box = job.box;
                    std::vector<cv::Point2f> keypoints = job.keypoints;
                    mirrorDetection(flipped.cols, box, keypoints);
                    result.mirrored = embedder_.embed(flipped, box, keypoints);
                }
            }
        } catch (const std::exception& ex) {
            std::cerr << "Warning: recognition failed: " << ex.what() << '\n';
        }
        if (result.mirrored.empty()) job.crop.release();  // only a photo needs it
        result.job = std::move(job);
        std::lock_guard<std::mutex> lock(mutex_);
        results_.push_back(std::move(result));
        --inFlight_;
        idle_.notify_all();
    }
}

void Recognizer::waitIdle() {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [this] { return inFlight_ == 0; });
}

void Recognizer::useLatestNet() {
    if (!reinforcer_) return;
    const int version = reinforcer_->version();
    if (version == netVersion_) return;
    netVersion_ = version;
    // A network trained for other labels (or another embedding model) can't be used.
    std::vector<std::string> names;
    for (const CustomLabel& label : library_.labels()) names.push_back(label.name);
    std::shared_ptr<const FaceNet> net = reinforcer_->net();
    net_ = net && net->fits(names, embedder_.modelId()) ? net : nullptr;
    netStatus_ = reinforcer_->status();
}

double Recognizer::lastPhoto(int label) const {
    const auto it = lastPhoto_.find(label);
    return it == lastPhoto_.end() ? -1 : it->second;
}

void Recognizer::apply(Result& result) {
    const int trackId = result.job.trackId;
    TrackIdentity& id = identities_[trackId];
    id.pending = false;
    if (result.embedding.empty()) return;
    const Embedding& e = result.embedding;
    ++id.checks;
    const LabelMatch m = library_.match(e, threshold_, params_.minMargin);
    const std::vector<float> probs = net_ ? net_->predict(e) : std::vector<float>{};
    const float threshold =
        m.best >= 0 ? thresholdOf(library_.labels()[m.best], threshold_) : threshold_;
    const LabelVerdict verdict = combineMatch(m, threshold, probs, params_);
    if (m.label >= 0 || verdict.relaxed) id.everMatched = true;
    id.netLabel.clear();
    id.netProb = -1;
    if (!probs.empty()) {
        const size_t k = std::max_element(probs.begin(), probs.end()) - probs.begin();
        id.netLabel = k < library_.labels().size() ? library_.labels()[k].name : "other";
        id.netProb = probs[k];
    }
    // The closest label, matched or not (for the debug view and dump), and how
    // far each label is from matching.
    id.closest.clear();
    float nearest = -FLT_MAX;  // the highest score minus its label's threshold
    for (const CustomLabel& label : library_.labels()) {
        const float s = labelScore(label, e);
        if (id.closest.empty() || s > id.closestScore) id.closestScore = s, id.closest = label.name;
        nearest = std::max(nearest, s - thresholdOf(label, threshold_));
    }
    const std::string assigned = id.label >= 0 ? library_.labels()[id.label].name : "";

    vote(trackId, id, verdict.label, m.score);
    // A labeled face may be photographed for its label; a face unlike every
    // label is kept as someone else's (handed over once its track ends).
    id.photo.clear();
    if (id.label >= 0) {
        considerPhoto(id, result, m);
    } else if (reinforcer_ && params_.autoPhotos && !id.everMatched &&
               nearest < -params_.others.below && result.job.detected &&
               result.job.box.width >= params_.capture.minWidth &&
               result.sharpness >= params_.capture.minSharpness &&
               static_cast<int>(id.others.size()) < params_.others.perTrack &&
               (id.otherAt < 0 || result.job.time - id.otherAt >= params_.others.interval)) {
        id.others.push_back(e);
        id.otherAt = result.job.time;
    }

    if (debug_)
        *debug_ << now_ << ',' << trackId << ',' << classes_.name(classes_.face()) << ','
                << id.closest << ',' << id.closestScore << ',' << m.runnerUp << ','
                << (verdict.label >= 0 ? 1 : 0) << ',' << assigned << ',' << id.netLabel << ','
                << id.netProb << ',' << result.sharpness << ',' << id.photo << '\n';
}

void Recognizer::vote(int trackId, TrackIdentity& id, int matched, float score) {
    if (matched >= 0) ++id.votes[matched];
    if (matched >= 0 && matched == id.streakLabel) {
        ++id.streak;
    } else {
        id.streakLabel = matched;
        id.streak = matched >= 0 ? 1 : 0;
    }

    if (id.label >= 0) {
        if (matched == id.label) {
            id.misses = 0;
            id.score = score;
        } else if (++id.misses >= params_.missesToDrop) {
            id.votes.erase(id.label);  // it stopped looking like that item
            id.label = -1;
            id.misses = 0;
        }
        return;
    }
    if (matched < 0 || id.votes[matched] < params_.votesToAssign) return;

    // Each label is one specific item: if another track has it, the one with
    // more matching results keeps it.
    for (auto& [otherId, other] : identities_) {
        if (otherId == trackId || other.label != matched) continue;
        if (other.votes[matched] >= id.votes[matched]) return;
        other.label = -1;
        other.votes.erase(matched);
    }
    id.label = matched;
    id.score = score;
    id.misses = 0;
}

// Whether a labeled track's next crop may give a photo, from what's known
// before it's embedded (the rest is taken to pass).
bool Recognizer::mayPhotograph(const TrackIdentity& id, const Job& job) const {
    if (!reinforcer_ || !params_.autoPhotos || id.label < 0) return false;
    CaptureInput in;
    in.enabled = library_.labels()[id.label].autoPhotos;
    in.streak = (id.streakLabel == id.label ? id.streak : 0) + 1;  // if this one matches too
    in.score = 1;
    in.detected = job.detected;
    in.width = job.box.width;
    in.sharpness = FLT_MAX;
    in.similarity = -1;
    in.now = job.time;
    in.lastPhoto = lastPhoto(id.label);
    return capturePolicy(in, params_.capture) == CaptureGate::Take;
}

void Recognizer::considerPhoto(TrackIdentity& id, const Result& r, const LabelMatch& m) {
    const CustomLabel& label = library_.labels()[id.label];
    CaptureInput in;
    in.enabled = reinforcer_ && params_.autoPhotos && label.autoPhotos;
    in.streak = id.streakLabel == id.label ? id.streak : 0;
    // The photos' verdict on this label, not the network's.
    in.score = m.best == id.label ? m.score : labelScore(label, r.embedding);
    in.threshold = thresholdOf(label, threshold_);
    in.runnerUp = m.best == id.label ? m.runnerUp : m.score;
    in.detected = r.job.detected;
    in.width = r.job.box.width;
    in.sharpness = r.sharpness;
    in.similarity = maxSimilarity(label, r.embedding);
    in.now = r.job.time;
    in.lastPhoto = lastPhoto(id.label);
    CaptureGate gate = capturePolicy(in, params_.capture);
    // Only a crop sent as a possible photo has its mirrored embedding (it
    // wasn't when the label had just become steady).
    if (gate == CaptureGate::Take && r.mirrored.empty()) gate = CaptureGate::Unstable;
    if (gate != CaptureGate::Take) {
        id.photo = describe(gate);
        return;
    }
    takePhoto(id, r);
}

void Recognizer::takePhoto(TrackIdentity& id, const Result& r) {
    const int index = id.label;
    CustomLabel& label = library_.labels()[index];
    const std::string folder = library_.folder(label.name);
    PhotoRecord record;
    record.file.name = autoPhotoName(std::time(nullptr), [&](const std::string& name) {
        return std::any_of(label.samples.begin(), label.samples.end(),
                           [&](const LabelSample& s) { return s.image == name; }) ||
               fs::exists(fs::path(folder) / name);
    });
    record.used = true;
    char note[64];
    std::snprintf(note, sizeof(note), "taken automatically while recognized (score %.2f)",
                  labelScore(label, r.embedding));
    record.note = note;
    record.roi = cv::Rect(r.job.box);
    record.conf = r.job.conf;
    record.embedding = r.embedding;
    record.mirrored = r.mirrored;

    // Used at once. When the label is full, the automatic photo that adds the
    // least goes, which may be this one.
    library_.addPhoto(index, record);
    lastPhoto_[index] = r.job.time;
    const std::string drop = photoToReplace(label, params_.capture.maxPhotos);
    if (drop == record.file.name) {
        library_.removePhoto(index, drop);
        id.photo = describe(CaptureGate::NotNew);
        return;
    }
    if (!drop.empty()) library_.removePhoto(index, drop);
    reinforcer_->savePhoto({folder, r.job.crop, record, drop},
                           faceNetData(library_, embedder_.modelId()));
    id.photo = describe(CaptureGate::Take);
}

// A track that never matched a label, looked at often enough: its faces are
// someone else's.
void Recognizer::handOver(TrackIdentity& id) {
    if (!reinforcer_ || id.everMatched || id.checks < params_.others.minChecks || id.others.empty())
        return;
    reinforcer_->addOthers(std::move(id.others));
    id.others.clear();
}

void Recognizer::update(const cv::Mat& frame, const std::vector<Track>& tracks, double now) {
    now_ = now;
    useLatestNet();  // a network the reinforcer has just trained
    // 1. Use finished embeddings.
    std::vector<Result> done;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        done.swap(results_);
    }
    for (Result& r : done)
        if (identities_.count(r.job.trackId)) apply(r);  // not if the track is gone

    // 2. Forget tracks that are gone.
    for (auto it = identities_.begin(); it != identities_.end();) {
        const bool alive = std::any_of(tracks.begin(), tracks.end(),
                                       [&](const Track& t) { return t.id == it->first; });
        if (!alive) handOver(it->second);
        it = alive ? std::next(it) : identities_.erase(it);
    }

    // 3. Send crops of faces that are due a (new) look: unlabeled ones first,
    //    then the ones looked at longest ago.
    std::vector<const Track*> due;
    for (const Track& t : tracks) {
        if (library_.labels().empty() || classes_.base(t.classId) != classes_.face()) continue;
        if (!t.visible || !t.matched) continue;
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
        TrackIdentity& id = identities_[t->id];
        Job job;
        job.trackId = t->id;
        job.time = now;
        job.crop = frame(area).clone();
        const cv::Point2f origin(static_cast<float>(area.x), static_cast<float>(area.y));
        job.box = cv::Rect2f(t->box.tl() - origin, t->box.size());
        for (const cv::Point2f& p : t->keypoints) job.keypoints.push_back(p - origin);
        job.conf = t->conf;
        job.detected = t->matched && !t->coasting && t->lastStrong >= t->lastSeen;
        job.photo = mayPhotograph(id, job);
        jobs_.push_back(std::move(job));
        ++inFlight_;
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
