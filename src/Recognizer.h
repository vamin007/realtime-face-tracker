#pragma once
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <thread>
#include <vector>

#include "ClassRegistry.h"
#include "Embedder.h"
#include "Labels.h"
#include "Tracker.h"

// The two embedding models, loaded on first use (DINOv2 isn't needed unless an
// object label exists). Virtual so tests can stand in simpler ones.
class Embedders {
public:
    Embedders(std::string faceModel, std::string objectModel, Backend backend,
              std::string cacheDir);
    virtual ~Embedders() = default;
    virtual IEmbedder& forKind(ClassKind kind);  // faces: SFace; anything else: DINOv2
    std::string info(ClassKind kind);
    // The model's id and default threshold, without loading it.
    virtual std::string modelId(ClassKind kind) const;
    virtual float defaultThreshold(ClassKind kind) const;

private:
    std::string faceModel_, objectModel_, cacheDir_;
    Backend backend_;
    std::unique_ptr<FaceEmbedder> face_;
    std::unique_ptr<ObjectEmbedder> object_;
};

struct RecognizerParams {
    double recheckInterval = 0.25;  // s between embeddings of a track without a label
    double labeledRecheck =
        1.0;                     // s between embeddings of a labeled track (checking it still fits)
    float resizeTrigger = 0.3f;  // ... or sooner once its size changed this much
    int votesToAssign = 3;       // matching results before a track gets a label
    int missesToDrop = 4;        // results in a row not matching its label to take it away
    float minMargin = 0.05f;     // the best label must beat the runner-up by this
    int maxPending = 2;          // crops waiting for or being embedded
};

// What recognition knows about one track.
struct TrackIdentity {
    int label = -1;            // index into LabelLibrary::labels(); -1: none
    std::map<int, int> votes;  // label index -> matching results
    int misses = 0;            // results in a row that didn't match the assigned label
    float score = 0;           // the assigned label's latest score
    double askedAt = -1;       // when this track was last sent for embedding
    float askedSize = 0;       // its size then
    bool pending = false;
    std::string closest;     // the closest label in the latest check, matched or not
    float closestScore = 0;  // ... and its score
};

// Gives tracks custom labels. Each frame it sends a few crops of tracks whose
// class has labels (e.g. cups, when "my_mug" exists) to a background thread
// for embedding, then compares the results with the labels' samples. A track
// gets a label after several consistent matches and keeps it until it stops
// matching; each label (a specific item) belongs to at most one track.
class Recognizer {
public:
    Recognizer(const LabelLibrary& library, const ClassRegistry& classes, Embedders& embedders,
               RecognizerParams params = {});
    ~Recognizer();
    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    // Call after the tracker each frame, before drawing on the frame.
    void update(const cv::Mat& frame, const std::vector<Track>& tracks, double now);

    int labelOf(int trackId) const;                    // -1 if none
    const std::string* labelName(int trackId) const;   // null if none
    const TrackIdentity* identity(int trackId) const;  // null if never looked at

    // Developer aid: one CSV line per embedding result (time, track, scores).
    void setDebugStream(std::ostream* out) { debug_ = out; }

private:
    struct Job {
        int trackId;
        ClassKind kind;
        cv::Mat crop;  // a copy: the frame gets drawn on while the worker runs
        cv::Rect2f box;
        std::vector<cv::Point2f> keypoints;
    };
    struct Result {
        int trackId;
        ClassKind kind;
        Embedding embedding;
    };

    void work();
    void apply(const Result& result, const std::string& baseClass);
    std::string baseClassOf(const Track& t) const;

    const LabelLibrary& library_;
    const ClassRegistry& classes_;
    Embedders& embedders_;
    RecognizerParams params_;
    std::map<ClassKind, float> thresholds_;
    std::map<int, TrackIdentity> identities_;
    std::map<int, std::string> baseClasses_;  // track id -> its base class name
    std::ostream* debug_ = nullptr;
    double now_ = 0;

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Job> jobs_;
    std::vector<Result> results_;
    int inFlight_ = 0;
    bool stop_ = false;
};
