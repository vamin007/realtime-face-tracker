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
#include "FaceNet.h"
#include "Labels.h"
#include "Reinforcement.h"
#include "Tracker.h"

struct RecognizerParams {
    double recheckInterval = 0.25;  // s between embeddings of a track without a label
    double labeledRecheck =
        1.0;                     // s between embeddings of a labeled track (checking it still fits)
    float resizeTrigger = 0.3f;  // ... or sooner once its size changed this much
    int votesToAssign = 3;       // matching results before a track gets a label
    int missesToDrop = 4;        // results in a row not matching its label to take it away
    float minMargin = 0.05f;     // the best label must beat the runner-up by this
    int maxPending = 2;          // crops waiting for or being embedded
    // The face network, when it's on (see FaceNet): a face scoring up to
    // `relax` below a label's threshold also matches it when the network gives
    // that label at least `netSure`; a match doesn't count when the network
    // says "someone else" that surely.
    float relax = 0.08f;
    float netSure = 0.9f;
    // Reinforcement (with a Reinforcer): photos of recognized faces for their
    // label, and the embeddings of faces that match no label.
    bool autoPhotos = true;
    CaptureParams capture;
    OtherFaceParams others;
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
    int streak = 0;          // results in a row matching streakLabel
    int streakLabel = -1;
    int checks = 0;            // results so far
    bool everMatched = false;  // a result matched a label by its photos
    std::string netLabel;      // the network's likeliest answer in the latest check ("other": none)
    float netProb = -1;        // ... and its probability; -1: the network was off
    std::string photo;         // labeled tracks: whether the latest check took a photo, or why not
    std::vector<Embedding> others;  // kept while it matched no label (other people's faces)
    double otherAt = -1;            // when the last of them was kept
};

// How the face network changes a match by the photos (LabelLibrary::match).
// `threshold` is the best label's; `probs` the network's answer (each label's
// probability, then "other"), empty when it's off.
struct LabelVerdict {
    int label = -1;        // the label matched; -1: none
    bool relaxed = false;  // matched only because the network was sure
    bool vetoed = false;   // the photos matched, but the network was sure it's someone else
};
LabelVerdict combineMatch(const LabelMatch& gallery, float threshold,
                          const std::vector<float>& probs, const RecognizerParams& params);

// Gives face tracks custom labels (specific people). Each frame it sends a few
// face crops to a background thread for embedding, then compares the results
// with the labels' samples (and asks the face network, when it's on). A track
// gets a label after several consistent matches and keeps it until it stops
// matching; each label (a specific person) belongs to at most one track.
//
// With a Reinforcer, it also takes photos of recognized faces for their label
// (see CaptureParams) and adds them to the label at once, and keeps the
// embeddings of faces that match no label. Only this (the tracking) thread
// reads or changes the labels; the embedding thread gets crops, and the
// reinforcer copies.
class Recognizer {
public:
    // Loads the embedding model now, so loading errors show up at startup.
    Recognizer(LabelLibrary& library, const ClassRegistry& classes, IEmbedder& embedder,
               RecognizerParams params = {}, Reinforcer* reinforcer = nullptr);
    ~Recognizer();  // hands the other faces it kept to the reinforcer
    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    // Call after the tracker each frame, before drawing on the frame.
    void update(const cv::Mat& frame, const std::vector<Track>& tracks, double now);

    int labelOf(int trackId) const;                    // -1 if none
    const std::string* labelName(int trackId) const;   // null if none
    const TrackIdentity* identity(int trackId) const;  // null if never looked at
    // The face network's state, e.g. "off: needs other faces (3 of 20 seen)";
    // empty without a reinforcer.
    const std::string& netStatus() const { return netStatus_; }

    // Developer aid: one CSV line per embedding result (time, track, scores).
    void setDebugStream(std::ostream* out) { debug_ = out; }
    // Waits until every crop sent has been embedded (tests).
    void waitIdle();

private:
    struct Job {
        int trackId;
        double time;   // when the crop was taken
        cv::Mat crop;  // a copy: the frame gets drawn on while the worker runs
        cv::Rect2f box;
        std::vector<cv::Point2f> keypoints;
        float conf = 0;
        bool detected = false;  // confidently detected in that frame
        bool photo = false;     // it may be photographed: embed it mirrored too, keep the crop
    };
    struct Result {
        Job job;
        Embedding embedding, mirrored;
        float sharpness = 0;
    };

    void work();
    void apply(Result& result);
    void vote(int trackId, TrackIdentity& id, int matched, float score);
    bool mayPhotograph(const TrackIdentity& id, const Job& job) const;
    void considerPhoto(TrackIdentity& id, const Result& result, const LabelMatch& m);
    void takePhoto(TrackIdentity& id, const Result& result);
    void handOver(TrackIdentity& id);
    void useLatestNet();
    double lastPhoto(int label) const;

    LabelLibrary& library_;
    const ClassRegistry& classes_;
    IEmbedder& embedder_;
    RecognizerParams params_;
    Reinforcer* reinforcer_;
    float threshold_;  // the model's default; a label's own threshold overrides it
    std::map<int, TrackIdentity> identities_;
    std::ostream* debug_ = nullptr;
    double now_ = 0;
    std::shared_ptr<const FaceNet> net_;  // in use; null while off
    int netVersion_ = -1;
    std::string netStatus_;
    std::map<int, double> lastPhoto_;  // label index -> when its last photo was taken

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable wake_, idle_;
    std::deque<Job> jobs_;
    std::vector<Result> results_;
    int inFlight_ = 0;
    bool stop_ = false;
};
