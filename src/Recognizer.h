// Recognizer: gives face tracks a name. For each face track, about every recheck interval:
//   1. (tracking thread) the face is cropped out of the frame and queued as a Job;
//   2. (worker thread) the crop is embedded with the SFace network;
//   3. (tracking thread, a later update()) the embedding is compared with the labels' samples
//      and, when it's on, with the face network (FaceNet), and the answer counts as one vote;
//   4. a track gets a label after several agreeing votes, and loses it after several misses in a
//      row.
// Embedding runs on a worker so the tracking loop never waits for the network; the result simply
// shows up a frame or more later.
//
// Threading: update(), the identities, the LabelLibrary and everything read from the Reinforcer
// belong to the tracking thread. The worker only gets copied crops (guarded queue, mutex_) and
// returns embeddings. Trained FaceNets come from the Reinforcer's own thread as
// shared_ptr<const FaceNet>, which is never modified after it's published.
#pragma once
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "ClassRegistry.h"
#include "Embedder.h"
#include "FaceNet.h"
#include "Labels.h"
#include "Reinforcement.h"
#include "Tracker.h"

// Tuning knobs for recognition. Times are in seconds, on the clock passed to update().
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
    double litCooldown = 2.0;  // s after a face was lit (or the TARGET) before it's photographed
    CaptureParams capture;
    OtherFaceParams others;
};

// What recognition knows about one track. Each embedding result is one vote for a label (see
// Recognizer::vote); the fields below are that bookkeeping, plus what the debug view and the
// photo and other-face logic need.
struct TrackIdentity {
    int label = -1;            // index into LabelLibrary::labels(); -1: none
    std::map<int, int> votes;  // label index -> matching results
    int misses = 0;            // results in a row that didn't match the assigned label
    float score = 0;           // the assigned label's latest score
    double askedAt = -1;       // when this track was last sent for embedding
    float askedSize = 0;       // its size then
    // a crop of this track is queued or being embedded; no second one is sent until it's applied
    bool pending = false;
    std::string closest;     // the closest label in the latest check, matched or not
    float closestScore = 0;  // ... and its score
    int streak = 0;          // results in a row matching streakLabel
    // the label the latest result matched (-1: none); `streak` counts the results in a row that did
    int streakLabel = -1;
    int checks = 0;            // results so far
    bool everMatched = false;  // a result matched a label by its photos
    std::string netLabel;      // the network's likeliest answer in the latest check ("other": none)
    float netProb = -1;        // ... and its probability; -1: the network was off
    std::string photo;         // labeled tracks: whether the latest check took a photo, or why not
    double litAt = -1;         // last time it was lit or the TARGET
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
// The photos stay in charge. The network can relax a near miss (the best label scored within
// `relax` below its threshold, still leads the runner-up by `minMargin`, and the network gives
// that label at least `netSure`) or veto a match (it gives "other" at least `netSure`). Without a
// network (`probs` empty) the photos decide alone.
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
    // Loads the embedding model now, so loading errors show up at startup. Starts the worker
    // thread. The library, classes, embedder and reinforcer (optional) must outlive this object.
    Recognizer(LabelLibrary& library, const ClassRegistry& classes, IEmbedder& embedder,
               RecognizerParams params = {}, Reinforcer* reinforcer = nullptr);
    ~Recognizer();  // hands the other faces it kept to the reinforcer
    Recognizer(const Recognizer&) = delete;
    Recognizer& operator=(const Recognizer&) = delete;

    // Call after the tracker each frame, before drawing on the frame. `lit`:
    // face tracks that are the TARGET or have the laser dot on them now (they look
    // unlike themselves, so they aren't photographed). `now`: the frame's time in seconds.
    // Applies the embeddings that finished since the last call, forgets tracks that are gone,
    // then queues crops of the faces that are due a look.
    void update(const cv::Mat& frame, const std::vector<Track>& tracks, double now,
                const std::set<int>& lit = {});

    // Lookups for drawing and logging, valid until the next update() (tracking thread only).
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
    // A face crop on its way to the worker thread.
    struct Job {
        int trackId;
        double time;   // when the crop was taken
        cv::Mat crop;  // a copy: the frame gets drawn on while the worker runs
        cv::Rect2f box;
        std::vector<cv::Point2f> keypoints;
        float conf = 0;
        bool detected = false;  // confidently detected in that frame
        bool lit = false;       // lit, or only just
        bool photo = false;     // it may be photographed: embed it mirrored too, keep the crop
    };
    // What the worker returns. `mirrored` is only computed for a crop that may become a photo and
    // is sharp enough; `sharpness` is the face's focus score (see sharpness() in Reinforcement.h).
    struct Result {
        Job job;
        Embedding embedding, mirrored;
        float sharpness = 0;
    };

    // The worker thread: embeds queued crops until stopped.
    void work();
    // Uses one finished result: matches it against the labels, votes, then considers a photo (a
    // labeled track) or keeping the face as someone else's (an unlabeled one).
    void apply(Result& result);
    // Counts a result (the label it supports, or -1) toward giving the track a label or taking it
    // away.
    void vote(int trackId, TrackIdentity& id, int matched, float score);
    // Whether a crop is worth embedding mirrored too because it may become a photo. Decided before
    // the embedding exists, from what's already known.
    bool mayPhotograph(const TrackIdentity& id, const Job& job) const;
    // Runs the capture gates on a result and takes the photo if all pass.
    void considerPhoto(TrackIdentity& id, const Result& result, const LabelMatch& m);
    // Adds the photo to the label at once and queues saving it.
    void takePhoto(TrackIdentity& id, const Result& result);
    // Gives the reinforcer the other faces kept from a track, if it qualifies.
    void handOver(TrackIdentity& id);
    // Picks up a network the reinforcer published since the last frame.
    void useLatestNet();
    // When the label's last photo was taken (this run, same clock as update()); -1 if none yet.
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
    // Reinforcer::version() at the last refresh: net() is only fetched again when it changes.
    int netVersion_ = -1;
    std::string netStatus_;
    std::map<int, double> lastPhoto_;  // label index -> when its last photo was taken

    // Worker side, guarded by mutex_: jobs_ wait to be embedded and results_ are finished;
    // inFlight_ counts the jobs queued or being embedded.
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable wake_, idle_;
    std::deque<Job> jobs_;
    std::vector<Result> results_;
    int inFlight_ = 0;
    bool stop_ = false;
};
