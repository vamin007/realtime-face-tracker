// Reinforcement: how the tracker improves itself while it runs, and the data behind it.
//   - capturePolicy(): the gates a recognized face has to pass before it is photographed for its
//     label. A pure function, so each gate is unit-tested on its own.
//   - sharpness(), maxSimilarity(), photoToReplace(), autoPhotoName(): helpers for those gates and
//     for keeping a label's automatic photos to a fixed number.
//   - NegativeStore: embeddings of faces that matched no label, kept as the "other people"
//     examples the face network (FaceNet.h) trains on.
//   - Reinforcer: a background thread that writes the photos, collects those embeddings and
//     retrains the face network, so the tracking loop never waits for disk or training.
//
// Threading: the tracking thread (Recognizer) decides what to photograph and queues the work. The
// Reinforcer's thread does all file writes and training, on copies of the data, and publishes each
// trained network as a shared_ptr<const FaceNet> that is never modified afterwards.
#pragma once
#include <atomic>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <string>
#include <thread>
#include <vector>

#include "FaceNet.h"
#include "Labels.h"

// Reinforcement: while a person is recognized, the tracker now and then takes
// a photo of their face for their label (auto_<date>_<time>.jpg in the label's
// folder), so the label learns how they look from more angles and in more
// light. A wrong photo would teach it the wrong face, which would then be
// recognized more readily and photographed again, so a photo is only taken of
// a face recognized clearly and steadily, seen well, and different enough from
// the label's photos. Faces that match no label are kept too, as embeddings
// only (never photos), so the face network (FaceNet) learns what other people
// look like.

// When a recognized face is worth a photo for its label.
struct CaptureParams {
    int minStreak = 5;            // matching results in a row for the track's label
    float scoreLead = 0.05f;      // the label's score is at least this above its threshold
    float minMargin = 0.1f;       // ... and this above the runner-up's
    float minWidth = 80;          // px: the face's width in the frame
    float minSharpness = 15;      // see sharpness()
    float maxSimilarity = 0.85f;  // to any of the label's samples: it must be a new view
    double interval = 30;         // s between photos of one label
    int maxPhotos = 40;           // automatic photos per label; then the most redundant goes
};

// What's known about one recognition result of a labeled track.
struct CaptureInput {
    bool enabled = true;    // auto-photos are on, for this label too
    int streak = 0;         // results in a row matching the track's label, this one included
    float score = 0;        // the label's score for this face
    float threshold = 0;    // the label's threshold
    float runnerUp = 0;     // the next best label's score
    bool detected = false;  // confidently detected in the frame the crop is from (not coasting)
    float width = 0;        // px
    float sharpness = 0;
    bool lit = false;       // the TARGET or under the laser, now or a moment ago
    float similarity = 0;   // the highest similarity to the label's samples
    double now = 0;         // s, when the crop was taken
    double lastPhoto = -1;  // s, when the label's last photo was; -1: none yet
};

// Whether to take the photo, or the first reason not to. The gates are checked in the order
// listed.
enum class CaptureGate {
    Take,
    Off,
    Unstable,
    NotDetected,
    TooSmall,
    Lit,
    TooSoon,
    LowScore,
    CloseRunnerUp,
    Blurry,
    NotNew,
};
CaptureGate capturePolicy(const CaptureInput& in, const CaptureParams& params = {});
const char* describe(CaptureGate gate);  // "taken", "too soon", ...

// How sharp a face is: the variance of the Laplacian over the middle 60% of
// its box (eyes, nose and mouth; hair and background left out), in gray, with
// the box scaled to 112 x 112 (the size a face is embedded at). 0 outside the
// image. On hands_face.mov, clear faces measured 19-200 (median 71), a hand
// sweeping across the face 9.5; over the whole box that hand measured 45, among
// clear but smooth faces (35-48), hence the middle only. 15 is the threshold.
// (The Laplacian is an edge filter, a second derivative: sharp detail gives large and varied
// responses, blur gives flat ones, so the variance of its output measures focus.)
float sharpness(const cv::Mat& image, const cv::Rect2f& face);
// The highest similarity of e to any of the label's samples.
float maxSimilarity(const CustomLabel& label, const Embedding& e);
// The label's automatic photos (auto_*.jpg), each counted once.
int autoPhotoCount(const CustomLabel& label);
// When the label has more than maxPhotos automatic photos, the one to delete:
// the automatic photo most like another of its photos (the one that adds the
// least). Never one of the user's own photos. Empty while there's room.
std::string photoToReplace(const CustomLabel& label, int maxPhotos);
// auto_YYYYMMDD_HHMMSS.jpg for local time t, or with _2, _3, ... while taken(name).
std::string autoPhotoName(std::time_t t, const std::function<bool(const std::string&)>& taken);

// Which faces are kept as other people's: their best label's score is well
// below its threshold, in every result of the track (it never matched), and
// the track was checked often enough to be sure.
struct OtherFaceParams {
    float below = 0.1f;     // the best label's score is this far below its threshold
    double interval = 2.0;  // s between faces kept from one track
    int perTrack = 10;      // faces kept from one track at most
    int minChecks = 5;      // results a track needs before its faces count
};

// Other people's faces: embeddings only, never photos, in labels/.negatives.yml.
// Each face remembers the track it came from, so it's known how many people
// they show (roughly: one person can be several tracks). These are the negatives for FaceNet's
// "other" output: examples of "none of the labels".
class NegativeStore {
public:
    static constexpr int kCapacity = 500;
    explicit NegativeStore(int capacity = kCapacity) : capacity_(capacity) {}

    // Adds a face seen on `source` (a track). When it's full, the face most
    // like another one goes, which may be this one.
    void add(const Embedding& e, const std::string& source);
    const std::vector<Embedding>& faces() const { return faces_; }
    int people() const;  // how many sources
    int capacity() const { return capacity_; }

    // Faces of another embedding model are no use: they're dropped (false).
    bool load(const std::string& file, const std::string& model);
    void save(const std::string& file, const std::string& model) const;  // throws if it can't

private:
    int capacity_;
    std::vector<Embedding> faces_;
    // sources_[i] is the track faces_[i] came from.
    std::vector<std::string> sources_;
};

// In the labels directory, written by the tracker.
constexpr const char* kNegativesFile = ".negatives.yml";

// The labels' samples, for training FaceNet (without other faces).
FaceNetData faceNetData(const LabelLibrary& library, const std::string& model);

// The face network's state, as --list-labels shows it: "on (...)", "off: needs
// other faces (3 of 20 seen)", or "not trained yet (...)". Changes nothing.
std::string faceNetStatus(const LabelLibrary& library, const std::string& model);
// Trains it now if it needs training (and there's enough data), and saves it,
// for --sync-labels. Returns its state.
std::string trainFaceNet(const LabelLibrary& library, const std::string& model);

// A background thread that does the slow part of reinforcement: it writes the
// photos and their records, keeps the other faces, and trains the face network
// again whenever they change. It gets copies of what it needs (a photo, the
// labels' samples), so it never reads the labels the tracking thread uses, and
// it hands each newly trained network over whole: net() is never changed once
// returned. The public methods are safe to call from the tracking thread: they only queue work
// or read what the thread last published.
class Reinforcer {
public:
    // Loads the network saved in labelsDir if it was trained on exactly these
    // labels' samples and the other faces kept so far; otherwise trains one now
    // (before tracking starts), if there's enough data. `labels.others` is
    // ignored: the reinforcer keeps the other faces itself.
    Reinforcer(std::string labelsDir, FaceNetData labels);
    ~Reinforcer();  // finishes what's queued
    Reinforcer(const Reinforcer&) = delete;
    Reinforcer& operator=(const Reinforcer&) = delete;

    struct Photo {
        std::string folder;    // the label's folder
        cv::Mat image;         // saved as JPEG
        PhotoRecord record;    // added to the folder's .embeddings.yml (its name set)
        std::string replaces;  // an automatic photo to delete to make room; empty if none
    };
    // Queues a photo; `labels`: every label's samples with the photo's.
    void savePhoto(Photo photo, FaceNetData labels);
    // Queues the faces kept from one track of someone who isn't a label.
    void addOthers(std::vector<Embedding> faces);

    std::shared_ptr<const FaceNet> net() const;  // the latest network; null while off
    std::string status() const;                  // "on (...)" or "off: why"
    int version() const { return version_; }     // changes whenever net() or status() do
    int others() const;                          // other faces kept
    void wait();                                 // until everything queued is done

private:
    // One unit of queued work: a photo to save (with every label's samples, to train on), or the
    // faces kept from one track of someone who isn't a label.
    struct Task {
        bool photo = false;
        Photo shot;
        FaceNetData labels;
        std::vector<Embedding> others;
    };
    // The thread's body: takes everything queued, does it, and retrains once if anything changed.
    void work();
    // Does one task: saves a photo and its record, or stores the faces.
    bool run(Task& task);  // true if the network should be trained again
    // Trains on data(), saves the network and publishes it (or why it's off).
    void retrain();
    // The training data now: the labels' samples as of the last photo, plus every kept face.
    FaceNetData data() const;
    // Swaps in a new network and status under the mutex, then bumps version_.
    void publish(std::shared_ptr<const FaceNet> net, std::string status);

    const std::string dir_;
    FaceNetData labels_;   // this thread's copy
    NegativeStore store_;  // used by the thread only (and before it starts)
    std::string run_;      // names this run's tracks in the store
    // Tracks handed over so far this run; numbers the next source name.
    int tracks_ = 0;

    // mutex_ guards the queue, the flags and what is published (net_, status_, others_).
    mutable std::mutex mutex_;
    std::condition_variable wake_, idle_;
    std::deque<Task> queue_;
    bool busy_ = false;
    bool stop_ = false;
    std::shared_ptr<const FaceNet> net_;
    std::string status_;
    int others_ = 0;
    // Counts publications, so the Recognizer notices a new network with one atomic read instead of
    // taking the mutex every frame.
    std::atomic<int> version_{0};
    std::thread thread_;
};
