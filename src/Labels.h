#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "Embedder.h"

// One photo of a labeled person.
struct LabelSample {
    std::string image;     // file name in the label's folder
    cv::Rect roi;          // where the face is in that photo
    bool flipped = false;  // the left-right mirrored copy (the live camera view is mirrored)
    Embedding embedding;
};

// A specific person the tracker was taught to recognize by their face, e.g. "alex".
struct CustomLabel {
    std::string name;
    std::string model;                 // the embedding model its samples were made with
    float threshold = 0;               // similarity needed to recognize it; 0 = the model's default
    std::vector<LabelSample> samples;  // several photos: more angles and light, better recognition
    bool autoPhotos = true;            // photos are taken of the face while it's recognized
};

struct LabelMatch {
    int label = -1;      // index into LabelLibrary::labels(); -1 if nothing matched
    float score = 0;     // the best label's score
    float runnerUp = 0;  // the next best label's score
    int best = -1;       // the best label, matched or not; -1 if there are no labels
};

// label.yml in a label's folder: optional settings, written by hand.
struct LabelSettings {
    float threshold = 0;     // "threshold": similarity needed; 0 = the model's default
    bool autoPhotos = true;  // "auto_photos": photos of the face are taken while it's recognized
    // Settings of an earlier version, which also had object labels. They aren't
    // used; a class other than face means the folder can't be a label.
    std::string oldClass;              // "class"
    std::vector<std::string> oldKeys;  // which of "class" and "whole_photo" the file has
};

// An image file in a label's folder. Its size and time show whether it changed.
struct PhotoFile {
    std::string name;
    long long size = 0;
    long long mtime = 0;  // modification time, ns since the epoch
};

// What a sync found in one photo.
struct PhotoRecord {
    PhotoFile file;
    bool used = false;              // the face was found and embedded
    std::string note;               // what was found, or why the photo isn't used
    cv::Rect roi;                   // where the face is in the photo
    float conf = 0;                 // the detector's confidence in it
    Embedding embedding, mirrored;  // of the photo, and of the photo flipped left-right
};

// .embeddings.yml in a label's folder, written by a sync: what it found in each
// photo, and the model it used (when that changes, every photo is redone). An
// earlier version, which also had object labels, reads and writes the same
// file, so the fields it needs are kept.
struct LabelCache {
    std::string model;        // embedding model
    std::string baseClass;    // the class it looked for: face (or an earlier version's object
                              // class, or empty if that version found nothing to choose)
    bool autoClass = false;   // ... not set in label.yml (that version chose it from the photos)
    bool wholePhoto = false;  // that version's whole photos: each photo showed just the item
    std::vector<PhotoRecord> photos;
};

// The class every label is now. Labels of other classes, made by an earlier
// version, are reported and left alone.
constexpr const char* kFaceClass = "face";

// What a sync has to do for one label folder.
struct SyncPlan {
    bool redoAll = false;              // every photo has to be looked at again ...
    std::string reason;                // ... because of this (empty for a new label)
    std::vector<PhotoRecord> keep;     // records still valid
    std::vector<PhotoFile> look;       // photos to look at: new or changed (all if redoAll)
    std::vector<std::string> changed;  // the ones among them that changed
    std::vector<std::string> removed;  // photos no longer in the folder
    bool empty() const { return !redoAll && look.empty() && removed.empty(); }
};

// Compares the photos in a label's folder with what the last sync found.
// `model` is the embedding model (empty: not checked).
SyncPlan planSync(const LabelCache& cache, const std::vector<PhotoFile>& photos,
                  const std::string& model);

// One folder in the labels directory, as found on disk.
struct LabelFolder {
    std::string name;  // the folder name, which is the label name
    std::string path;
    LabelSettings settings;
    LabelCache cache;
    std::vector<PhotoFile> photos;  // image files in the folder now
    SyncPlan plan;                  // what a sync has to do (the model isn't checked)
    std::string problem;            // why it can't be a label (a bad name, a broken label.yml,
                                    // an earlier version's object label)
    int usedPhotos() const;         // photos the last sync could use
};

// Labels on disk: one folder per label, <dir>/<name>/, holding photos of the
// person, an optional label.yml (settings) and .embeddings.yml (written by a sync).
class LabelLibrary {
public:
    explicit LabelLibrary(std::string dir) : dir_(std::move(dir)) {}

    // Reads every label folder: its settings, photos, and what the last sync
    // found. Folders made by an older version are converted first. labels()
    // gets each label the last sync found usable photos for.
    void load();
    // Writes a label's threshold (label.yml) and samples (.embeddings.yml).
    void save(const CustomLabel& label) const;
    const std::string& dir() const { return dir_; }
    std::string folder(const std::string& name) const { return dir_ + "/" + name; }

    std::vector<CustomLabel>& labels() { return labels_; }
    const std::vector<CustomLabel>& labels() const { return labels_; }
    int find(const std::string& name) const;  // -1 if unknown
    // Adds a photo's samples to a label in memory (its embedding and mirrored
    // copy, as a sync would have), or removes them. Files aren't touched.
    void addPhoto(int label, const PhotoRecord& record);
    void removePhoto(int label, const std::string& image);

    const std::vector<LabelFolder>& folders() const { return folders_; }
    int findFolder(const std::string& name) const;  // -1 if there's no such folder
    // Photos were added, changed or removed since the last sync (or a label's
    // settings changed). Folders with a problem don't count: a sync can't fix them.
    bool needsSync() const;
    // Photos lying directly in dir, outside any label folder.
    const std::vector<std::string>& loosePhotos() const { return loosePhotos_; }

    // The best label for a face's embedding. A label's score is the mean of its
    // 3 most similar samples; the best must reach its threshold and beat the
    // runner-up by at least `minMargin`.
    LabelMatch match(const Embedding& e, float defaultThreshold, float minMargin) const;

private:
    std::string dir_;
    std::vector<CustomLabel> labels_;
    std::vector<LabelFolder> folders_;
    std::vector<std::string> loosePhotos_;
};

// Score of one label for an embedding: the mean of its 3 most similar samples.
float labelScore(const CustomLabel& label, const Embedding& e);
// The similarity a label needs: its own threshold, or `fallback` (the model's).
inline float thresholdOf(const CustomLabel& label, float fallback) {
    return label.threshold > 0 ? label.threshold : fallback;
}

// Label names become class names, so they're limited to letters, digits, _ and -.
bool validLabelName(const std::string& name);
std::string labelNameProblem(const std::string& name);  // why it isn't valid; empty if it is

// Files in a label's folder.
constexpr const char* kSettingsFile = "label.yml";
constexpr const char* kCacheFile = ".embeddings.yml";
constexpr const char* kOldLabelBackup = ".label-v1.yml";  // label.yml of an older version
bool isPhotoName(const std::string& fileName);            // by extension
// A photo the tracker took of a recognized face: auto_YYYYMMDD_HHMMSS.jpg,
// perhaps with _2, _3, ... before the extension.
bool isAutoPhotoName(const std::string& fileName);
// Image files in a folder, sorted by name.
std::vector<PhotoFile> listPhotos(const std::string& folder);

// label.yml. Missing file: the defaults. Throws for a file it can't use. An
// earlier version's class and whole_photo settings are read, not used.
LabelSettings readSettings(const std::string& file);
void writeSettings(const std::string& file, const LabelSettings& settings);
// .embeddings.yml. Returns false (and an empty cache) if it's missing or unreadable.
bool readCache(const std::string& file, LabelCache& cache);
void writeCache(const std::string& file, const LabelCache& cache);
// Adds a photo's record to a label folder's .embeddings.yml, as if a sync had
// looked at it: its size and time are taken from the file. A record of the
// same name is replaced, and `removed`'s (a photo deleted) is dropped. Returns
// false if the folder has no cache to add to (it was never synced).
bool appendPhotoRecord(const std::string& folder, PhotoRecord record,
                       const std::string& removed = "");

// Converts a folder made by an older version, whose label.yml held the
// embeddings, keeping the old file as .label-v1.yml. An object label stays one
// (and is reported as such). Returns false if the folder didn't need it.
bool migrateOldLabel(const std::string& folder);

// How well each photo agrees with the label's other photos: the mean similarity
// of its embedding to theirs (not to its own mirrored copy). One per photo.
struct PhotoAgreement {
    std::string image;
    float agreement = 0;
};
std::vector<PhotoAgreement> photoAgreement(const CustomLabel& label);

// How much label b gets in the way of recognizing label a. Each photo of a is
// scored as the recognizer would score a live crop: against a's other photos
// and against b. It's ambiguous when b reaches `threshold` and a doesn't beat b
// by `margin`: the recognizer would give no label, or b's.
struct LabelConfusion {
    int photos = 0;
    int ambiguous = 0;
    float ownScore = 0;    // mean score of a's photos against a's other photos
    float otherScore = 0;  // ... and against b
};
LabelConfusion labelConfusion(const CustomLabel& a, const CustomLabel& b, float threshold,
                              float margin);
