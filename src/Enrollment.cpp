#include "Enrollment.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "Recognizer.h"  // RecognizerParams

namespace fs = std::filesystem;

namespace {

constexpr int kRecommendedPhotos = 5;  // fewer usable photos than this gets a tip
constexpr float kOutlierGap = 0.15f;  // a photo agreeing this much less than the average stands out

// What a sync did to one label, for its report.
struct Change {
    std::string what;                // "new label", "updated", ...
    std::vector<std::string> lines;  // one per photo looked at or removed
};

std::string number(float x) {
    char text[16];
    std::snprintf(text, sizeof(text), "%.2f", x);
    return text;
}

// How much a detection stands out in a photo: confident and large.
float prominence(const Detection& d, cv::Size image) {
    return (d.weak ? 0.5f : 1.0f) * d.conf * d.box.area() / (image.width * image.height);
}

// The detection that stands out most, of class `only` if that's given.
const Detection* mostProminent(const std::vector<Detection>& detections, cv::Size image,
                               ClassId only = -1) {
    const Detection* best = nullptr;
    for (const Detection& d : detections)
        if ((only < 0 || d.classId == only) &&
            (!best || prominence(d, image) > prominence(*best, image)))
            best = &d;
    return best;
}

std::string extensionOf(const std::string& name) {
    std::string ext = fs::path(name).extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return ext;
}

std::string unreadableNote(const std::string& name) {
    const std::string ext = extensionOf(name);
    if (ext == ".heic" || ext == ".heif" || ext == ".avif")
        return "can't read " + ext.substr(1) +
               " photos; convert it to JPEG, e.g. sips -s format jpeg " + name + " --out " +
               fs::path(name).replace_extension(".jpg").string();
    return "can't read this file as an image";
}

// Reads a photo. Empty if OpenCV can't read the file (it isn't an image, or its
// header claims an impossibly large one): that's down to the file, so the photo
// is remembered as unusable. Running out of memory may not happen next time, so
// that still throws.
cv::Mat readPhoto(const std::string& path) {
    try {
        return cv::imread(path);
    } catch (const cv::Exception& e) {
        if (e.code == cv::Error::StsNoMem) throw;
        return {};
    }
}

// Why a photo couldn't be looked at this time (a model failed, memory ran out).
// Such a photo isn't remembered, so the next sync looks at it again.
std::string retryNote(const std::string& error) {
    return "looking at it failed (" + error + "); the next sync tries it again";
}

bool faceMissing(const PhotoRecord& r) { return !r.used && r.note.rfind("no ", 0) == 0; }

// Looks at one photo: finds the face and embeds it, and the photo mirrored
// left-right, since the live camera view is mirrored.
PhotoRecord examine(const PhotoFile& file, const cv::Mat& image,
                    const std::vector<Detection>& detections, const ClassRegistry& classes,
                    IEmbedder& embedder) {
    PhotoRecord r;
    r.file = file;
    if (image.empty()) {
        r.note = unreadableNote(file.name);
        return r;
    }
    const Detection* face = mostProminent(detections, image.size(), classes.face());
    if (!face) {
        r.note = "no face found";
        return r;
    }
    cv::Rect2f box = face->box;
    std::vector<cv::Point2f> keypoints = face->keypoints;
    r.conf = face->conf;
    r.note = "face found, confidence " + number(face->conf);
    r.roi = cv::Rect(box);
    r.embedding = embedder.embed(image, box, keypoints);
    if (r.embedding.empty()) {
        r.note = "the face is too small (" + std::to_string(r.roi.width) + "x" +
                 std::to_string(r.roi.height) + " px)";
        return r;
    }
    cv::Mat mirrored;
    cv::flip(image, mirrored, 1);
    mirrorDetection(image.cols, box, keypoints);
    r.mirrored = embedder.embed(mirrored, box, keypoints);
    r.used = true;
    return r;
}

// Brings one label folder up to date (its plan isn't empty). Writes its .embeddings.yml.
Change syncFolder(const LabelFolder& f, const SyncPlan& plan, Detector& detector,
                  const ClassRegistry& classes, IEmbedder& embedder) {
    Change change;
    const bool isNew = f.cache.photos.empty() && f.cache.model.empty();
    change.what = isNew ? "new label"
                  : plan.redoAll && !plan.reason.empty()
                      ? "every photo looked at again, as " + plan.reason
                      : "updated";
    for (const std::string& name : plan.removed) change.lines.push_back(name + ": removed");
    const std::string cacheFile = (fs::path(f.path) / kCacheFile).string();
    auto pathOf = [&](const PhotoFile& p) { return (fs::path(f.path) / p.name).string(); };

    // The class is written as an earlier version, which also reads this file,
    // would: set in label.yml or not.
    LabelCache cache{embedder.modelId(), kFaceClass, f.settings.oldClass.empty(), false, plan.keep};
    for (const PhotoFile& photo : plan.look) {
        PhotoRecord r;
        std::string failed;  // why looking at it failed, when it may not fail next time
        try {
            const cv::Mat image = readPhoto(pathOf(photo));
            // Only faces are looked for, so only the face detector is needed.
            const std::vector<Detection> faces =
                image.empty() ? std::vector<Detection>{} : detector.detect(image, {classes.face()});
            r = examine(photo, image, faces, classes, embedder);
        } catch (const std::exception& e) {
            failed = e.what();
            r.note = retryNote(failed);
        }
        const bool changed = std::count(plan.changed.begin(), plan.changed.end(), photo.name) > 0;
        change.lines.push_back(photo.name + (changed ? " (changed)" : "") + ": " +
                               (r.used ? "" : "not used, ") + r.note);
        if (failed.empty()) cache.photos.push_back(std::move(r));
    }
    std::sort(cache.photos.begin(), cache.photos.end(),
              [](const PhotoRecord& a, const PhotoRecord& b) { return a.file.name < b.file.name; });
    writeCache(cacheFile, cache);
    return change;
}

// What a label is like after a sync: usable photos, how well they agree, and tips.
void printReport(const LabelFolder& f, const Change* change, const LabelLibrary& library,
                 const IEmbedder& embedder) {
    const std::string settingsFile = (fs::path(f.path) / kSettingsFile).string();
    std::printf("Label '%s': %s\n", f.name.c_str(), change ? change->what.c_str() : "unchanged");
    if (change)
        for (const std::string& line : change->lines) std::printf("  %s\n", line.c_str());
    if (!f.settings.oldKeys.empty()) {
        std::string keys;
        for (size_t i = 0; i < f.settings.oldKeys.size(); ++i)
            keys += (i == 0 ? "" : " and ") + f.settings.oldKeys[i];
        std::printf(
            "  Note: %s in %s %s from an earlier version and not used now (every label "
            "is a face, found in each photo).\n",
            keys.c_str(), settingsFile.c_str(), f.settings.oldKeys.size() == 1 ? "is" : "are");
    }

    // The photos in the folder: one that couldn't be looked at this time isn't in the cache.
    const int total = static_cast<int>(f.photos.size());
    const int used = f.usedPhotos();
    const bool missing = std::any_of(f.cache.photos.begin(), f.cache.photos.end(), faceMissing);
    const int index = library.find(f.name);
    if (total == 0) {
        std::printf("  No photos yet: put 5-10 photos of the person's face in %s\n",
                    f.path.c_str());
        return;
    }
    if (index < 0) {
        std::printf("  None of its %d photo(s) is usable, so it can't be recognized.\n", total);
        if (missing)
            std::printf(
                "  Each photo needs a clear view of the face, from the front or a little "
                "to the side.\n");
        return;
    }

    const CustomLabel& label = library.labels()[index];
    const float threshold = label.threshold > 0 ? label.threshold : embedder.defaultThreshold();
    const std::vector<PhotoAgreement> agreement = photoAgreement(label);
    float mean = 0;
    for (const PhotoAgreement& a : agreement) mean += a.agreement;
    mean /= agreement.size();
    std::printf("  %d of %d photo(s) usable", used, total);
    if (agreement.size() > 1)
        std::printf("; they agree at %.2f on average (recognition needs %.2f)", mean, threshold);
    std::printf(".\n");
    // A photo that agrees much less than the rest probably shows something
    // else, or the face from a very unusual angle. (Two photos have no "rest".)
    if (agreement.size() > 2)
        for (const PhotoAgreement& a : agreement)
            if (a.agreement < mean - kOutlierGap || a.agreement < threshold)
                std::printf(
                    "  Check %s: it agrees with the others only at %.2f. Does it show the "
                    "same person?\n",
                    a.image.c_str(), a.agreement);
    if (missing) std::printf("  Photos where no face was found aren't used.\n");
    if (used < kRecommendedPhotos)
        std::printf("  Tip: 5-10 photos from different angles and in different light work best.\n");
}

// Warns about pairs of labels that the recognizer would have trouble telling
// apart. Only pairs with a label in `changed`, unless `all`.
void warnLookAlikes(const LabelLibrary& library, const std::map<std::string, Change>& changed,
                    bool all, const IEmbedder& embedder) {
    const float margin = RecognizerParams{}.minMargin;
    const std::vector<CustomLabel>& labels = library.labels();
    auto lookAlike = [](const LabelConfusion& c) {
        return c.ambiguous > 0 && c.ambiguous * 4 >= c.photos;  // a quarter of the photos or more
    };
    for (size_t i = 0; i < labels.size(); ++i)
        for (size_t j = i + 1; j < labels.size(); ++j) {
            const CustomLabel& a = labels[i];
            const CustomLabel& b = labels[j];
            if (!all && !changed.count(a.name) && !changed.count(b.name)) continue;
            const float fallback = embedder.defaultThreshold();
            const LabelConfusion ab =
                labelConfusion(a, b, b.threshold > 0 ? b.threshold : fallback, margin);
            const LabelConfusion ba =
                labelConfusion(b, a, a.threshold > 0 ? a.threshold : fallback, margin);
            if (!lookAlike(ab) && !lookAlike(ba)) continue;
            std::printf(
                "Warning: '%s' and '%s' look alike. Photos of %s score %.2f as %s and "
                "%.2f as %s on average; photos of %s score %.2f as %s and %.2f as %s. Recognition "
                "needs a lead of %.2f, so it may mix them up or show neither. Photos showing how "
                "they differ help; if they are the same person, keep one label.\n",
                a.name.c_str(), b.name.c_str(), a.name.c_str(), ab.ownScore, a.name.c_str(),
                ab.otherScore, b.name.c_str(), b.name.c_str(), ba.ownScore, b.name.c_str(),
                ba.otherScore, a.name.c_str(), margin);
        }
}

// True if both files hold the same bytes.
bool sameContent(const fs::path& a, const fs::path& b) {
    if (fs::file_size(a) != fs::file_size(b)) return false;
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    return std::equal(std::istreambuf_iterator<char>(fa), std::istreambuf_iterator<char>(),
                      std::istreambuf_iterator<char>(fb));
}

// Copies a photo into the label's folder under a name not used yet. Returns
// false if the same photo is there already.
bool storePhoto(const std::string& path, const std::string& folder) {
    const fs::path source(path);
    fs::path target = fs::path(folder) / source.filename();
    for (int n = 2; fs::exists(target); ++n) {
        if (sameContent(source, target)) return false;
        target = fs::path(folder) /
                 (source.stem().string() + "_" + std::to_string(n) + source.extension().string());
    }
    fs::copy_file(source, target);
    return true;
}

// "2 new, 1 changed, 1 removed", or why every photo has to be looked at again.
std::string describe(const SyncPlan& plan) {
    if (plan.redoAll && !plan.reason.empty()) return plan.reason;
    std::string text;
    auto add = [&](size_t n, const char* what) {
        if (n == 0) return;
        text += (text.empty() ? "" : ", ") + std::to_string(n) + " " + what;
    };
    add(plan.look.size() - plan.changed.size(), "new");
    add(plan.changed.size(), "changed");
    add(plan.removed.size(), "removed");
    return text + " photo(s)";
}

}  // namespace

int syncLabels(LabelLibrary& library, Detector& detector, const ClassRegistry& classes,
               IEmbedder& embedder, bool all) {
    std::map<std::string, Change> changes;
    for (const LabelFolder& f : library.folders()) {
        if (!f.problem.empty()) continue;
        const SyncPlan plan = planSync(f.cache, f.photos, embedder.modelId());
        if (plan.empty()) continue;
        changes[f.name] = syncFolder(f, plan, detector, classes, embedder);
    }
    library.load();

    if (all && library.folders().empty())
        std::printf(
            "No labels in %s. Make a folder there for each person (e.g. %s/alex/) and put "
            "5-10 photos of their face in it.\n",
            library.dir().c_str(), library.dir().c_str());
    for (const LabelFolder& f : library.folders()) {
        if (!f.problem.empty()) {
            std::printf("Label '%s' can't be used: %s\n", f.name.c_str(), f.problem.c_str());
            continue;
        }
        const auto change = changes.find(f.name);
        if (change == changes.end() && !all) continue;
        printReport(f, change == changes.end() ? nullptr : &change->second, library, embedder);
    }
    warnLookAlikes(library, changes, all, embedder);
    if (!library.loosePhotos().empty())
        std::printf(
            "%zu photo(s) in %s aren't in a label folder (e.g. %s). Each label is a folder "
            "of photos, e.g. %s/alex/.\n",
            library.loosePhotos().size(), library.dir().c_str(),
            library.loosePhotos().front().c_str(), library.dir().c_str());
    return static_cast<int>(changes.size());
}

std::vector<std::string> missingForSync(const std::vector<std::string>& loaded) {
    if (std::find(loaded.begin(), loaded.end(), "face") != loaded.end()) return {};
    return {"face"};
}

void enroll(const std::string& name, const std::vector<std::string>& images, Detector& detector,
            const ClassRegistry& classes, IEmbedder& embedder, LabelLibrary& library,
            bool mustExist) {
    if (const std::string why = labelNameProblem(name); !why.empty())
        throw std::runtime_error("'" + name + "' can't be a label name: " + why);
    const ClassId sameName = classes.find(name);
    if (sameName >= 0 && classes.kind(sameName) != ClassKind::Custom)
        throw std::runtime_error("'" + name + "' is already a class name; pick another label name");
    if (images.empty()) throw std::runtime_error("--enroll needs at least one photo");

    const int existing = library.findFolder(name);
    if (mustExist && existing < 0)
        throw std::runtime_error("No label named '" + name +
                                 "' (see --list-labels; --enroll makes a new one)");
    const LabelFolder* f = existing >= 0 ? &library.folders()[existing] : nullptr;
    if (f && !f->problem.empty())
        throw std::runtime_error("Label '" + name + "' can't be used: " + f->problem);
    const std::string folder = library.folder(name);
    // Check every photo before copying any.
    for (const std::string& path : images) {
        if (!fs::is_regular_file(path)) throw std::runtime_error("No such photo: " + path);
        if (!isPhotoName(path))
            throw std::runtime_error(path + " isn't a photo (JPEG, PNG and the like)");
        if (!cv::haveImageReader(path))
            throw std::runtime_error("Can't read " + path + ": " + unreadableNote(path));
    }

    fs::create_directories(folder);
    int copied = 0;
    for (const std::string& path : images) copied += storePhoto(path, folder);
    std::printf("Copied %d photo(s) to %s", copied, folder.c_str());
    if (copied < static_cast<int>(images.size()))
        std::printf(" (%d were there already)", static_cast<int>(images.size()) - copied);
    std::printf("\n");

    library.load();
    syncLabels(library, detector, classes, embedder, false);
    if (library.find(name) < 0)
        throw std::runtime_error("'" + name +
                                 "' has no usable photos yet, so it can't be recognized (its "
                                 "photos are in " +
                                 folder + ")");
}

void listLabels(const LabelLibrary& library) {
    const std::string& dir = library.dir();
    if (library.folders().empty()) {
        std::cout << "No labels in " << dir << ". Make a folder there for each person (e.g. " << dir
                  << "/alex/) and put 5-10 photos of their face in it, or use --enroll.\n";
    } else {
        std::cout << "Labels in " << dir << ":\n";
    }
    size_t width = 0;
    for (const LabelFolder& f : library.folders()) width = std::max(width, f.name.size());
    bool pending = false;
    for (const LabelFolder& f : library.folders()) {
        std::cout << "  " << f.name << std::string(width - f.name.size(), ' ') << "  ";
        if (!f.problem.empty()) {
            std::cout << "can't be used: " << f.problem << '\n';
            continue;
        }
        std::cout << f.photos.size() << " photo(s), " << f.usedPhotos() << " usable";
        if (!f.plan.empty()) {
            pending = true;
            std::cout << "; not synced yet: " << describe(f.plan);
        } else if (library.find(f.name) < 0) {
            std::cout << "; can't be recognized (see --sync-labels)";
        } else if (f.usedPhotos() < kRecommendedPhotos) {
            std::cout << "; 5-10 photos work best";
        }
        std::cout << '\n';
    }
    if (pending)
        std::cout << "New and changed photos are looked at when the tracker starts, or now with "
                     "--sync-labels.\n";
    if (!library.loosePhotos().empty())
        std::cout << library.loosePhotos().size() << " photo(s) in " << dir
                  << " aren't in a label folder; each label is a folder of photos.\n";
}

void renameLabel(const std::string& from, const std::string& to, const LabelLibrary& library,
                 const ClassRegistry& classes) {
    if (library.findFolder(from) < 0)
        throw std::runtime_error("No label named '" + from + "' (see --list-labels)");
    if (const std::string why = labelNameProblem(to); !why.empty())
        throw std::runtime_error("'" + to + "' can't be a label name: " + why);
    const ClassId sameName = classes.find(to);
    if (sameName >= 0 && classes.kind(sameName) != ClassKind::Custom)
        throw std::runtime_error("'" + to + "' is already a class name; pick another label name");
    const fs::path source = library.folder(from), target = library.folder(to);
    // (A different case of the same name is the same folder on macOS.)
    if (fs::exists(target) && !fs::equivalent(source, target))
        throw std::runtime_error("There is already a label or folder named '" + to + "'");
    fs::rename(source, target);
    std::cout << "Renamed label '" << from << "' to '" << to << "' (" << target.string() << ")\n";
}

std::string removeLabel(const std::string& name, const LabelLibrary& library) {
    const int index = library.findFolder(name);
    if (index < 0) throw std::runtime_error("No label named '" + name + "' (see --list-labels)");
    const size_t photos = library.folders()[index].photos.size();
    const fs::path removed = fs::path(library.dir()) / ".removed";
    fs::create_directories(removed);
    fs::path target = removed / name;
    for (int n = 2; fs::exists(target); ++n) target = removed / (name + "_" + std::to_string(n));
    fs::rename(library.folder(name), target);
    std::cout << "Removed label '" << name << "'. Its folder, with its " << photos
              << " photo(s), is now " << target.string()
              << ": move it back to restore the label, or delete it to free the space.\n";
    return target.string();
}
