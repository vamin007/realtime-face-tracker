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

namespace fs = std::filesystem;

namespace {

constexpr int kRecommendedPhotos = 5;  // fewer usable photos than this gets a tip
constexpr float kOutlierGap = 0.15f;  // a photo agreeing this much less than the average stands out

// What a sync did to one label, for its report.
struct Change {
    std::string what;                // "new label", "updated", ...
    std::vector<std::string> lines;  // one per photo looked at or removed
    bool choseClass = false;         // the class was chosen from the photos just now
};

ClassKind embedKind(ClassId base, const ClassRegistry& classes) {
    return classes.kind(base) == ClassKind::Face ? ClassKind::Face : ClassKind::Object;
}

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

// Which kind of thing the photos show: the class that stands out most across
// them. A person is recognized by their face (the face model is made for
// telling people apart), so the face is preferred when the photos show one.
// Unreadable photos have an empty size. -1 if nothing was found.
ClassId chooseClass(const std::vector<std::vector<Detection>>& detections,
                    const std::vector<cv::Size>& sizes, const ClassRegistry& classes) {
    std::map<ClassId, float> weight;
    int photos = 0, withFace = 0;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (sizes[i].area() == 0) continue;
        ++photos;
        if (const Detection* best = mostProminent(detections[i], sizes[i]))
            weight[best->classId] += prominence(*best, sizes[i]);
        withFace += std::any_of(detections[i].begin(), detections[i].end(),
                                [&](const Detection& d) { return d.classId == classes.face(); });
    }
    if (weight.empty()) return -1;
    ClassId base = std::max_element(weight.begin(), weight.end(), [](const auto& a, const auto& b) {
                       return a.second < b.second;
                   })->first;
    if (base == classes.person() && withFace * 2 >= photos) base = classes.face();
    return base;
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

bool itemMissing(const PhotoRecord& r) { return !r.used && r.note.rfind("no ", 0) == 0; }

// Looks at one photo: finds the item (or takes the whole photo) and embeds it,
// and the photo mirrored left-right, since the live camera view is mirrored.
PhotoRecord examine(const PhotoFile& file, const cv::Mat& image,
                    const std::vector<Detection>& detections, ClassId base, bool wholePhoto,
                    const ClassRegistry& classes, IEmbedder& embedder) {
    PhotoRecord r;
    r.file = file;
    if (image.empty()) {
        r.note = unreadableNote(file.name);
        return r;
    }
    const std::string& name = classes.name(base);
    cv::Rect2f box(0, 0, static_cast<float>(image.cols), static_cast<float>(image.rows));
    std::vector<cv::Point2f> keypoints;
    if (wholePhoto) {
        r.note = "the whole photo";
        r.conf = 1;
    } else {
        const Detection* item = mostProminent(detections, image.size(), base);
        if (!item) {
            r.note = "no " + name + " found";
            return r;
        }
        box = item->box;
        keypoints = item->keypoints;
        r.conf = item->conf;
        r.note = name + " found, confidence " + number(item->conf);
    }
    r.roi = cv::Rect(box);
    r.embedding = embedder.embed(image, box, keypoints);
    if (r.embedding.empty()) {
        r.note = "the " + name + " is too small (" + std::to_string(r.roi.width) + "x" +
                 std::to_string(r.roi.height) + " px)";
        return r;
    }
    cv::Mat mirrored;
    cv::flip(image, mirrored, 1);
    mirrorDetection(image.cols, box, keypoints, classes.kind(base) == ClassKind::Face);
    r.mirrored = embedder.embed(mirrored, box, keypoints);
    r.used = true;
    return r;
}

// Brings one label folder up to date (its plan isn't empty). Writes its .embeddings.yml.
Change syncFolder(const LabelFolder& f, ClassId base, const SyncPlan& plan, Detector& detector,
                  const ClassRegistry& classes, Embedders& embedders) {
    Change change;
    const bool isNew = f.cache.photos.empty() && f.cache.baseClass.empty();
    change.what = isNew ? "new label"
                  : plan.redoAll && !plan.reason.empty()
                      ? "every photo looked at again, as " + plan.reason
                      : "updated";
    for (const std::string& name : plan.removed) change.lines.push_back(name + ": removed");
    const std::string cacheFile = (fs::path(f.path) / kCacheFile).string();
    auto pathOf = [&](const PhotoFile& p) { return (fs::path(f.path) / p.name).string(); };
    const bool wholePhoto = f.settings.wholePhoto;

    std::vector<std::vector<Detection>> detections(plan.look.size());
    std::vector<bool> detected(plan.look.size(), false);
    // Why looking at a photo failed, when it may not fail next time (see retryNote).
    std::vector<std::string> failed(plan.look.size());
    if (base < 0) {
        // The class is chosen from the photos, so all of them are looked at first.
        std::vector<cv::Size> sizes(plan.look.size());
        for (size_t i = 0; i < plan.look.size(); ++i) {
            try {
                const cv::Mat image = readPhoto(pathOf(plan.look[i]));
                if (image.empty()) continue;
                detections[i] = detector.detect(image, {});
                sizes[i] = image.size();
                detected[i] = true;
            } catch (const std::exception& e) {
                failed[i] = e.what();
            }
        }
        base = chooseClass(detections, sizes, classes);
        change.choseClass = base >= 0;
        if (base < 0) {
            // Remembered too, so the same photos aren't looked at on every start.
            LabelCache cache;
            cache.autoClass = true;
            cache.wholePhoto = wholePhoto;
            for (size_t i = 0; i < plan.look.size(); ++i) {
                PhotoRecord r;
                r.file = plan.look[i];
                r.note = !failed[i].empty()    ? retryNote(failed[i])
                         : sizes[i].area() > 0 ? "nothing found"
                                               : unreadableNote(r.file.name);
                change.lines.push_back(r.file.name + ": not used, " + r.note);
                if (failed[i].empty()) cache.photos.push_back(std::move(r));
            }
            writeCache(cacheFile, cache);
            return change;
        }
    }

    IEmbedder& embedder = embedders.forKind(embedKind(base, classes));
    LabelCache cache{embedder.modelId(), classes.name(base), f.settings.baseClass.empty(),
                     wholePhoto, plan.keep};
    for (size_t i = 0; i < plan.look.size(); ++i) {
        const PhotoFile& photo = plan.look[i];
        PhotoRecord r;
        if (failed[i].empty()) {
            try {
                const cv::Mat image = readPhoto(pathOf(photo));
                if (!image.empty() && !wholePhoto && !detected[i])
                    detections[i] = detector.detect(image, {});
                r = examine(photo, image, detections[i], base, wholePhoto, classes, embedder);
            } catch (const std::exception& e) {
                failed[i] = e.what();
            }
        }
        if (!failed[i].empty()) r.note = retryNote(failed[i]);
        const bool changed = std::count(plan.changed.begin(), plan.changed.end(), photo.name) > 0;
        change.lines.push_back(photo.name + (changed ? " (changed)" : "") + ": " +
                               (r.used ? "" : "not used, ") + r.note);
        if (failed[i].empty()) cache.photos.push_back(std::move(r));
    }
    std::sort(cache.photos.begin(), cache.photos.end(),
              [](const PhotoRecord& a, const PhotoRecord& b) { return a.file.name < b.file.name; });
    writeCache(cacheFile, cache);
    return change;
}

// What a label is like after a sync: usable photos, how well they agree, and tips.
void printReport(const LabelFolder& f, const Change* change, const LabelLibrary& library,
                 const ClassRegistry& classes, const Embedders& embedders) {
    const std::string settingsFile = (fs::path(f.path) / kSettingsFile).string();
    std::string about = f.cache.baseClass.empty() ? "class not chosen" : f.cache.baseClass;
    if (!f.cache.baseClass.empty() && f.cache.autoClass) about += ", chosen from the photos";
    if (f.cache.wholePhoto) about += ", whole photos";
    std::printf("Label '%s' (%s): %s\n", f.name.c_str(), about.c_str(),
                change ? change->what.c_str() : "unchanged");
    if (change)
        for (const std::string& line : change->lines) std::printf("  %s\n", line.c_str());
    if (change && change->choseClass)
        std::printf("  Not a %s? Say what it is in %s, e.g.  class: %s\n",
                    f.cache.baseClass.c_str(), settingsFile.c_str(),
                    f.cache.baseClass == "cup" ? "bottle" : "cup");

    // The photos in the folder: one that couldn't be looked at this time isn't in the cache.
    const int total = static_cast<int>(f.photos.size());
    const int used = f.usedPhotos();
    const bool missing = std::any_of(f.cache.photos.begin(), f.cache.photos.end(), itemMissing);
    const int index = library.find(f.name);
    if (total == 0) {
        std::printf("  No photos yet: put 5-10 photos of the item in %s\n", f.path.c_str());
        return;
    }
    if (f.cache.baseClass.empty() && !f.cache.photos.empty())
        std::printf("  No object found in the photos. Say what it is in %s, e.g.  class: cup\n",
                    settingsFile.c_str());
    if (index < 0) {
        if (!f.cache.baseClass.empty())
            std::printf("  None of its %d photo(s) is usable, so it can't be recognized.\n", total);
        if (missing && !f.cache.wholePhoto)
            std::printf(
                "  If the photos are close-ups of the item, add  whole_photo: true  to %s\n",
                settingsFile.c_str());
        return;
    }

    const CustomLabel& label = library.labels()[index];
    const float threshold =
        label.threshold > 0
            ? label.threshold
            : embedders.defaultThreshold(embedKind(classes.find(label.baseClass), classes));
    const std::vector<PhotoAgreement> agreement = photoAgreement(label);
    float mean = 0;
    for (const PhotoAgreement& a : agreement) mean += a.agreement;
    mean /= agreement.size();
    std::printf("  %d of %d photo(s) usable", used, total);
    if (agreement.size() > 1)
        std::printf("; they agree at %.2f on average (recognition needs %.2f)", mean, threshold);
    std::printf(".\n");
    // A photo that agrees much less than the rest probably shows something
    // else, or the item from a very unusual angle. (Two photos have no "rest".)
    if (agreement.size() > 2)
        for (const PhotoAgreement& a : agreement)
            if (a.agreement < mean - kOutlierGap || a.agreement < threshold)
                std::printf(
                    "  Check %s: it agrees with the others only at %.2f. Does it show the "
                    "same %s?\n",
                    a.image.c_str(), a.agreement, label.baseClass.c_str());
    if (missing && !f.cache.wholePhoto)
        std::printf(
            "  Photos without a %s found aren't used. For close-ups of the item, add  "
            "whole_photo: true  to %s\n",
            label.baseClass.c_str(), settingsFile.c_str());
    if (used < kRecommendedPhotos)
        std::printf("  Tip: 5-10 photos from different angles and in different light work best.\n");
}

// Warns about pairs of labels of the same class that the recognizer would
// have trouble telling apart. Only pairs with a label in `changed`, unless `all`.
void warnLookAlikes(const LabelLibrary& library, const std::map<std::string, Change>& changed,
                    bool all, const ClassRegistry& classes, const Embedders& embedders) {
    const float margin = RecognizerParams{}.minMargin;
    const std::vector<CustomLabel>& labels = library.labels();
    auto lookAlike = [](const LabelConfusion& c) {
        return c.ambiguous > 0 && c.ambiguous * 4 >= c.photos;  // a quarter of the photos or more
    };
    for (size_t i = 0; i < labels.size(); ++i)
        for (size_t j = i + 1; j < labels.size(); ++j) {
            const CustomLabel& a = labels[i];
            const CustomLabel& b = labels[j];
            if (a.baseClass != b.baseClass) continue;
            if (!all && !changed.count(a.name) && !changed.count(b.name)) continue;
            const float fallback =
                embedders.defaultThreshold(embedKind(classes.find(a.baseClass), classes));
            const LabelConfusion ab =
                labelConfusion(a, b, b.threshold > 0 ? b.threshold : fallback, margin);
            const LabelConfusion ba =
                labelConfusion(b, a, a.threshold > 0 ? a.threshold : fallback, margin);
            if (!lookAlike(ab) && !lookAlike(ba)) continue;
            std::printf(
                "Warning: '%s' and '%s' (both %s) look alike. Photos of %s score %.2f as %s and "
                "%.2f as %s on average; photos of %s score %.2f as %s and %.2f as %s. Recognition "
                "needs a lead of %.2f, so it may mix them up or show neither. Photos showing how "
                "they differ help; if they are the same item, keep one label.\n",
                a.name.c_str(), b.name.c_str(), a.baseClass.c_str(), a.name.c_str(), ab.ownScore,
                a.name.c_str(), ab.otherScore, b.name.c_str(), b.name.c_str(), ba.ownScore,
                b.name.c_str(), ba.otherScore, a.name.c_str(), margin);
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
               Embedders& embedders, bool all) {
    std::map<std::string, Change> changes;
    std::map<std::string, std::string> problems;
    for (const LabelFolder& f : library.folders()) {
        if (!f.problem.empty()) continue;
        // The class: set in label.yml, or chosen from the photos by an earlier sync.
        ClassId base = -1;
        if (!f.settings.baseClass.empty()) {
            base = classes.find(f.settings.baseClass);
            if (base < 0 || classes.kind(base) == ClassKind::Custom) {
                problems[f.name] = (fs::path(f.path) / kSettingsFile).string() +
                                   ": unknown class '" + f.settings.baseClass +
                                   "' (see --list-classes)";
                continue;
            }
        } else if (f.cache.autoClass) {
            base = classes.find(f.cache.baseClass);
            if (base >= 0 && classes.kind(base) == ClassKind::Custom) base = -1;
        }
        const std::string model = base >= 0 ? embedders.modelId(embedKind(base, classes)) : "";
        const SyncPlan plan = planSync(f.settings, f.cache, f.photos, model);
        if (plan.empty()) continue;
        changes[f.name] = syncFolder(f, base, plan, detector, classes, embedders);
    }
    library.load();

    if (all && library.folders().empty())
        std::printf(
            "No labels in %s. Make a folder there for each item (e.g. %s/my_mug/) and put "
            "5-10 photos of it in it.\n",
            library.dir().c_str(), library.dir().c_str());
    for (const LabelFolder& f : library.folders()) {
        const std::string problem = !f.problem.empty()       ? f.problem
                                    : problems.count(f.name) ? problems[f.name]
                                                             : "";
        if (!problem.empty()) {
            std::printf("Label '%s' can't be used: %s\n", f.name.c_str(), problem.c_str());
            continue;
        }
        const auto change = changes.find(f.name);
        if (change == changes.end() && !all) continue;
        printReport(f, change == changes.end() ? nullptr : &change->second, library, classes,
                    embedders);
    }
    warnLookAlikes(library, changes, all, classes, embedders);
    if (!library.loosePhotos().empty())
        std::printf(
            "%zu photo(s) in %s aren't in a label folder (e.g. %s). Each label is a folder "
            "of photos, e.g. %s/my_mug/.\n",
            library.loosePhotos().size(), library.dir().c_str(),
            library.loosePhotos().front().c_str(), library.dir().c_str());
    return static_cast<int>(changes.size());
}

void enroll(const std::string& name, const std::string& classHint,
            const std::vector<std::string>& images, bool wholePhoto, Detector& detector,
            const ClassRegistry& classes, Embedders& embedders, LabelLibrary& library,
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
    const std::string settingsFile = (fs::path(folder) / kSettingsFile).string();

    LabelSettings settings = f ? f->settings : LabelSettings{};
    bool newSettings = false;
    if (!classHint.empty()) {
        const ClassId base = classes.find(classHint);
        if (base < 0 || classes.kind(base) == ClassKind::Custom)
            throw std::runtime_error("Unknown class '" + classHint + "' (see --list-classes)");
        const std::string current = !settings.baseClass.empty() ? settings.baseClass
                                    : f                         ? f->cache.baseClass
                                                                : "";
        if (!current.empty() && current != classHint)
            throw std::runtime_error("'" + name + "' is a " + current + ", not a " + classHint +
                                     " (to change that, set its class in " + settingsFile + ")");
        newSettings = settings.baseClass != classHint;
        settings.baseClass = classHint;
    }
    if (wholePhoto && !settings.wholePhoto) {
        if (f && !f->cache.photos.empty())
            throw std::runtime_error("'" + name +
                                     "' finds the item in each photo, and --whole-photo would "
                                     "change that for all of its photos. To do that, set "
                                     "whole_photo: true in " +
                                     settingsFile);
        settings.wholePhoto = true;
        newSettings = true;
    }
    // Check every photo before copying any.
    for (const std::string& path : images) {
        if (!fs::is_regular_file(path)) throw std::runtime_error("No such photo: " + path);
        if (!isPhotoName(path))
            throw std::runtime_error(path + " isn't a photo (JPEG, PNG and the like)");
        if (!cv::haveImageReader(path))
            throw std::runtime_error("Can't read " + path + ": " + unreadableNote(path));
    }

    fs::create_directories(folder);
    if (newSettings) writeSettings(settingsFile, settings);
    int copied = 0;
    for (const std::string& path : images) copied += storePhoto(path, folder);
    std::printf("Copied %d photo(s) to %s", copied, folder.c_str());
    if (copied < static_cast<int>(images.size()))
        std::printf(" (%d were there already)", static_cast<int>(images.size()) - copied);
    std::printf("\n");

    library.load();
    syncLabels(library, detector, classes, embedders, false);
    if (library.find(name) < 0)
        throw std::runtime_error("'" + name +
                                 "' has no usable photos yet, so it can't be recognized (its "
                                 "photos are in " +
                                 folder + ")");
}

void listLabels(const LabelLibrary& library) {
    const std::string& dir = library.dir();
    if (library.folders().empty()) {
        std::cout << "No labels in " << dir << ". Make a folder there for each item (e.g. " << dir
                  << "/my_mug/) and put 5-10 photos of it in it, or use --enroll.\n";
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
        const std::string& base =
            !f.settings.baseClass.empty() ? f.settings.baseClass : f.cache.baseClass;
        std::cout << (base.empty() ? "class not chosen yet" : base);
        if (f.settings.baseClass.empty() && !base.empty()) std::cout << " (chosen from the photos)";
        std::cout << ", " << f.photos.size() << " photo(s), " << f.usedPhotos() << " usable";
        if (f.settings.wholePhoto) std::cout << ", whole photos";
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
