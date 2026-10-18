#include "Enrollment.h"

#include <cstdio>
#include <filesystem>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

struct Photo {
    std::string path;
    cv::Mat image;
    std::vector<Detection> detections;
    const Detection* item = nullptr;  // the detection used as the item
};

// How much a detection stands out in a photo: confident and large.
float prominence(const Detection& d, const cv::Mat& image) {
    return (d.weak ? 0.5f : 1.0f) * d.conf * d.box.area() / (image.cols * image.rows);
}

// Copies a photo into the label's folder under a name not used yet.
std::string storePhoto(const std::string& path, const std::string& folder) {
    const fs::path source(path);
    fs::path target = fs::path(folder) / source.filename();
    for (int n = 2; fs::exists(target); ++n)
        target = fs::path(folder) /
                 (source.stem().string() + "_" + std::to_string(n) + source.extension().string());
    fs::copy_file(source, target);
    return target.filename().string();
}

}  // namespace

void enroll(const std::string& name, const std::string& classHint,
            const std::vector<std::string>& images, bool wholePhoto, Detector& detector,
            const ClassRegistry& classes, Embedders& embedders, LabelLibrary& library) {
    if (!validLabelName(name))
        throw std::runtime_error("Label names may only use letters, digits, _ and -: '" + name +
                                 "'");
    const ClassId sameName = classes.find(name);
    if (sameName >= 0 && classes.kind(sameName) != ClassKind::Custom)
        throw std::runtime_error("'" + name + "' is already a class name; pick another label name");
    if (images.empty()) throw std::runtime_error("--enroll needs at least one photo");

    const int existing = library.find(name);
    ClassId base = -1;
    if (!classHint.empty()) {
        base = classes.find(classHint);
        if (base < 0 || classes.kind(base) == ClassKind::Custom)
            throw std::runtime_error("Unknown class '" + classHint + "' (see --list-classes)");
    }
    if (existing >= 0) {
        const ClassId known = classes.find(library.labels()[existing].baseClass);
        if (base >= 0 && base != known)
            throw std::runtime_error("'" + name + "' is a " + classes.name(known) + ", not a " +
                                     classHint);
        base = known;
    }

    std::vector<Photo> photos;
    for (const std::string& path : images) {
        Photo p{path, cv::imread(path)};
        if (p.image.empty()) throw std::runtime_error("Can't read image " + path);
        p.detections = detector.detect(p.image, {});
        photos.push_back(std::move(p));
    }

    // Which kind of thing it is: the given class, or the class that stands out
    // most across the photos.
    if (base < 0) {
        std::map<ClassId, float> weight;
        for (const Photo& p : photos) {
            const Detection* best = nullptr;
            for (const Detection& d : p.detections)
                if (!best || prominence(d, p.image) > prominence(*best, p.image)) best = &d;
            if (best) weight[best->classId] += prominence(*best, p.image);
        }
        if (weight.empty())
            throw std::runtime_error("No object found in the photos; say what it is with --class");
        base = std::max_element(weight.begin(), weight.end(), [](const auto& a, const auto& b) {
                   return a.second < b.second;
               })->first;
        // A person is recognized by their face (the face model is made for telling
        // people apart), so prefer the face when the photos show one.
        if (base == classes.person()) {
            const auto withFace = std::count_if(photos.begin(), photos.end(), [&](const Photo& p) {
                return std::any_of(p.detections.begin(), p.detections.end(),
                                   [&](const Detection& d) { return d.classId == classes.face(); });
            });
            if (withFace * 2 >= static_cast<long>(photos.size())) base = classes.face();
        }
        std::printf("Enrolling '%s' as a %s (use --class to choose otherwise)\n", name.c_str(),
                    classes.name(base).c_str());
    }
    for (Photo& p : photos)
        for (const Detection& d : p.detections)
            if (d.classId == base &&
                (!p.item || prominence(d, p.image) > prominence(*p.item, p.image)))
                p.item = &d;

    const ClassKind kind =
        classes.kind(base) == ClassKind::Face ? ClassKind::Face : ClassKind::Object;
    IEmbedder& embedder = embedders.forKind(kind);
    CustomLabel label = existing >= 0 ? library.labels()[existing]
                                      : CustomLabel{name, classes.name(base), embedder.modelId()};
    if (label.model != embedder.modelId())
        throw std::runtime_error("'" + name + "' was made with " + label.model +
                                 "; remove it (--remove-label) and enroll it again");

    const std::string folder = library.folder(name);
    fs::create_directories(folder);
    const size_t firstNew = label.samples.size();
    for (const Photo& p : photos) {
        cv::Rect2f box(0, 0, static_cast<float>(p.image.cols), static_cast<float>(p.image.rows));
        std::vector<cv::Point2f> keypoints;
        if (wholePhoto) {
            std::printf("  %s: using the whole photo\n", p.path.c_str());
        } else if (p.item) {
            box = p.item->box;
            keypoints = p.item->keypoints;
            std::printf("  %s: %s at (%d, %d) %dx%d, confidence %.2f\n", p.path.c_str(),
                        classes.name(base).c_str(), static_cast<int>(box.x),
                        static_cast<int>(box.y), static_cast<int>(box.width),
                        static_cast<int>(box.height), p.item->conf);
        } else {
            // The whole photo would mostly show other things; leave it out.
            std::printf("  %s: no %s found, skipped (for a close-up, try --whole-photo)\n",
                        p.path.c_str(), classes.name(base).c_str());
            continue;
        }
        const std::string file = storePhoto(p.path, folder);
        label.samples.push_back(
            {file, cv::Rect(box), false, embedder.embed(p.image, box, keypoints)});

        // The live camera view is mirrored; keep a mirrored copy as well.
        cv::Mat mirrored;
        cv::flip(p.image, mirrored, 1);
        mirrorDetection(p.image.cols, box, keypoints, kind == ClassKind::Face);
        label.samples.push_back(
            {file, cv::Rect(box), true, embedder.embed(mirrored, box, keypoints)});
    }
    if (label.samples.size() == firstNew)
        throw std::runtime_error("None of the photos showed a " + classes.name(base) +
                                 "; nothing was saved");
    library.save(label);
    library.load();

    // How well the photos agree: each photo against the other photos (not its
    // own mirrored copy). A photo that agrees much less than the rest probably
    // shows something else, or the item from a very unusual angle.
    std::vector<float> agreement;
    for (size_t i = 0; i < label.samples.size(); i += 2) {
        float sum = 0;
        int n = 0;
        for (size_t j = 0; j < label.samples.size(); ++j) {
            if (j / 2 == i / 2) continue;
            sum += similarity(label.samples[i].embedding, label.samples[j].embedding);
            ++n;
        }
        agreement.push_back(n ? sum / n : 1);
    }
    float mean = 0;
    for (float a : agreement) mean += a;
    mean /= agreement.size();
    const float threshold = label.threshold > 0 ? label.threshold : embedder.defaultThreshold();
    std::printf("'%s' (%s) now has %zu photo(s), stored in %s\n", name.c_str(),
                label.baseClass.c_str(), label.samples.size() / 2, folder.c_str());
    if (agreement.size() > 1) {
        std::printf("Photos agree at %.2f on average (recognition needs %.2f).\n", mean, threshold);
        for (size_t k = 0; k < agreement.size(); ++k)
            if (agreement[k] < mean - 0.15f || agreement[k] < threshold)
                std::printf("  Check %s: it agrees with the others only at %.2f%s\n",
                            label.samples[2 * k].image.c_str(), agreement[k],
                            2 * k >= firstNew ? " (just added)" : "");
    }
    if (label.samples.size() / 2 < 5)
        std::printf("Tip: 5-10 photos from different angles and lighting work best.\n");
}
