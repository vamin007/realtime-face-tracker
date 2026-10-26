// Label folders end to end: syncing, enrolling, renaming, removing and
// converting labels of earlier versions, with stand-in models so the tests run
// in milliseconds.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>

#include "Enrollment.h"
#include "FaceDetector.h"

namespace fs = std::filesystem;

namespace {

// While their `failing` is set, the stand-in models below throw for photos with
// this background, as a real model can (out of memory, an ONNX Runtime error).
const cv::Scalar kTroubleBackground(0, 160, 0);

bool troublePhoto(const cv::Mat& frame) {
    const cv::Vec3b p = frame.at<cv::Vec3b>(0, 0);
    return cv::Scalar(p[0], p[1], p[2]) == kTroubleBackground;
}

// Finds the red rectangle in a test photo and calls it a face.
class RedFaceDetector : public IDetector {
public:
    explicit RedFaceDetector(ClassId face) : face_(face) {}
    const char* name() const override { return "red"; }
    bool canDetect(ClassId id) const override { return id == face_; }
    std::vector<Detection> detect(const cv::Mat& frame) override {
        if (failing && troublePhoto(frame)) throw std::bad_alloc();
        cv::Mat mask;
        cv::inRange(frame, cv::Scalar(0, 0, 200), cv::Scalar(60, 60, 255), mask);
        int left = mask.cols, top = mask.rows, right = -1, bottom = -1;
        for (int y = 0; y < mask.rows; ++y)
            for (int x = 0; x < mask.cols; ++x)
                if (mask.at<uchar>(y, x)) {
                    left = std::min(left, x), right = std::max(right, x);
                    top = std::min(top, y), bottom = std::max(bottom, y);
                }
        if (right < 0) return {};
        Detection d;
        d.box = cv::Rect2f(left, top, right - left + 1, bottom - top + 1);
        d.classId = face_;
        d.conf = 0.9f;
        return {d};
    }
    bool failing = false;

private:
    ClassId face_;
};

// Finds a person filling the whole photo, and counts how often it was asked.
class PersonDetector : public IDetector {
public:
    explicit PersonDetector(ClassId person) : person_(person) {}
    const char* name() const override { return "person"; }
    bool canDetect(ClassId id) const override { return id == person_; }
    std::vector<Detection> detect(const cv::Mat& frame) override {
        ++calls;
        Detection d;
        d.box = cv::Rect2f(0, 0, frame.cols, frame.rows);
        d.classId = person_;
        d.conf = 0.95f;
        return {d};
    }
    int calls = 0;

private:
    ClassId person_;
};

// Embeds a crop as its mean color, and counts how often it was asked.
class ColorEmbedder : public IEmbedder {
public:
    std::string modelId() const override { return "test-color"; }
    float defaultThreshold() const override { return 0.9f; }
    Embedding embed(const cv::Mat& frame, const cv::Rect2f& box,
                    const std::vector<cv::Point2f>&) override {
        ++calls;
        if (failing && troublePhoto(frame)) throw std::runtime_error("the model failed");
        const cv::Rect roi = cv::Rect(box) & cv::Rect(0, 0, frame.cols, frame.rows);
        if (roi.area() <= 0) return {};
        const cv::Scalar m = cv::mean(frame(roi));
        Embedding e{static_cast<float>(m[0] + 1), static_cast<float>(m[1] + 1),
                    static_cast<float>(m[2] + 1)};
        const float n = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
        for (float& x : e) x /= n;
        return e;
    }
    int calls = 0;
    bool failing = false;
};

// label.yml as the earlier version, which also had object labels, wrote it.
std::string oldSettings(const std::string& cls, bool wholePhoto = false) {
    return "%YAML 1.2\n---\n# Settings for this label. Its photos are the image files in this "
           "folder.\nclass: " +
           cls + "\nwhole_photo: " + (wholePhoto ? "true" : "false") + "\nthreshold: 0.\n";
}

class LabelSync : public testing::Test {
protected:
    void SetUp() override {
        root = fs::path(testing::TempDir()) / "label_sync";
        fs::remove_all(root);
        dir = (root / "labels").string();
        fs::create_directories(dir);
        detector.setParallel(false);
        auto faces = std::make_unique<RedFaceDetector>(classes.face());
        faceDetector = faces.get();
        detector.add(std::move(faces));
        // Like the app's other detectors: loaded, but not needed by a sync.
        auto people = std::make_unique<PersonDetector>(classes.person());
        personDetector = people.get();
        detector.add(std::move(people));
    }
    void TearDown() override { fs::remove_all(root); }

    // A gray photo with a red face at x (or none if x < 0).
    static void photo(const fs::path& path, int x, cv::Scalar background = {128, 128, 128}) {
        fs::create_directories(path.parent_path());
        cv::Mat image(120, 160, CV_8UC3, background);
        if (x >= 0) cv::rectangle(image, {x, 30, 40, 50}, {20, 20, 230}, cv::FILLED);
        cv::imwrite(path.string(), image);
    }

    // A BMP whose header claims 100000 x 100000 pixels, more than OpenCV will
    // read: cv::imread throws for it rather than returning an empty image.
    static void hugeHeader(const fs::path& path) {
        std::ofstream out(path, std::ios::binary);
        auto put = [&](uint32_t v, int bytes) {
            for (int i = 0; i < bytes; ++i) out.put(static_cast<char>(v >> (8 * i)));
        };
        out << "BM";
        put(54, 4), put(0, 4), put(54, 4);           // file size, reserved, pixel data offset
        put(40, 4), put(100000, 4), put(100000, 4);  // header size, width, height
        put(1, 2), put(24, 2);                       // planes, bits per pixel
        for (int i = 0; i < 6; ++i) put(0, 4);       // uncompressed; sizes, colors unset
    }

    // .embeddings.yml as the earlier version wrote it for the photos in `folder`:
    // each photo used, its embedding (1, 0, 0) and mirrored (0, 1, 0).
    static void oldCache(const fs::path& folder, const std::string& model, const std::string& cls,
                         bool chosen, bool wholePhoto = false) {
        LabelCache cache{model, cls, chosen, wholePhoto, {}};
        for (const PhotoFile& p : listPhotos(folder.string())) {
            PhotoRecord r;
            r.file = p;
            r.used = true;
            r.note = cls + " found, confidence 0.90";
            r.roi = {10, 30, 40, 50};
            r.conf = 0.9f;
            r.embedding = {1, 0, 0};
            r.mirrored = {0, 1, 0};
            cache.photos.push_back(r);
        }
        writeCache((folder / kCacheFile).string(), cache);
    }

    int sync(LabelLibrary& library, bool all = false) {
        return syncLabels(library, detector, classes, embedder, all);
    }

    // What a sync prints.
    std::string syncReport(LabelLibrary& library, bool all = false) {
        testing::internal::CaptureStdout();
        try {
            sync(library, all);
        } catch (...) {
            testing::internal::GetCapturedStdout();
            throw;
        }
        return testing::internal::GetCapturedStdout();
    }

    fs::path root;
    std::string dir;
    ClassRegistry classes{{"person"}};
    Detector detector;
    RedFaceDetector* faceDetector = nullptr;   // owned by detector
    PersonDetector* personDetector = nullptr;  // owned by detector
    ColorEmbedder embedder;
};

// The message of the error f() throws; empty if it doesn't throw.
template <typename F>
std::string errorOf(F f) {
    try {
        f();
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

// Every file in a folder: its bytes and modification time.
std::map<std::string, std::pair<std::string, fs::file_time_type>> snapshot(const fs::path& folder) {
    std::map<std::string, std::pair<std::string, fs::file_time_type>> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        std::ifstream in(entry.path(), std::ios::binary);
        std::stringstream bytes;
        bytes << in.rdbuf();
        files[entry.path().filename().string()] = {bytes.str(), entry.last_write_time()};
    }
    return files;
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

}  // namespace

TEST_F(LabelSync, NewFolderBecomesALabelAndUnchangedPhotosAreNotLookedAtAgain) {
    photo(fs::path(dir) / "alex" / "a.png", 10);
    photo(fs::path(dir) / "alex" / "b.png", 60);
    photo(fs::path(dir) / "alex" / "c.png", 100);
    photo(fs::path(dir) / "alex" / "no_face.png", -1, {200, 100, 0});
    LabelLibrary library(dir);
    library.load();
    EXPECT_TRUE(library.needsSync());
    EXPECT_EQ(sync(library), 1);

    ASSERT_EQ(library.labels().size(), 1u);
    const CustomLabel& label = library.labels()[0];
    EXPECT_EQ(label.name, "alex");
    EXPECT_EQ(label.model, "test-color");
    EXPECT_EQ(label.samples.size(), 6u);  // 3 usable photos, each also mirrored
    EXPECT_EQ(embedder.calls, 6);
    const LabelFolder& f = library.folders()[0];
    EXPECT_EQ(f.cache.photos.size(), 4u);  // the unusable photo is remembered too
    EXPECT_EQ(f.usedPhotos(), 3);
    EXPECT_EQ(f.cache.photos[0].roi, cv::Rect(10, 30, 40, 50));
    EXPECT_EQ(f.cache.photos[3].note, "no face found");
    // Written as the earlier version, which also reads the file, expects it.
    EXPECT_EQ(f.cache.baseClass, "face");
    EXPECT_TRUE(f.cache.autoClass) << "label.yml doesn't set a class";
    EXPECT_FALSE(f.cache.wholePhoto);

    // A later start: nothing to do.
    LabelLibrary again(dir);
    again.load();
    EXPECT_FALSE(again.needsSync());
    EXPECT_EQ(sync(again), 0);
    EXPECT_EQ(embedder.calls, 6);
    EXPECT_EQ(again.labels().size(), 1u);
}

TEST_F(LabelSync, OnlyNewAndChangedPhotosAreLookedAtAndRemovedOnesDropped) {
    const fs::path alex = fs::path(dir) / "alex";
    photo(alex / "a.png", 10);
    photo(alex / "b.png", 60);
    photo(alex / "c.png", 100);
    LabelLibrary library(dir);
    library.load();
    sync(library);
    ASSERT_EQ(embedder.calls, 6);

    photo(alex / "b.png", 20, {90, 90, 90});  // changed
    fs::remove(alex / "c.png");               // removed
    photo(alex / "d.png", 70);                // new
    library.load();
    const SyncPlan& plan = library.folders()[0].plan;
    EXPECT_FALSE(plan.redoAll);
    EXPECT_EQ(plan.look.size(), 2u);
    EXPECT_EQ(plan.changed, std::vector<std::string>{"b.png"});
    EXPECT_EQ(plan.removed, std::vector<std::string>{"c.png"});
    EXPECT_EQ(sync(library), 1);
    EXPECT_EQ(embedder.calls, 10);  // b and d, each also mirrored

    const LabelFolder& f = library.folders()[0];
    ASSERT_EQ(f.cache.photos.size(), 3u);
    EXPECT_EQ(f.cache.photos[0].file.name, "a.png");
    EXPECT_EQ(f.cache.photos[1].file.name, "b.png");
    EXPECT_EQ(f.cache.photos[1].roi, cv::Rect(20, 30, 40, 50));
    EXPECT_EQ(f.cache.photos[2].file.name, "d.png");
    EXPECT_EQ(library.labels()[0].samples.size(), 6u);
}

TEST_F(LabelSync, ASyncLooksOnlyForFaces) {
    photo(fs::path(dir) / "alex" / "a.png", 10);
    LabelLibrary library(dir);
    library.load();
    sync(library);
    EXPECT_EQ(personDetector->calls, 0) << "only the face detector is needed";
    ASSERT_EQ(library.labels().size(), 1u);
    const PhotoRecord& r = library.folders()[0].cache.photos[0];
    EXPECT_EQ(r.note, "face found, confidence 0.90");
    EXPECT_EQ(r.roi, cv::Rect(10, 30, 40, 50)) << "the face, not the person around it";
}

TEST_F(LabelSync, PhotosWithNoFaceFoundAreNotLookedAtOnEveryStart) {
    const fs::path alex = fs::path(dir) / "alex";
    photo(alex / "blank.png", -1);
    LabelLibrary library(dir);
    library.load();
    const std::string report = syncReport(library);
    EXPECT_TRUE(contains(report, "blank.png: not used, no face found")) << report;
    EXPECT_TRUE(contains(report, "Each photo needs a clear view of the face")) << report;
    EXPECT_TRUE(library.labels().empty());
    EXPECT_FALSE(library.needsSync());

    photo(alex / "a.png", 10);  // a photo that does show a face: only it is looked at
    library.load();
    EXPECT_TRUE(library.needsSync());
    EXPECT_EQ(library.folders()[0].plan.look.size(), 1u);
    sync(library);
    ASSERT_EQ(library.labels().size(), 1u);
    EXPECT_EQ(embedder.calls, 2);
}

TEST_F(LabelSync, PhotosThatCantBeReadAreRememberedAsUnusable) {
    const fs::path alex = fs::path(dir) / "alex";
    photo(alex / "a.png", 10);
    std::ofstream(alex / "notes.jpg") << "not a photo\n";
    hugeHeader(alex / "huge.bmp");
    LabelLibrary library(dir);
    library.load();
    const std::string report = syncReport(library);
    EXPECT_TRUE(contains(report, "huge.bmp: not used, can't read this file as an image")) << report;
    EXPECT_TRUE(contains(report, "notes.jpg: not used, can't read this file as an image"))
        << report;
    ASSERT_EQ(library.labels().size(), 1u);
    const LabelFolder& f = library.folders()[0];
    EXPECT_EQ(f.cache.photos.size(), 3u);
    EXPECT_EQ(f.usedPhotos(), 1);
    EXPECT_FALSE(library.needsSync()) << "the files won't read better next time";
}

TEST_F(LabelSync, APhotoTheModelFailsOnIsReportedAndTriedAgainAtTheNextSync) {
    const fs::path alex = fs::path(dir) / "alex";
    photo(alex / "a.png", 10);
    photo(alex / "b.png", 60, kTroubleBackground);
    photo(alex / "c.png", 100);
    photo(fs::path(dir) / "sam" / "a.png", 30);  // synced after alex
    embedder.failing = true;
    LabelLibrary library(dir);
    library.load();
    const std::string report = syncReport(library);
    EXPECT_TRUE(contains(report, "b.png: not used, looking at it failed (the model failed)"))
        << report;
    EXPECT_TRUE(contains(report, "2 of 3 photo(s) usable")) << report;
    ASSERT_EQ(library.labels().size(), 2u);
    const LabelFolder& f = library.folders()[library.findFolder("alex")];
    EXPECT_EQ(f.usedPhotos(), 2);
    // Not remembered as unusable: the model may well work next time.
    EXPECT_EQ(f.cache.photos.size(), 2u);
    ASSERT_EQ(f.plan.look.size(), 1u);
    EXPECT_EQ(f.plan.look[0].name, "b.png");
    EXPECT_TRUE(library.needsSync());

    embedder.failing = false;
    EXPECT_EQ(sync(library), 1);
    EXPECT_EQ(library.folders()[library.findFolder("alex")].usedPhotos(), 3);
    EXPECT_FALSE(library.needsSync());
}

TEST_F(LabelSync, PhotosTheDetectorFailsOnAreTriedAgainAtTheNextSync) {
    photo(fs::path(dir) / "alex" / "a.png", 10);
    photo(fs::path(dir) / "alex" / "b.png", 60, kTroubleBackground);
    photo(fs::path(dir) / "sam" / "a.png", 10, kTroubleBackground);  // its only photo
    faceDetector->failing = true;  // runs out of memory on those photos
    LabelLibrary library(dir);
    library.load();
    const std::string report = syncReport(library);
    size_t failed = 0;
    for (size_t at = report.find(": not used, looking at it failed ("); at != std::string::npos;
         at = report.find(": not used, looking at it failed (", at + 1))
        ++failed;
    EXPECT_EQ(failed, 2u) << report;
    EXPECT_FALSE(contains(report, "No photos yet")) << report;
    ASSERT_EQ(library.labels().size(), 1u);
    const LabelFolder& alex = library.folders()[library.findFolder("alex")];
    EXPECT_EQ(alex.usedPhotos(), 1);
    EXPECT_EQ(alex.cache.photos.size(), 1u);
    EXPECT_TRUE(library.folders()[library.findFolder("sam")].cache.photos.empty());
    EXPECT_TRUE(library.needsSync());

    faceDetector->failing = false;
    EXPECT_EQ(sync(library), 2);
    EXPECT_EQ(library.labels().size(), 2u);
    EXPECT_EQ(library.folders()[library.findFolder("alex")].usedPhotos(), 2);
    EXPECT_FALSE(library.needsSync());
}

TEST_F(LabelSync, BadFoldersAreReportedNotLoaded) {
    photo(fs::path(dir) / "alex smith" / "a.png", 10);  // a space: not a valid label name
    photo(fs::path(dir) / "sam" / "a.png", 10);
    std::ofstream(fs::path(dir) / "sam" / "label.yml") << "treshold: 0.5\n";
    photo(fs::path(dir) / "loose.png", 10);
    LabelLibrary library(dir);
    library.load();
    ASSERT_EQ(library.folders().size(), 2u);
    EXPECT_TRUE(contains(library.folders()[0].problem, "letters, digits"));
    EXPECT_TRUE(contains(library.folders()[1].problem, "unknown setting 'treshold'"));
    EXPECT_EQ(library.folders()[1].photos.size(), 1u) << "counted for --remove-label";
    EXPECT_EQ(library.loosePhotos(), std::vector<std::string>{"loose.png"});
    sync(library);
    EXPECT_TRUE(library.labels().empty());
}

TEST_F(LabelSync, FoldersThatCantBeLabelsDontCallForASync) {
    photo(fs::path(dir) / "alex" / "a.png", 10);
    LabelLibrary library(dir);
    library.load();
    sync(library);
    photo(fs::path(dir) / "alex smith" / "a.png", 10);  // a bad name
    photo(fs::path(dir) / "sam" / "a.png", 10);
    std::ofstream(fs::path(dir) / "sam" / "label.yml") << "treshold: 0.5\n";  // a bad label.yml
    photo(fs::path(dir) / "my_phone" / "a.png", 10);  // an earlier version's object label
    std::ofstream(fs::path(dir) / "my_phone" / "label.yml") << oldSettings("cell phone");
    library.load();
    EXPECT_FALSE(library.needsSync()) << "no photos changed, and a sync can't fix them";

    photo(fs::path(dir) / "alex" / "b.png", 60);
    library.load();
    EXPECT_TRUE(library.needsSync());
}

// Earlier versions also had object labels. Such a folder is reported and left
// as it is (photos, label.yml and cache), so a version with object labels can
// still use it.
TEST_F(LabelSync, ObjectLabelsOfAnEarlierVersionAreReportedAndLeftAlone) {
    // Its class set in label.yml, or chosen from the photos.
    const fs::path phone = fs::path(dir) / "my_phone", mug = fs::path(dir) / "my_mug";
    photo(phone / "a.png", 10);
    photo(phone / "b.png", 60);
    std::ofstream(phone / "label.yml") << oldSettings("cell phone");
    oldCache(phone, "dinov2-small-patchmean", "cell phone", false);
    photo(mug / "a.png", 10);
    oldCache(mug, "dinov2-small-patchmean", "cup", true);
    photo(phone / "c.png", 100);  // added since: not looked at either
    const auto phoneBefore = snapshot(phone), mugBefore = snapshot(mug);

    LabelLibrary library(dir);
    library.load();
    const LabelFolder& p = library.folders()[library.findFolder("my_phone")];
    const LabelFolder& m = library.folders()[library.findFolder("my_mug")];
    EXPECT_EQ(p.problem,
              "label.yml sets class: cell phone, but only face labels are supported in this "
              "version; its photos are kept");
    EXPECT_EQ(m.problem,
              "it's a cup label made by an earlier version, but only face labels are supported "
              "in this version; its photos are kept");
    EXPECT_EQ(p.photos.size(), 3u) << "counted for --remove-label";
    EXPECT_TRUE(library.labels().empty());
    EXPECT_FALSE(library.needsSync()) << "a sync can't make them face labels";

    const std::string report = syncReport(library, true);
    EXPECT_TRUE(
        contains(report, "Label 'my_phone' can't be used: label.yml sets class: cell phone"))
        << report;
    EXPECT_TRUE(contains(report, "Label 'my_mug' can't be used: it's a cup label")) << report;
    const std::string error = errorOf([&] {
        enroll("my_phone", {(phone / "c.png").string()}, detector, classes, embedder, library,
               true);
    });
    EXPECT_TRUE(contains(error, "Label 'my_phone' can't be used")) << error;

    EXPECT_EQ(embedder.calls, 0);
    EXPECT_TRUE(snapshot(phone) == phoneBefore) << "nothing in the folder changed";
    EXPECT_TRUE(snapshot(mug) == mugBefore);
}

// A class: line was only checked by a sync, so a run that couldn't sync (with
// --no-faces) said "Label photos changed" for a folder no sync could fix.
TEST_F(LabelSync, AClassOtherThanFaceIsFoundWithoutASync) {
    photo(fs::path(dir) / "alex" / "a.png", 10);
    std::ofstream(fs::path(dir) / "alex" / "label.yml") << "class: fcae\n";
    LabelLibrary library(dir);
    library.load();
    EXPECT_TRUE(contains(library.folders()[0].problem, "label.yml sets class: fcae"));
    EXPECT_FALSE(library.needsSync());
}

TEST_F(LabelSync, FaceLabelsOfAnEarlierVersionKeepTheirEmbeddings) {
    const fs::path me = fs::path(dir) / "me";
    photo(me / "a.png", 10);
    photo(me / "b.png", 60);
    std::ofstream(me / "label.yml") << oldSettings("face");
    oldCache(me, embedder.modelId(), "face", false);
    const auto settingsBefore = snapshot(me)["label.yml"];

    LabelLibrary library(dir);
    library.load();
    EXPECT_TRUE(library.folders()[0].problem.empty()) << library.folders()[0].problem;
    ASSERT_EQ(library.labels().size(), 1u);
    ASSERT_EQ(library.labels()[0].samples.size(), 4u);
    EXPECT_EQ(library.labels()[0].samples[0].embedding, (Embedding{1, 0, 0})) << "from the cache";
    EXPECT_FALSE(library.needsSync());
    const std::string report = syncReport(library, true);
    EXPECT_EQ(embedder.calls, 0);
    EXPECT_TRUE(contains(report, "Note: class and whole_photo in")) << report;
    EXPECT_TRUE(contains(report, "are from an earlier version and not used now")) << report;

    // A new photo: only it is looked at. The cache stays as the earlier version
    // reads it, and label.yml isn't touched.
    photo(me / "c.png", 100);
    library.load();
    sync(library);
    EXPECT_EQ(embedder.calls, 2);
    LabelCache cache;
    ASSERT_TRUE(readCache((me / kCacheFile).string(), cache));
    EXPECT_EQ(cache.baseClass, "face");
    EXPECT_FALSE(cache.autoClass) << "its label.yml sets the class";
    EXPECT_EQ(cache.photos.size(), 3u);
    EXPECT_EQ(cache.photos[0].embedding, (Embedding{1, 0, 0}));
    EXPECT_TRUE(snapshot(me)["label.yml"] == settingsBefore);
}

TEST_F(LabelSync, WholePhotoFaceLabelsOfAnEarlierVersionAreLookedAtAgain) {
    const fs::path me = fs::path(dir) / "me";
    photo(me / "a.png", 10);
    std::ofstream(me / "label.yml") << oldSettings("face", true);
    oldCache(me, embedder.modelId(), "face", false, true);
    LabelLibrary library(dir);
    library.load();
    EXPECT_TRUE(library.folders()[0].plan.redoAll);
    const std::string report = syncReport(library);
    EXPECT_TRUE(contains(report,
                         "every photo looked at again, as whole photos aren't used any "
                         "more (the face is found in each photo)"))
        << report;
    const LabelCache& cache = library.folders()[0].cache;
    EXPECT_FALSE(cache.wholePhoto);
    EXPECT_EQ(cache.photos[0].note, "face found, confidence 0.90");
    EXPECT_FALSE(library.needsSync()) << "label.yml's whole_photo isn't used";
}

TEST_F(LabelSync, TargetIsReservedForTheRedTargetBox) {
    const fs::path source = root / "photos";
    photo(source / "a.png", 10);
    const std::vector<std::string> photos = {(source / "a.png").string()};
    LabelLibrary library(dir);
    library.load();
    for (const char* name : {"TARGET", "target", "Target"}) {
        const std::string error =
            errorOf([&] { enroll(name, photos, detector, classes, embedder, library); });
        EXPECT_TRUE(contains(error, "TARGET is reserved")) << name << ": " << error;
    }
    EXPECT_TRUE(fs::is_empty(dir)) << "no folder made";

    enroll("alex", photos, detector, classes, embedder, library);
    const std::string error = errorOf([&] { renameLabel("alex", "Target", library, classes); });
    EXPECT_TRUE(contains(error, "TARGET is reserved")) << error;
    EXPECT_TRUE(fs::exists(fs::path(dir) / "alex"));

    // A folder made by hand is reported, not loaded, and can be renamed.
    photo(fs::path(dir) / "TARGET" / "a.png", 60);
    library.load();
    const int f = library.findFolder("TARGET");
    ASSERT_GE(f, 0);
    EXPECT_TRUE(contains(library.folders()[f].problem, "TARGET is reserved"));
    EXPECT_TRUE(contains(library.folders()[f].problem, "rename the folder"));
    sync(library);
    ASSERT_EQ(library.labels().size(), 1u);
    EXPECT_EQ(library.labels()[0].name, "alex");
    renameLabel("TARGET", "sam", library, classes);
    library.load();
    sync(library);
    EXPECT_GE(library.find("sam"), 0);
}

TEST_F(LabelSync, EnrollCopiesPhotosIntoTheFolderOnce) {
    const fs::path source = root / "photos";
    photo(source / "a.png", 10);
    photo(source / "b.png", 60);
    LabelLibrary library(dir);
    library.load();
    enroll("alex", {(source / "a.png").string(), (source / "b.png").string()}, detector, classes,
           embedder, library);
    ASSERT_EQ(library.labels().size(), 1u);
    EXPECT_TRUE(fs::exists(fs::path(dir) / "alex" / "a.png"));
    EXPECT_TRUE(fs::exists(source / "a.png")) << "the originals stay";
    EXPECT_FALSE(fs::exists(fs::path(dir) / "alex" / "label.yml")) << "nothing to set";

    // The same photos again: nothing new. A different photo with a taken name: renamed.
    photo(root / "other" / "a.png", 100);
    enroll("alex",
           {(source / "a.png").string(), (source / "b.png").string(),
            (root / "other" / "a.png").string()},
           detector, classes, embedder, library, true);
    EXPECT_EQ(library.folders()[0].photos.size(), 3u);
    EXPECT_TRUE(fs::exists(fs::path(dir) / "alex" / "a_2.png"));

    photo(root / "blank.png", -1);
    const std::string noFace = errorOf([&] {
        enroll("sam", {(root / "blank.png").string()}, detector, classes, embedder, library);
    });
    EXPECT_TRUE(contains(noFace, "'sam' has no usable photos yet")) << noFace;
    EXPECT_THROW(
        enroll("bo", {(source / "a.png").string()}, detector, classes, embedder, library, true),
        std::runtime_error);  // --add-photos to a label that doesn't exist
    for (const char* className : {"face", "person"})
        EXPECT_THROW(
            enroll(className, {(source / "a.png").string()}, detector, classes, embedder, library),
            std::runtime_error);  // a class name
    EXPECT_THROW(
        enroll("alex", {(source / "missing.png").string()}, detector, classes, embedder, library),
        std::runtime_error);
}

TEST_F(LabelSync, RenameKeepsTheEmbeddings) {
    photo(fs::path(dir) / "alex" / "a.png", 10);
    LabelLibrary library(dir);
    library.load();
    sync(library);
    const int calls = embedder.calls;
    renameLabel("alex", "alex_b", library, classes);
    library.load();
    EXPECT_FALSE(library.needsSync());
    ASSERT_EQ(library.labels().size(), 1u);
    EXPECT_EQ(library.labels()[0].name, "alex_b");
    EXPECT_EQ(embedder.calls, calls);

    photo(fs::path(dir) / "sam" / "a.png", 10);
    library.load();
    EXPECT_THROW(renameLabel("alex_b", "sam", library, classes), std::runtime_error);
    EXPECT_THROW(renameLabel("alex_b", "face", library, classes), std::runtime_error);
    EXPECT_THROW(renameLabel("alex_b", "alex b", library, classes), std::runtime_error);
    EXPECT_THROW(renameLabel("nothing", "x", library, classes), std::runtime_error);
}

TEST_F(LabelSync, RemoveMovesTheFolderAsideWithItsPhotos) {
    photo(fs::path(dir) / "alex" / "a.png", 10);
    LabelLibrary library(dir);
    library.load();
    sync(library);
    const fs::path moved = removeLabel("alex", library);
    EXPECT_EQ(moved, fs::path(dir) / ".removed" / "alex");
    EXPECT_TRUE(fs::exists(moved / "a.png"));
    EXPECT_FALSE(fs::exists(fs::path(dir) / "alex"));
    library.load();
    EXPECT_TRUE(library.folders().empty()) << ".removed isn't a label";

    photo(fs::path(dir) / "alex" / "a.png", 10);
    library.load();
    EXPECT_EQ(fs::path(removeLabel("alex", library)), fs::path(dir) / ".removed" / "alex_2");
    EXPECT_TRUE(fs::exists(moved / "a.png")) << "the first one is still there";
    EXPECT_THROW(removeLabel("alex", library), std::runtime_error);
}

TEST_F(LabelSync, LabelsFromTheOldFormatAreConvertedAndKeepTheirEmbeddings) {
    // What --enroll used to write: label.yml with the embeddings, next to the photos.
    const Embedding e1{1, 0, 0}, e2{0, 1, 0}, e3{0, 0, 1};
    auto oldLabel = [&](const fs::path& folder, const std::string& baseClass,
                        const std::string& model) {
        photo(folder / "a.png", 10);
        photo(folder / "b.png", -1, {20, 20, 230});
        cv::FileStorage out((folder / "label.yml").string(), cv::FileStorage::WRITE);
        out << "name" << folder.filename().string() << "base_class" << baseClass << "model" << model
            << "threshold" << 0.8f;
        out << "samples" << "[";
        out << "{" << "image" << "a.png" << "roi" << cv::Rect(10, 30, 40, 50) << "flipped" << 0
            << "embedding" << cv::Mat(e1) << "}";
        out << "{" << "image" << "a.png" << "roi" << cv::Rect(110, 30, 40, 50) << "flipped" << 1
            << "embedding" << cv::Mat(e2) << "}";
        out << "{" << "image" << "b.png" << "roi" << cv::Rect(0, 0, 160, 120) << "flipped" << 0
            << "embedding" << cv::Mat(e3) << "}";
        out << "]";
    };
    const fs::path me = fs::path(dir) / "me", phone = fs::path(dir) / "my_phone";
    oldLabel(me, "face", "test-color");
    oldLabel(phone, "cell phone", "dinov2-small-patchmean");
    LabelLibrary library(dir);
    library.load();
    EXPECT_TRUE(fs::exists(me / ".label-v1.yml"));
    ASSERT_EQ(library.labels().size(), 1u);
    const CustomLabel& label = library.labels()[0];
    EXPECT_EQ(label.name, "me");
    EXPECT_FLOAT_EQ(label.threshold, 0.8f);
    ASSERT_EQ(label.samples.size(), 3u);
    EXPECT_EQ(label.samples[0].embedding, e1);
    EXPECT_TRUE(label.samples[1].flipped);
    EXPECT_EQ(label.samples[1].embedding, e2);
    EXPECT_EQ(label.samples[2].embedding, e3);
    const LabelSettings settings = readSettings((me / "label.yml").string());
    EXPECT_FLOAT_EQ(settings.threshold, 0.8f);
    EXPECT_TRUE(settings.oldKeys.empty());
    EXPECT_FALSE(library.folders()[library.findFolder("me")].cache.wholePhoto)
        << "a.png's face was a box in the photo";

    // An object label is converted for the earlier version that can use it: the
    // class in the cache, which that version takes when label.yml sets none.
    EXPECT_TRUE(fs::exists(phone / ".label-v1.yml"));
    const LabelFolder& p = library.folders()[library.findFolder("my_phone")];
    EXPECT_TRUE(contains(p.problem, "it's a cell phone label made by an earlier version"))
        << p.problem;
    EXPECT_EQ(p.cache.baseClass, "cell phone");
    EXPECT_TRUE(p.cache.autoClass);
    EXPECT_EQ(p.cache.model, "dinov2-small-patchmean");
    EXPECT_EQ(p.cache.photos.size(), 2u);

    // The photos haven't changed, so a sync has nothing to do.
    EXPECT_FALSE(library.needsSync());
    EXPECT_EQ(sync(library), 0);
    EXPECT_EQ(embedder.calls, 0);
    LabelLibrary again(dir);
    again.load();  // converted once only
    EXPECT_EQ(again.labels()[0].samples[0].embedding, e1);
}

// A sync only looks for faces: without the face detector, photos would be
// recorded as showing no face, for good. The other detectors don't matter.
TEST(SyncDetectors, ASyncNeedsOnlyTheFaceDetector) {
    EXPECT_TRUE(missingForSync({"face"}).empty());
    EXPECT_TRUE(missingForSync({"hand", "face", "yolo"}).empty());
    EXPECT_EQ(missingForSync({"yolo", "hand"}), std::vector<std::string>{"face"});
    EXPECT_EQ(missingForSync({}), std::vector<std::string>{"face"});
}

// The name a sync asks for is the real face detector's name. Loads the model.
TEST(SyncDetectors, TheRealFaceDetectorIsEnough) {
    const std::string face = PROJECT_DIR "/models/face_detection_yunet_2023mar.onnx";
    if (!fs::exists(face)) GTEST_SKIP() << "missing " << face;
    const ClassRegistry classes({"person"});
    Detector detector;
    detector.add(std::make_unique<FaceDetector>(face, classes.face(), DetectorThresholds{0.6f}));
    EXPECT_TRUE(missingForSync(detector.names()).empty());
}
