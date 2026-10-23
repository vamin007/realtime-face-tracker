// Label folders end to end: syncing, enrolling, renaming, removing and
// converting old labels, with stand-in models so the tests run in milliseconds.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "Enrollment.h"
#include "FaceDetector.h"
#include "HandDetector.h"
#include "YoloDetector.h"

namespace fs = std::filesystem;

namespace {

// While their `failing` is set, the stand-in models below throw for photos with
// this background, as a real model can (out of memory, an ONNX Runtime error).
const cv::Scalar kTroubleBackground(0, 160, 0);

bool troublePhoto(const cv::Mat& frame) {
    const cv::Vec3b p = frame.at<cv::Vec3b>(0, 0);
    return cv::Scalar(p[0], p[1], p[2]) == kTroubleBackground;
}

// Finds the red rectangle in a test photo and calls it a cup.
class RedCupDetector : public IDetector {
public:
    explicit RedCupDetector(ClassId cup) : cup_(cup) {}
    const char* name() const override { return "red"; }
    bool canDetect(ClassId id) const override { return id == cup_; }
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
        d.classId = cup_;
        d.conf = 0.9f;
        return {d};
    }
    bool failing = false;

private:
    ClassId cup_;
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

class TestEmbedders : public Embedders {
public:
    TestEmbedders() : Embedders("", "", Backend::Cpu, "") {}
    IEmbedder& forKind(ClassKind) override { return embedder; }
    std::string modelId(ClassKind) const override { return "test-color"; }
    float defaultThreshold(ClassKind) const override { return 0.9f; }
    ColorEmbedder embedder;
};

class LabelSync : public testing::Test {
protected:
    void SetUp() override {
        root = fs::path(testing::TempDir()) / "label_sync";
        fs::remove_all(root);
        dir = (root / "labels").string();
        fs::create_directories(dir);
        detector.setParallel(false);
        auto cups = std::make_unique<RedCupDetector>(classes.find("cup"));
        cupDetector = cups.get();
        detector.add(std::move(cups));
    }
    void TearDown() override { fs::remove_all(root); }

    // A gray photo with a red cup at x (or none if x < 0).
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

    int sync(LabelLibrary& library) {
        return syncLabels(library, detector, classes, embedders, false);
    }

    // What a sync prints.
    std::string syncReport(LabelLibrary& library) {
        testing::internal::CaptureStdout();
        try {
            sync(library);
        } catch (...) {
            testing::internal::GetCapturedStdout();
            throw;
        }
        return testing::internal::GetCapturedStdout();
    }

    fs::path root;
    std::string dir;
    ClassRegistry classes{{"person", "cup", "bottle"}};
    Detector detector;
    RedCupDetector* cupDetector = nullptr;  // owned by detector
    TestEmbedders embedders;
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

}  // namespace

TEST_F(LabelSync, NewFolderBecomesALabelAndUnchangedPhotosAreNotLookedAtAgain) {
    photo(fs::path(dir) / "mug" / "a.png", 10);
    photo(fs::path(dir) / "mug" / "b.png", 60);
    photo(fs::path(dir) / "mug" / "c.png", 100);
    photo(fs::path(dir) / "mug" / "no_cup.png", -1, {200, 100, 0});
    LabelLibrary library(dir);
    library.load();
    EXPECT_TRUE(library.needsSync());
    EXPECT_EQ(sync(library), 1);

    ASSERT_EQ(library.labels().size(), 1u);
    const CustomLabel& label = library.labels()[0];
    EXPECT_EQ(label.name, "mug");
    EXPECT_EQ(label.baseClass, "cup");      // chosen from the photos
    EXPECT_EQ(label.samples.size(), 6u);    // 3 usable photos, each also mirrored
    EXPECT_EQ(embedders.embedder.calls, 6);
    const LabelFolder& f = library.folders()[0];
    EXPECT_TRUE(f.cache.autoClass);
    EXPECT_EQ(f.cache.photos.size(), 4u);  // the unusable photo is remembered too
    EXPECT_EQ(f.usedPhotos(), 3);
    EXPECT_EQ(f.cache.photos[0].roi, cv::Rect(10, 30, 40, 50));

    // A later start: nothing to do.
    LabelLibrary again(dir);
    again.load();
    EXPECT_FALSE(again.needsSync());
    EXPECT_EQ(sync(again), 0);
    EXPECT_EQ(embedders.embedder.calls, 6);
    EXPECT_EQ(again.labels().size(), 1u);
}

TEST_F(LabelSync, OnlyNewAndChangedPhotosAreLookedAtAndRemovedOnesDropped) {
    const fs::path mug = fs::path(dir) / "mug";
    photo(mug / "a.png", 10);
    photo(mug / "b.png", 60);
    photo(mug / "c.png", 100);
    LabelLibrary library(dir);
    library.load();
    sync(library);
    ASSERT_EQ(embedders.embedder.calls, 6);

    photo(mug / "b.png", 20, {90, 90, 90});  // changed
    fs::remove(mug / "c.png");                // removed
    photo(mug / "d.png", 70);                 // new
    library.load();
    const SyncPlan& plan = library.folders()[0].plan;
    EXPECT_FALSE(plan.redoAll);
    EXPECT_EQ(plan.look.size(), 2u);
    EXPECT_EQ(plan.changed, std::vector<std::string>{"b.png"});
    EXPECT_EQ(plan.removed, std::vector<std::string>{"c.png"});
    EXPECT_EQ(sync(library), 1);
    EXPECT_EQ(embedders.embedder.calls, 10);  // b and d, each also mirrored

    const LabelFolder& f = library.folders()[0];
    ASSERT_EQ(f.cache.photos.size(), 3u);
    EXPECT_EQ(f.cache.photos[0].file.name, "a.png");
    EXPECT_EQ(f.cache.photos[1].file.name, "b.png");
    EXPECT_EQ(f.cache.photos[1].roi, cv::Rect(20, 30, 40, 50));
    EXPECT_EQ(f.cache.photos[2].file.name, "d.png");
    EXPECT_EQ(library.labels()[0].samples.size(), 6u);
}

TEST_F(LabelSync, SettingsInLabelYmlChooseTheClassAndWholePhotos) {
    const fs::path mug = fs::path(dir) / "mug";
    photo(mug / "a.png", 10);
    photo(mug / "close_up.png", -1, {20, 20, 230});  // all red: the cup fills the photo
    std::ofstream(mug / "label.yml") << "class: bottle\n";
    LabelLibrary library(dir);
    library.load();
    sync(library);
    EXPECT_TRUE(library.labels().empty()) << "no bottle in the photos";
    EXPECT_EQ(library.folders()[0].cache.baseClass, "bottle");
    EXPECT_FALSE(library.folders()[0].cache.autoClass);

    std::ofstream(mug / "label.yml") << "# my mug\nclass: cup\nwhole_photo: true\n";
    library.load();
    EXPECT_TRUE(library.folders()[0].plan.redoAll);
    sync(library);
    ASSERT_EQ(library.labels().size(), 1u);
    EXPECT_TRUE(library.labels()[0].wholePhoto);
    EXPECT_EQ(library.folders()[0].usedPhotos(), 2);
    EXPECT_EQ(library.folders()[0].cache.photos[0].roi, cv::Rect(0, 0, 160, 120));
}

TEST_F(LabelSync, PhotosWithNothingFoundAreNotLookedAtOnEveryStart) {
    const fs::path mug = fs::path(dir) / "mug";
    photo(mug / "blank.png", -1);
    LabelLibrary library(dir);
    library.load();
    sync(library);
    EXPECT_TRUE(library.labels().empty());
    EXPECT_FALSE(library.needsSync());

    photo(mug / "a.png", 10);  // a photo that does show it: the class is chosen again
    library.load();
    EXPECT_TRUE(library.needsSync());
    sync(library);
    ASSERT_EQ(library.labels().size(), 1u);
    EXPECT_EQ(library.labels()[0].baseClass, "cup");
}

TEST_F(LabelSync, PhotosThatCantBeReadAreRememberedAsUnusable) {
    const fs::path mug = fs::path(dir) / "mug";
    photo(mug / "a.png", 10);
    std::ofstream(mug / "notes.jpg") << "not a photo\n";
    hugeHeader(mug / "huge.bmp");
    LabelLibrary library(dir);
    library.load();
    const std::string report = syncReport(library);
    EXPECT_NE(report.find("huge.bmp: not used, can't read this file as an image"),
              std::string::npos)
        << report;
    EXPECT_NE(report.find("notes.jpg: not used, can't read this file as an image"),
              std::string::npos)
        << report;
    ASSERT_EQ(library.labels().size(), 1u);
    const LabelFolder& f = library.folders()[0];
    EXPECT_EQ(f.cache.photos.size(), 3u);
    EXPECT_EQ(f.usedPhotos(), 1);
    EXPECT_FALSE(library.needsSync()) << "the files won't read better next time";
}

TEST_F(LabelSync, APhotoTheModelFailsOnIsReportedAndTriedAgainAtTheNextSync) {
    const fs::path mug = fs::path(dir) / "mug";
    photo(mug / "a.png", 10);
    photo(mug / "b.png", 60, kTroubleBackground);
    photo(mug / "c.png", 100);
    photo(fs::path(dir) / "thermos" / "a.png", 30);  // synced after mug
    embedders.embedder.failing = true;
    LabelLibrary library(dir);
    library.load();
    const std::string report = syncReport(library);
    EXPECT_NE(report.find("b.png: not used, looking at it failed (the model failed)"),
              std::string::npos)
        << report;
    EXPECT_NE(report.find("2 of 3 photo(s) usable"), std::string::npos) << report;
    ASSERT_EQ(library.labels().size(), 2u);
    const LabelFolder& f = library.folders()[library.findFolder("mug")];
    EXPECT_EQ(f.usedPhotos(), 2);
    // Not remembered as unusable: the model may well work next time.
    EXPECT_EQ(f.cache.photos.size(), 2u);
    ASSERT_EQ(f.plan.look.size(), 1u);
    EXPECT_EQ(f.plan.look[0].name, "b.png");
    EXPECT_TRUE(library.needsSync());

    embedders.embedder.failing = false;
    EXPECT_EQ(sync(library), 1);
    EXPECT_EQ(library.folders()[library.findFolder("mug")].usedPhotos(), 3);
    EXPECT_FALSE(library.needsSync());
}

TEST_F(LabelSync, PhotosTheDetectorFailsOnAreTriedAgainAtTheNextSync) {
    // No class set: every photo is searched first, to choose it.
    photo(fs::path(dir) / "mug" / "a.png", 10);
    photo(fs::path(dir) / "mug" / "b.png", 60, kTroubleBackground);
    // A class set: each photo is searched for it.
    photo(fs::path(dir) / "thermos" / "a.png", 10);
    photo(fs::path(dir) / "thermos" / "b.png", 60, kTroubleBackground);
    std::ofstream(fs::path(dir) / "thermos" / "label.yml") << "class: cup\n";
    // No other photo to choose the class from.
    photo(fs::path(dir) / "jug" / "a.png", 10, kTroubleBackground);
    cupDetector->failing = true;  // runs out of memory on those photos
    LabelLibrary library(dir);
    library.load();
    const std::string report = syncReport(library);
    size_t failed = 0;
    for (size_t at = report.find(": not used, looking at it failed ("); at != std::string::npos;
         at = report.find(": not used, looking at it failed (", at + 1))
        ++failed;
    EXPECT_EQ(failed, 3u) << report;
    EXPECT_EQ(report.find("No photos yet"), std::string::npos) << report;
    EXPECT_EQ(report.find("No object found"), std::string::npos) << report;
    ASSERT_EQ(library.labels().size(), 2u);
    for (const char* name : {"mug", "thermos"}) {
        const LabelFolder& f = library.folders()[library.findFolder(name)];
        EXPECT_EQ(f.usedPhotos(), 1) << name;
        EXPECT_EQ(f.cache.photos.size(), 1u) << name;
    }
    EXPECT_TRUE(library.folders()[library.findFolder("jug")].cache.photos.empty());
    EXPECT_TRUE(library.needsSync());

    cupDetector->failing = false;
    EXPECT_EQ(sync(library), 3);
    EXPECT_EQ(library.labels().size(), 3u);
    for (const char* name : {"mug", "thermos"})
        EXPECT_EQ(library.folders()[library.findFolder(name)].usedPhotos(), 2) << name;
    EXPECT_FALSE(library.needsSync());
}

TEST_F(LabelSync, BadFoldersAreReportedNotLoaded) {
    photo(fs::path(dir) / "my mug" / "a.png", 10);  // a space: not a valid label name
    photo(fs::path(dir) / "cup2" / "a.png", 10);
    std::ofstream(fs::path(dir) / "cup2" / "label.yml") << "clas: cup\n";
    photo(fs::path(dir) / "loose.png", 10);
    LabelLibrary library(dir);
    library.load();
    ASSERT_EQ(library.folders().size(), 2u);
    EXPECT_NE(library.folders()[0].problem.find("unknown setting 'clas'"), std::string::npos);
    EXPECT_NE(library.folders()[1].problem.find("letters, digits"), std::string::npos);
    EXPECT_EQ(library.loosePhotos(), std::vector<std::string>{"loose.png"});
    sync(library);
    EXPECT_TRUE(library.labels().empty());
}

TEST_F(LabelSync, FoldersThatCantBeLabelsDontCallForASync) {
    photo(fs::path(dir) / "mug" / "a.png", 10);
    LabelLibrary library(dir);
    library.load();
    sync(library);
    photo(fs::path(dir) / "my mug" / "a.png", 10);  // a bad name
    photo(fs::path(dir) / "cup2" / "a.png", 10);
    std::ofstream(fs::path(dir) / "cup2" / "label.yml") << "clas: cup\n";  // a bad label.yml
    library.load();
    EXPECT_FALSE(library.needsSync()) << "no photos changed, and a sync can't fix them";

    photo(fs::path(dir) / "mug" / "b.png", 60);
    library.load();
    EXPECT_TRUE(library.needsSync());
}

TEST_F(LabelSync, TargetIsReservedForTheRedTargetBox) {
    const fs::path source = root / "photos";
    photo(source / "a.png", 10);
    const std::vector<std::string> photos = {(source / "a.png").string()};
    LabelLibrary library(dir);
    library.load();
    for (const char* name : {"TARGET", "target", "Target"}) {
        const std::string error = errorOf(
            [&] { enroll(name, "", photos, false, detector, classes, embedders, library); });
        EXPECT_NE(error.find("TARGET is reserved"), std::string::npos) << name << ": " << error;
    }
    EXPECT_TRUE(fs::is_empty(dir)) << "no folder made";

    enroll("lamp", "", photos, false, detector, classes, embedders, library);
    const std::string error = errorOf([&] { renameLabel("lamp", "Target", library, classes); });
    EXPECT_NE(error.find("TARGET is reserved"), std::string::npos) << error;
    EXPECT_TRUE(fs::exists(fs::path(dir) / "lamp"));

    // A folder made by hand is reported, not loaded, and can be renamed.
    photo(fs::path(dir) / "TARGET" / "a.png", 60);
    library.load();
    const int f = library.findFolder("TARGET");
    ASSERT_GE(f, 0);
    EXPECT_NE(library.folders()[f].problem.find("TARGET is reserved"), std::string::npos);
    EXPECT_NE(library.folders()[f].problem.find("rename the folder"), std::string::npos);
    sync(library);
    ASSERT_EQ(library.labels().size(), 1u);
    EXPECT_EQ(library.labels()[0].name, "lamp");
    renameLabel("TARGET", "torch", library, classes);
    library.load();
    sync(library);
    EXPECT_GE(library.find("torch"), 0);
}

TEST_F(LabelSync, EnrollCopiesPhotosIntoTheFolderOnce) {
    const fs::path source = root / "photos";
    photo(source / "a.png", 10);
    photo(source / "b.png", 60);
    LabelLibrary library(dir);
    library.load();
    enroll("mug", "", {(source / "a.png").string(), (source / "b.png").string()}, false, detector,
           classes, embedders, library);
    ASSERT_EQ(library.labels().size(), 1u);
    EXPECT_TRUE(fs::exists(fs::path(dir) / "mug" / "a.png"));
    EXPECT_TRUE(fs::exists(source / "a.png")) << "the originals stay";
    EXPECT_FALSE(fs::exists(fs::path(dir) / "mug" / "label.yml")) << "nothing to set";

    // The same photos again: nothing new. A different photo with a taken name: renamed.
    photo(root / "other" / "a.png", 100);
    enroll("mug", "cup",
           {(source / "a.png").string(), (source / "b.png").string(),
            (root / "other" / "a.png").string()},
           false, detector, classes, embedders, library);
    EXPECT_EQ(library.folders()[0].photos.size(), 3u);
    EXPECT_TRUE(fs::exists(fs::path(dir) / "mug" / "a_2.png"));
    EXPECT_EQ(readSettings((fs::path(dir) / "mug" / "label.yml").string()).baseClass, "cup");

    EXPECT_THROW(enroll("mug", "bottle", {(source / "a.png").string()}, false, detector, classes,
                        embedders, library),
                 std::runtime_error);  // it's a cup
    EXPECT_THROW(enroll("mug", "", {(source / "a.png").string()}, true, detector, classes,
                        embedders, library),
                 std::runtime_error);  // whole photos would change its other photos
    EXPECT_THROW(enroll("bowl", "", {(source / "a.png").string()}, false, detector, classes,
                        embedders, library, true),
                 std::runtime_error);  // --add-photos to a label that doesn't exist
    EXPECT_THROW(enroll("cup", "", {(source / "a.png").string()}, false, detector, classes,
                        embedders, library),
                 std::runtime_error);  // a class name
    EXPECT_THROW(enroll("mug", "", {(source / "missing.png").string()}, false, detector, classes,
                        embedders, library),
                 std::runtime_error);
}

TEST_F(LabelSync, RenameKeepsTheEmbeddings) {
    photo(fs::path(dir) / "mug" / "a.png", 10);
    LabelLibrary library(dir);
    library.load();
    sync(library);
    const int calls = embedders.embedder.calls;
    renameLabel("mug", "my_mug", library, classes);
    library.load();
    EXPECT_FALSE(library.needsSync());
    ASSERT_EQ(library.labels().size(), 1u);
    EXPECT_EQ(library.labels()[0].name, "my_mug");
    EXPECT_EQ(embedders.embedder.calls, calls);

    photo(fs::path(dir) / "bowl" / "a.png", 10);
    library.load();
    EXPECT_THROW(renameLabel("my_mug", "bowl", library, classes), std::runtime_error);
    EXPECT_THROW(renameLabel("my_mug", "cup", library, classes), std::runtime_error);
    EXPECT_THROW(renameLabel("my_mug", "my mug", library, classes), std::runtime_error);
    EXPECT_THROW(renameLabel("nothing", "x", library, classes), std::runtime_error);
}

TEST_F(LabelSync, RemoveMovesTheFolderAsideWithItsPhotos) {
    photo(fs::path(dir) / "mug" / "a.png", 10);
    LabelLibrary library(dir);
    library.load();
    sync(library);
    const fs::path moved = removeLabel("mug", library);
    EXPECT_EQ(moved, fs::path(dir) / ".removed" / "mug");
    EXPECT_TRUE(fs::exists(moved / "a.png"));
    EXPECT_FALSE(fs::exists(fs::path(dir) / "mug"));
    library.load();
    EXPECT_TRUE(library.folders().empty()) << ".removed isn't a label";

    photo(fs::path(dir) / "mug" / "a.png", 10);
    library.load();
    EXPECT_EQ(fs::path(removeLabel("mug", library)), fs::path(dir) / ".removed" / "mug_2");
    EXPECT_TRUE(fs::exists(moved / "a.png")) << "the first one is still there";
    EXPECT_THROW(removeLabel("mug", library), std::runtime_error);
}

TEST_F(LabelSync, LabelsFromTheOldFormatAreConvertedAndKeepTheirEmbeddings) {
    // What --enroll used to write: label.yml with the embeddings, next to the photos.
    const fs::path mug = fs::path(dir) / "my_mug";
    photo(mug / "a.png", 10);
    photo(mug / "b.png", -1, {20, 20, 230});
    const Embedding e1{1, 0, 0}, e2{0, 1, 0}, e3{0, 0, 1};
    {
        cv::FileStorage out((mug / "label.yml").string(), cv::FileStorage::WRITE);
        out << "name" << "my_mug" << "base_class" << "cup" << "model" << "test-color"
            << "threshold" << 0.8f;
        out << "samples" << "[";
        out << "{" << "image" << "a.png" << "roi" << cv::Rect(10, 30, 40, 50) << "flipped" << 0
            << "embedding" << cv::Mat(e1) << "}";
        out << "{" << "image" << "a.png" << "roi" << cv::Rect(110, 30, 40, 50) << "flipped" << 1
            << "embedding" << cv::Mat(e2) << "}";
        out << "{" << "image" << "b.png" << "roi" << cv::Rect(0, 0, 160, 120) << "flipped" << 0
            << "embedding" << cv::Mat(e3) << "}";
        out << "]";
    }
    LabelLibrary library(dir);
    library.load();
    EXPECT_TRUE(fs::exists(mug / ".label-v1.yml"));
    ASSERT_EQ(library.labels().size(), 1u);
    const CustomLabel& label = library.labels()[0];
    EXPECT_EQ(label.baseClass, "cup");
    EXPECT_FLOAT_EQ(label.threshold, 0.8f);
    EXPECT_FALSE(label.wholePhoto) << "a.png's item was a box in the photo";
    ASSERT_EQ(label.samples.size(), 3u);
    EXPECT_EQ(label.samples[0].embedding, e1);
    EXPECT_TRUE(label.samples[1].flipped);
    EXPECT_EQ(label.samples[1].embedding, e2);
    EXPECT_EQ(label.samples[2].embedding, e3);
    const LabelSettings settings = readSettings((mug / "label.yml").string());
    EXPECT_EQ(settings.baseClass, "cup");
    EXPECT_FLOAT_EQ(settings.threshold, 0.8f);

    // The photos haven't changed, so a sync has nothing to do.
    EXPECT_FALSE(library.needsSync());
    EXPECT_EQ(sync(library), 0);
    EXPECT_EQ(embedders.embedder.calls, 0);
    LabelLibrary again(dir);
    again.load();  // converted once only
    EXPECT_EQ(again.labels()[0].samples[0].embedding, e1);
}

// A sync looks for every class in the photos: with a detector missing, photos of
// its classes would be recorded as showing nothing, for good.
TEST(SyncDetectors, ASyncNeedsYoloFacesAndHands) {
    EXPECT_TRUE(missingForSync({"yolo", "face", "hand"}).empty());
    EXPECT_TRUE(missingForSync({"hand", "face", "yolo"}).empty());
    EXPECT_EQ(missingForSync({"face", "hand"}), std::vector<std::string>{"yolo"});
    EXPECT_EQ(missingForSync({"yolo"}), (std::vector<std::string>{"face", "hand"}));
    EXPECT_EQ(missingForSync({}), (std::vector<std::string>{"yolo", "face", "hand"}));
}

// The names a sync asks for are the real detectors' names. Loads the models.
TEST(SyncDetectors, TheRealDetectorsAreEnough) {
    const std::string models = PROJECT_DIR "/models/";
    std::string yolo = models + "yolov8n_384x640.onnx";
    if (!fs::exists(yolo)) yolo = models + "yolov8n.onnx";
    const std::string face = models + "face_detection_yunet_2023mar.onnx";
    const std::string hand = models + "palm_detection_mediapipe_2023feb.onnx";
    for (const std::string& path : {yolo, face, hand})
        if (!fs::exists(path)) GTEST_SKIP() << "missing " << path;
    const ClassRegistry classes = ClassRegistry::fromFile(models + "coco.names");
    const std::string cache = PROJECT_DIR "/.cache/coreml";
    Detector detector;
    detector.add(std::make_unique<YoloDetector>(yolo, classes.cocoCount(), DetectorThresholds{0.4f},
                                                Backend::Cpu, cache));
    detector.add(std::make_unique<FaceDetector>(face, classes.face(), DetectorThresholds{0.6f}));
    detector.add(std::make_unique<HandDetector>(hand, classes.hand(), DetectorThresholds{0.6f},
                                                Backend::Cpu, cache));
    EXPECT_TRUE(missingForSync(detector.names()).empty());
}
