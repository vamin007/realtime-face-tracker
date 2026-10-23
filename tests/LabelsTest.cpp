#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "Labels.h"

namespace {

Embedding unit(std::vector<float> v) {
    float n = 0;
    for (float x : v) n += x * x;
    for (float& x : v) x /= std::sqrt(n);
    return v;
}

LabelSample sample(std::vector<float> v) { return {"photo.jpg", {1, 2, 3, 4}, false, unit(v)}; }

CustomLabel makeLabel(const std::string& name, const std::string& base,
                      std::vector<LabelSample> samples) {
    return {name, base, "test-model", 0, std::move(samples)};
}

}  // namespace

TEST(Labels, NamesAreSafeForFoldersAndClasses) {
    EXPECT_TRUE(validLabelName("my_mug"));
    EXPECT_TRUE(validLabelName("Vivek-2"));
    EXPECT_FALSE(validLabelName(""));
    EXPECT_FALSE(validLabelName("my mug"));
    EXPECT_FALSE(validLabelName("../etc"));
    EXPECT_FALSE(validLabelName("a,b"));
}

TEST(Labels, TargetIsNotALabelNameInAnyCase) {
    EXPECT_FALSE(validLabelName("TARGET")) << "the red box's tag";
    EXPECT_FALSE(validLabelName("target"));
    EXPECT_FALSE(validLabelName("Target"));
    EXPECT_TRUE(validLabelName("target2"));
    EXPECT_TRUE(validLabelName("my_target"));
}

TEST(Labels, ScoreIsTheMeanOfTheThreeBestSamples) {
    const CustomLabel label = makeLabel(
        "x", "cup",
        {sample({1, 0}), sample({1, 0}), sample({0, 1}), sample({1, 1}), sample({-1, 0})});
    // Similarities to (1, 0): 1, 1, 0, 0.707, -1 -> best three: 1, 1, 0.707.
    EXPECT_NEAR(labelScore(label, unit({1, 0})), (2 + std::sqrt(0.5f)) / 3, 1e-5);
}

TEST(Labels, MatchNeedsTheThresholdAndAClearWinner) {
    LabelLibrary library("unused");
    library.labels() = {makeLabel("mug_a", "cup", {sample({1, 0, 0})}),
                        makeLabel("mug_b", "cup", {sample({0.9f, 0.1f, 0})}),
                        makeLabel("bottle", "bottle", {sample({1, 0, 0})})};
    // Close to both mugs: no clear winner.
    LabelMatch m = library.match("cup", unit({1, 0.05f, 0}), 0.5f, 0.05f);
    EXPECT_EQ(m.label, -1);
    // Clearly mug_a.
    library.labels()[1] = makeLabel("mug_b", "cup", {sample({0, 1, 0})});
    m = library.match("cup", unit({1, 0.05f, 0}), 0.5f, 0.05f);
    EXPECT_EQ(m.label, 0);
    // Below the threshold.
    m = library.match("cup", unit({0.3f, 0.3f, 1}), 0.5f, 0.05f);
    EXPECT_EQ(m.label, -1);
    // Only labels of the asked class compete.
    m = library.match("bottle", unit({1, 0, 0}), 0.5f, 0.05f);
    EXPECT_EQ(library.labels()[m.label].name, "bottle");
    EXPECT_EQ(library.match("chair", unit({1, 0, 0}), 0.5f, 0.05f).label, -1);
}

TEST(Labels, SaveAndLoadRoundTrip) {
    const std::string dir = testing::TempDir() + "/labels_roundtrip";
    std::filesystem::remove_all(dir);
    LabelLibrary library(dir);
    CustomLabel label = makeLabel("my_mug", "cup", {sample({1, 2, 3}), sample({3, 2, 1})});
    label.samples[1].flipped = true;
    label.threshold = 0.7f;
    library.save(label);

    LabelLibrary loaded(dir);
    loaded.load();
    ASSERT_EQ(loaded.labels().size(), 1u);
    const CustomLabel& back = loaded.labels()[0];
    EXPECT_EQ(back.name, "my_mug");
    EXPECT_EQ(back.baseClass, "cup");
    EXPECT_EQ(back.model, "test-model");
    EXPECT_FLOAT_EQ(back.threshold, 0.7f);
    ASSERT_EQ(back.samples.size(), 2u);
    EXPECT_EQ(back.samples[0].roi, cv::Rect(1, 2, 3, 4));
    EXPECT_FALSE(back.samples[0].flipped);
    EXPECT_TRUE(back.samples[1].flipped);
    ASSERT_EQ(back.samples[0].embedding.size(), 3u);
    EXPECT_NEAR(similarity(back.samples[0].embedding, label.samples[0].embedding), 1, 1e-5);
    std::filesystem::remove_all(dir);
}

namespace {

PhotoFile file(const std::string& name, long long size = 100, long long mtime = 5) {
    return {name, size, mtime};
}

PhotoRecord record(const std::string& name, long long size = 100, long long mtime = 5) {
    PhotoRecord r;
    r.file = file(name, size, mtime);
    r.used = true;
    r.embedding = unit({1, 0});
    return r;
}

LabelCache cupCache(std::vector<PhotoRecord> photos) {
    return {"test-model", "cup", true, false, std::move(photos)};
}

}  // namespace

TEST(Labels, PlanSyncFindsNewChangedAndRemovedPhotos) {
    const LabelCache cache = cupCache({record("a.jpg"), record("b.jpg"), record("gone.jpg")});
    const SyncPlan plan = planSync({}, cache,
                                   {file("a.jpg"), file("b.jpg", 100, 6), file("new.jpg")},
                                   "test-model");
    EXPECT_FALSE(plan.redoAll);
    ASSERT_EQ(plan.keep.size(), 1u);
    EXPECT_EQ(plan.keep[0].file.name, "a.jpg");
    ASSERT_EQ(plan.look.size(), 2u);
    EXPECT_EQ(plan.look[0].name, "b.jpg");
    EXPECT_EQ(plan.look[1].name, "new.jpg");
    EXPECT_EQ(plan.changed, std::vector<std::string>{"b.jpg"});
    EXPECT_EQ(plan.removed, std::vector<std::string>{"gone.jpg"});

    const SyncPlan same = planSync({}, cache, {file("a.jpg"), file("b.jpg"), file("gone.jpg")},
                                   "test-model");
    EXPECT_TRUE(same.empty());
    EXPECT_EQ(same.keep.size(), 3u);
}

TEST(Labels, PlanSyncRedoesEveryPhotoWhenWhatTheyWereMadeWithChanges) {
    const LabelCache cache = cupCache({record("a.jpg"), record("b.jpg")});
    const std::vector<PhotoFile> photos = {file("a.jpg"), file("b.jpg")};
    auto redo = [&](const LabelSettings& s, const LabelCache& c, const std::string& model) {
        const SyncPlan plan = planSync(s, c, photos, model);
        return plan.redoAll && plan.keep.empty() && plan.look.size() == 2;
    };
    EXPECT_FALSE(redo({"cup"}, cache, "test-model")) << "the class chosen before, now set";
    EXPECT_TRUE(redo({"bottle"}, cache, "test-model"));
    EXPECT_TRUE(redo({"", true}, cache, "test-model")) << "whole_photo turned on";
    EXPECT_TRUE(redo({}, cache, "other-model"));
    EXPECT_FALSE(redo({}, cache, "")) << "model not checked";

    LabelCache given = cache;
    given.autoClass = false;
    EXPECT_FALSE(redo({"cup"}, given, "test-model"));
    EXPECT_TRUE(redo({}, given, "test-model")) << "label.yml no longer sets the class";

    LabelCache none = cache;  // no class could be chosen last time
    none.baseClass.clear();
    EXPECT_FALSE(planSync({}, none, photos, "").redoAll) << "same photos: still nothing to find";
    EXPECT_TRUE(planSync({}, none, {file("a.jpg")}, "").redoAll);
    EXPECT_TRUE(planSync({}, {}, photos, "").redoAll) << "a new label";
}

TEST(Labels, SettingsAreReadFromAHandWrittenFile) {
    const std::string path = testing::TempDir() + "/label_settings.yml";
    auto read = [&](const std::string& text) {
        std::ofstream(path) << text;
        return readSettings(path);
    };
    LabelSettings s = read("# my mug\nclass: cup\nwhole_photo: yes\nthreshold: 0.7\n");
    EXPECT_EQ(s.baseClass, "cup");
    EXPECT_TRUE(s.wholePhoto);
    EXPECT_FLOAT_EQ(s.threshold, 0.7f);
    s = read("class: cell phone\nwhole_photo: false\n");
    EXPECT_EQ(s.baseClass, "cell phone");
    EXPECT_FALSE(s.wholePhoto);
    EXPECT_TRUE(read("whole_photo: true\n").wholePhoto);
    EXPECT_TRUE(read("# nothing set yet\n\n").baseClass.empty());
    EXPECT_TRUE(read("").baseClass.empty());
    EXPECT_THROW(read("clas: cup\n"), std::runtime_error);
    EXPECT_THROW(read("whole_photo: maybe\n"), std::runtime_error);
    EXPECT_THROW(read("threshold: 3\n"), std::runtime_error);
    std::filesystem::remove(path);
    EXPECT_TRUE(readSettings(path).baseClass.empty()) << "no file: the defaults";

    writeSettings(path, {"wine glass", true, 0.65f});
    s = readSettings(path);
    EXPECT_EQ(s.baseClass, "wine glass");
    EXPECT_TRUE(s.wholePhoto);
    EXPECT_FLOAT_EQ(s.threshold, 0.65f);
    std::filesystem::remove(path);
}

TEST(Labels, CacheRoundTrip) {
    const std::string path = testing::TempDir() + "/label_cache.yml";
    LabelCache cache = cupCache({record("a.jpg", 123456789012LL, 1792070584123456789LL)});
    cache.photos[0].roi = {1, 2, 3, 4};
    cache.photos[0].conf = 0.8f;
    cache.photos[0].mirrored = unit({0, 1});
    PhotoRecord unused;
    unused.file = file("blurry.jpg");
    unused.note = "no cup found";
    cache.photos.push_back(unused);
    writeCache(path, cache);

    LabelCache back;
    ASSERT_TRUE(readCache(path, back));
    EXPECT_EQ(back.model, "test-model");
    EXPECT_EQ(back.baseClass, "cup");
    EXPECT_TRUE(back.autoClass);
    ASSERT_EQ(back.photos.size(), 2u);
    EXPECT_EQ(back.photos[0].file.size, 123456789012LL);
    EXPECT_EQ(back.photos[0].file.mtime, 1792070584123456789LL);
    EXPECT_TRUE(back.photos[0].used);
    EXPECT_EQ(back.photos[0].roi, cv::Rect(1, 2, 3, 4));
    EXPECT_EQ(back.photos[0].embedding, cache.photos[0].embedding);
    EXPECT_EQ(back.photos[0].mirrored, cache.photos[0].mirrored);
    EXPECT_FALSE(back.photos[1].used);
    EXPECT_EQ(back.photos[1].note, "no cup found");
    std::filesystem::remove(path);
    EXPECT_FALSE(readCache(path, back));
}

TEST(Labels, PhotosAreRecognizedByExtension) {
    EXPECT_TRUE(isPhotoName("a.jpg"));
    EXPECT_TRUE(isPhotoName("IMG_0001.JPEG"));
    EXPECT_TRUE(isPhotoName("b.png"));
    EXPECT_TRUE(isPhotoName("c.heic")) << "so a sync can say it can't read it";
    EXPECT_FALSE(isPhotoName("label.yml"));
    EXPECT_FALSE(isPhotoName("notes.txt"));
    EXPECT_FALSE(isPhotoName("jpg"));
}

TEST(Labels, AgreementComparesEachPhotoWithTheOtherPhotos) {
    LabelSample a = sample({1, 0}), aMirror = sample({0, 1}), b = sample({1, 0});
    a.image = aMirror.image = "a.jpg";
    aMirror.flipped = true;
    b.image = "b.jpg";
    const std::vector<PhotoAgreement> agreement =
        photoAgreement(makeLabel("x", "cup", {a, aMirror, b}));
    ASSERT_EQ(agreement.size(), 2u);
    EXPECT_EQ(agreement[0].image, "a.jpg");
    EXPECT_NEAR(agreement[0].agreement, 1, 1e-5) << "only b, not its own mirrored copy";
    EXPECT_NEAR(agreement[1].agreement, 0.5f, 1e-5) << "a (1) and a's mirrored copy (0)";
}

TEST(Labels, ConfusionFlagsLabelsThatLookAlike) {
    auto photos = [](std::vector<std::vector<float>> vectors) {
        std::vector<LabelSample> samples;
        for (size_t i = 0; i < vectors.size(); ++i) {
            samples.push_back(sample(vectors[i]));
            samples.back().image = "p" + std::to_string(i) + ".jpg";
        }
        return samples;
    };
    const CustomLabel mugA = makeLabel("mug_a", "cup", photos({{1, 0, 0}, {0.95f, 0.1f, 0}}));
    const CustomLabel twin = makeLabel("twin", "cup", photos({{0.98f, 0.05f, 0}, {1, 0.08f, 0}}));
    const CustomLabel mugB = makeLabel("mug_b", "cup", photos({{0, 1, 0}, {0, 0.95f, 0.1f}}));

    const LabelConfusion alike = labelConfusion(mugA, twin, 0.5f, 0.05f);
    EXPECT_EQ(alike.photos, 2);
    EXPECT_EQ(alike.ambiguous, 2);
    EXPECT_GT(alike.otherScore, 0.9f);

    const LabelConfusion apart = labelConfusion(mugA, mugB, 0.5f, 0.05f);
    EXPECT_EQ(apart.ambiguous, 0);
    EXPECT_GT(apart.ownScore, 0.9f);
    EXPECT_LT(apart.otherScore, 0.2f);
}
