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

CustomLabel makeLabel(const std::string& name, std::vector<LabelSample> samples) {
    return {name, "test-model", 0, std::move(samples)};
}

}  // namespace

TEST(Labels, NamesAreSafeForFoldersAndClasses) {
    EXPECT_TRUE(validLabelName("alex_b"));
    EXPECT_TRUE(validLabelName("Vivek-2"));
    EXPECT_FALSE(validLabelName(""));
    EXPECT_FALSE(validLabelName("alex b"));
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
        "x", {sample({1, 0}), sample({1, 0}), sample({0, 1}), sample({1, 1}), sample({-1, 0})});
    // Similarities to (1, 0): 1, 1, 0, 0.707, -1 -> best three: 1, 1, 0.707.
    EXPECT_NEAR(labelScore(label, unit({1, 0})), (2 + std::sqrt(0.5f)) / 3, 1e-5);
}

TEST(Labels, MatchNeedsTheThresholdAndAClearWinner) {
    LabelLibrary library("unused");
    library.labels() = {makeLabel("alex", {sample({1, 0, 0})}),
                        makeLabel("sam", {sample({0.9f, 0.1f, 0})})};
    // Close to both: no clear winner.
    LabelMatch m = library.match(unit({1, 0.05f, 0}), 0.5f, 0.05f);
    EXPECT_EQ(m.label, -1);
    // Clearly alex.
    library.labels()[1] = makeLabel("sam", {sample({0, 1, 0})});
    m = library.match(unit({1, 0.05f, 0}), 0.5f, 0.05f);
    EXPECT_EQ(m.label, 0);
    // Below the threshold.
    m = library.match(unit({0.3f, 0.3f, 1}), 0.5f, 0.05f);
    EXPECT_EQ(m.label, -1);
    // A label's own threshold.
    library.labels()[0].threshold = 0.999f;
    EXPECT_EQ(library.match(unit({1, 0.05f, 0}), 0.5f, 0.05f).label, -1);
    EXPECT_EQ(LabelLibrary("unused").match(unit({1, 0, 0}), 0.5f, 0.05f).label, -1) << "no labels";
}

TEST(Labels, MatchNamesTheClosestLabelEvenWithoutAMatch) {
    LabelLibrary library("unused");
    library.labels() = {makeLabel("alex", {sample({1, 0, 0})}),
                        makeLabel("sam", {sample({0, 1, 0})})};
    LabelMatch m = library.match(unit({0.3f, 0.4f, 1}), 0.5f, 0.05f);
    EXPECT_EQ(m.label, -1) << "below the threshold";
    EXPECT_EQ(m.best, 1);
    EXPECT_NEAR(m.score, 0.4f / std::sqrt(1.25f), 1e-5);
    m = library.match(unit({1, 0.05f, 0}), 0.5f, 0.05f);
    EXPECT_EQ(m.label, 0);
    EXPECT_EQ(m.best, 0);
    EXPECT_EQ(LabelLibrary("unused").match(unit({1, 0, 0}), 0.5f, 0.05f).best, -1);
}

TEST(Labels, APhotosSamplesAreAddedAndRemovedInMemory) {
    LabelLibrary library("unused");
    library.labels() = {makeLabel("alex", {sample({1, 0})})};
    PhotoRecord r;
    r.file.name = "auto_20261104_090000.jpg";
    r.used = true;
    r.roi = {5, 6, 7, 8};
    r.embedding = unit({1, 1});
    r.mirrored = unit({1, -1});
    library.addPhoto(0, r);
    const CustomLabel& label = library.labels()[0];
    ASSERT_EQ(label.samples.size(), 3u);
    EXPECT_EQ(label.samples[1].image, r.file.name);
    EXPECT_EQ(label.samples[1].roi, r.roi);
    EXPECT_FALSE(label.samples[1].flipped);
    EXPECT_EQ(label.samples[1].embedding, r.embedding);
    EXPECT_TRUE(label.samples[2].flipped);
    EXPECT_EQ(label.samples[2].embedding, r.mirrored);

    library.removePhoto(0, r.file.name);
    ASSERT_EQ(label.samples.size(), 1u);
    EXPECT_EQ(label.samples[0].image, "photo.jpg");
}

TEST(Labels, SaveAndLoadRoundTrip) {
    const std::string dir = testing::TempDir() + "/labels_roundtrip";
    std::filesystem::remove_all(dir);
    LabelLibrary library(dir);
    CustomLabel label = makeLabel("alex", {sample({1, 2, 3}), sample({3, 2, 1})});
    label.samples[1].flipped = true;
    label.threshold = 0.7f;
    library.save(label);

    LabelLibrary loaded(dir);
    loaded.load();
    ASSERT_EQ(loaded.labels().size(), 1u);
    const CustomLabel& back = loaded.labels()[0];
    EXPECT_EQ(back.name, "alex");
    EXPECT_EQ(back.model, "test-model");
    EXPECT_FLOAT_EQ(back.threshold, 0.7f);
    ASSERT_EQ(back.samples.size(), 2u);
    EXPECT_EQ(back.samples[0].roi, cv::Rect(1, 2, 3, 4));
    EXPECT_FALSE(back.samples[0].flipped);
    EXPECT_TRUE(back.samples[1].flipped);
    ASSERT_EQ(back.samples[0].embedding.size(), 3u);
    EXPECT_NEAR(similarity(back.samples[0].embedding, label.samples[0].embedding), 1, 1e-5);
    // As the earlier version, which also had object labels, reads it.
    EXPECT_EQ(loaded.folders()[0].cache.baseClass, "face");
    EXPECT_TRUE(loaded.folders()[0].cache.autoClass) << "label.yml sets no class";
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

LabelCache faceCache(std::vector<PhotoRecord> photos) {
    return {"test-model", "face", true, false, std::move(photos)};
}

}  // namespace

TEST(Labels, PlanSyncFindsNewChangedAndRemovedPhotos) {
    const LabelCache cache = faceCache({record("a.jpg"), record("b.jpg"), record("gone.jpg")});
    const SyncPlan plan =
        planSync(cache, {file("a.jpg"), file("b.jpg", 100, 6), file("new.jpg")}, "test-model");
    EXPECT_FALSE(plan.redoAll);
    ASSERT_EQ(plan.keep.size(), 1u);
    EXPECT_EQ(plan.keep[0].file.name, "a.jpg");
    ASSERT_EQ(plan.look.size(), 2u);
    EXPECT_EQ(plan.look[0].name, "b.jpg");
    EXPECT_EQ(plan.look[1].name, "new.jpg");
    EXPECT_EQ(plan.changed, std::vector<std::string>{"b.jpg"});
    EXPECT_EQ(plan.removed, std::vector<std::string>{"gone.jpg"});

    const SyncPlan same =
        planSync(cache, {file("a.jpg"), file("b.jpg"), file("gone.jpg")}, "test-model");
    EXPECT_TRUE(same.empty());
    EXPECT_EQ(same.keep.size(), 3u);
}

TEST(Labels, PlanSyncRedoesEveryPhotoWhenWhatTheyWereMadeWithChanges) {
    const LabelCache cache = faceCache({record("a.jpg"), record("b.jpg")});
    const std::vector<PhotoFile> photos = {file("a.jpg"), file("b.jpg")};
    auto redo = [&](const LabelCache& c, const std::string& model) {
        const SyncPlan plan = planSync(c, photos, model);
        return plan.redoAll && plan.keep.empty() && plan.look.size() == 2 ? plan.reason : "-";
    };
    EXPECT_EQ(redo(cache, "test-model"), "-");
    EXPECT_EQ(redo(cache, "other-model"), "the embedding model changed");
    EXPECT_EQ(redo(cache, ""), "-") << "model not checked";

    // An earlier version also had object labels and whole photos.
    LabelCache object = cache;  // label.yml says class: face now
    object.baseClass = "cell phone";
    EXPECT_EQ(redo(object, "test-model"), "its class changed from cell phone to face");
    LabelCache whole = cache;
    whole.wholePhoto = true;
    EXPECT_EQ(redo(whole, "test-model"),
              "whole photos aren't used any more (the face is found in each photo)");

    LabelCache none = cache;  // that version found nothing to choose a class from
    none.baseClass.clear();
    EXPECT_EQ(redo(none, "test-model"), "-") << "same photos: still nothing to find";
    EXPECT_FALSE(planSync(none, {file("a.jpg"), file("c.jpg")}, "").redoAll) << "just the new one";
    const SyncPlan fresh = planSync({}, photos, "test-model");  // a new label
    EXPECT_EQ(fresh.look.size(), 2u);
    EXPECT_TRUE(fresh.reason.empty());
}

TEST(Labels, SettingsAreReadFromAHandWrittenFile) {
    const std::string path = testing::TempDir() + "/label_settings.yml";
    auto read = [&](const std::string& text) {
        std::ofstream(path) << text;
        return readSettings(path);
    };
    LabelSettings s = read("# alex\nthreshold: 0.7\n");
    EXPECT_FLOAT_EQ(s.threshold, 0.7f);
    EXPECT_TRUE(s.oldKeys.empty());
    EXPECT_FLOAT_EQ(read("# nothing set yet\n\n").threshold, 0);
    EXPECT_FLOAT_EQ(read("").threshold, 0);
    EXPECT_THROW(read("treshold: 0.5\n"), std::runtime_error);
    EXPECT_THROW(read("threshold: 3\n"), std::runtime_error);
    std::filesystem::remove(path);
    EXPECT_FLOAT_EQ(readSettings(path).threshold, 0) << "no file: the defaults";

    writeSettings(path, {0.65f});
    s = readSettings(path);
    EXPECT_FLOAT_EQ(s.threshold, 0.65f);
    EXPECT_TRUE(s.oldKeys.empty()) << "nothing but the threshold is written";
    std::filesystem::remove(path);
}

TEST(Labels, AutoPhotosCanBeTurnedOffInLabelYml) {
    const std::string path = testing::TempDir() + "/label_settings_auto.yml";
    auto read = [&](const std::string& text) {
        std::ofstream(path) << text;
        return readSettings(path);
    };
    EXPECT_TRUE(read("threshold: 0.5\n").autoPhotos) << "on unless turned off";
    EXPECT_FALSE(read("auto_photos: false\n").autoPhotos);
    EXPECT_TRUE(read("auto_photos: true\n").autoPhotos);
    EXPECT_FALSE(read("auto_photos: 0\n").autoPhotos);
    EXPECT_THROW(read("auto_photos: maybe\n"), std::runtime_error);
    EXPECT_THROW(read("auto_photos: 2\n"), std::runtime_error);

    writeSettings(path, {0.5f, false});
    LabelSettings s = readSettings(path);
    EXPECT_FALSE(s.autoPhotos);
    EXPECT_FLOAT_EQ(s.threshold, 0.5f);
    writeSettings(path, {0.5f, true});
    std::ifstream in(path);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    EXPECT_EQ(text.find("auto_photos"), std::string::npos) << "the default isn't written";
    std::filesystem::remove(path);

    // A label's setting reaches it, and save() keeps it.
    const std::string dir = testing::TempDir() + "/labels_auto";
    std::filesystem::remove_all(dir);
    LabelLibrary library(dir);
    CustomLabel label = makeLabel("alex", {sample({1, 0})});
    label.autoPhotos = false;
    library.save(label);
    library.load();
    ASSERT_EQ(library.labels().size(), 1u);
    EXPECT_FALSE(library.labels()[0].autoPhotos);
    EXPECT_FALSE(library.folders()[0].settings.autoPhotos);
    std::filesystem::remove_all(dir);
}

// An earlier version also had object labels, set with class and whole_photo.
// They're read but not used; a class other than face makes the folder unusable.
TEST(Labels, AnEarlierVersionsSettingsAreReadButNotUsed) {
    const std::string path = testing::TempDir() + "/label_settings_old.yml";
    auto read = [&](const std::string& text) {
        std::ofstream(path) << text;
        return readSettings(path);
    };
    LabelSettings s = read("class: face\nwhole_photo: false\nthreshold: 0.5\n");
    EXPECT_EQ(s.oldClass, "face");
    EXPECT_EQ(s.oldKeys, (std::vector<std::string>{"class", "whole_photo"}));
    EXPECT_FLOAT_EQ(s.threshold, 0.5f);
    EXPECT_EQ(read("class: cell phone\n").oldClass, "cell phone");
    s = read("whole_photo: maybe\n");  // not used, so not checked
    EXPECT_TRUE(s.oldClass.empty());
    EXPECT_EQ(s.oldKeys, std::vector<std::string>{"whole_photo"});
    EXPECT_THROW(read("class: [face]\n"), std::runtime_error);
    std::filesystem::remove(path);
}

TEST(Labels, CacheRoundTrip) {
    const std::string path = testing::TempDir() + "/label_cache.yml";
    LabelCache cache = faceCache({record("a.jpg", 123456789012LL, 1792070584123456789LL)});
    cache.photos[0].roi = {1, 2, 3, 4};
    cache.photos[0].conf = 0.8f;
    cache.photos[0].mirrored = unit({0, 1});
    PhotoRecord unused;
    unused.file = file("blurry.jpg");
    unused.note = "no face found";
    cache.photos.push_back(unused);
    writeCache(path, cache);

    LabelCache back;
    ASSERT_TRUE(readCache(path, back));
    EXPECT_EQ(back.model, "test-model");
    EXPECT_EQ(back.baseClass, "face");
    EXPECT_TRUE(back.autoClass);
    ASSERT_EQ(back.photos.size(), 2u);
    EXPECT_EQ(back.photos[0].file.size, 123456789012LL);
    EXPECT_EQ(back.photos[0].file.mtime, 1792070584123456789LL);
    EXPECT_TRUE(back.photos[0].used);
    EXPECT_EQ(back.photos[0].roi, cv::Rect(1, 2, 3, 4));
    EXPECT_EQ(back.photos[0].embedding, cache.photos[0].embedding);
    EXPECT_EQ(back.photos[0].mirrored, cache.photos[0].mirrored);
    EXPECT_FALSE(back.photos[1].used);
    EXPECT_EQ(back.photos[1].note, "no face found");
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
    const std::vector<PhotoAgreement> agreement = photoAgreement(makeLabel("x", {a, aMirror, b}));
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
    const CustomLabel alex = makeLabel("alex", photos({{1, 0, 0}, {0.95f, 0.1f, 0}}));
    const CustomLabel twin = makeLabel("twin", photos({{0.98f, 0.05f, 0}, {1, 0.08f, 0}}));
    const CustomLabel sam = makeLabel("sam", photos({{0, 1, 0}, {0, 0.95f, 0.1f}}));

    const LabelConfusion alike = labelConfusion(alex, twin, 0.5f, 0.05f);
    EXPECT_EQ(alike.photos, 2);
    EXPECT_EQ(alike.ambiguous, 2);
    EXPECT_GT(alike.otherScore, 0.9f);

    const LabelConfusion apart = labelConfusion(alex, sam, 0.5f, 0.05f);
    EXPECT_EQ(apart.ambiguous, 0);
    EXPECT_GT(apart.ownScore, 0.9f);
    EXPECT_LT(apart.otherScore, 0.2f);
}
