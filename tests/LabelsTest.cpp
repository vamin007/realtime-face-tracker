#include <gtest/gtest.h>

#include <filesystem>

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
