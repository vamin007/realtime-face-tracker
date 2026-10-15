#include <gtest/gtest.h>

#include <set>

#include "Overlay.h"  // isVisible
#include "Tracker.h"

namespace {

Detection det(float x, float y, int classId = 0, float size = 50) {
    return {cv::Rect2f(x, y, size, size), classId, 0.9f};
}

// IDs of the tracks that would be drawn this frame.
std::set<int> visibleIds(const Tracker& tracker) {
    std::set<int> ids;
    for (const Track& t : tracker.tracks())
        if (isVisible(t)) ids.insert(t.id);
    return ids;
}

}  // namespace

TEST(Tracker, TrackShownOnlyAfterMinHits) {
    Tracker tracker;  // minHits = 3
    tracker.update({det(100, 100)});
    EXPECT_TRUE(visibleIds(tracker).empty());
    tracker.update({det(100, 100)});
    EXPECT_TRUE(visibleIds(tracker).empty());
    tracker.update({det(100, 100)});
    EXPECT_EQ(visibleIds(tracker).size(), 1u);
}

TEST(Tracker, SlowMovingObjectKeepsOneId) {
    Tracker tracker;
    std::set<int> seen;
    for (int f = 0; f < 60; ++f) {
        tracker.update({det(100 + 5.0f * f, 200)});
        for (int id : visibleIds(tracker)) seen.insert(id);
    }
    EXPECT_EQ(seen.size(), 1u);
}

TEST(Tracker, ShortGapKeepsId) {
    Tracker tracker;
    std::set<int> seen;
    for (int f = 0; f < 10; ++f) tracker.update({det(300, 300)});
    for (int id : visibleIds(tracker)) seen.insert(id);
    for (int f = 0; f < 10; ++f) tracker.update({});
    for (int f = 0; f < 10; ++f) tracker.update({det(300, 300)});
    for (int id : visibleIds(tracker)) seen.insert(id);
    EXPECT_EQ(seen.size(), 1u);
}

TEST(Tracker, UnconfirmedTrackDroppedAfterOneMiss) {
    Tracker tracker;
    tracker.update({det(100, 100)});
    tracker.update({});
    EXPECT_TRUE(tracker.tracks().empty());
}

TEST(Tracker, ConfirmedTrackDroppedAfterMaxMisses) {
    TrackerParams params;
    Tracker tracker(params);
    for (int f = 0; f < 5; ++f) tracker.update({det(100, 100)});
    for (int f = 0; f < params.maxMisses; ++f) tracker.update({});
    EXPECT_EQ(tracker.tracks().size(), 1u);  // hidden, but still remembered
    EXPECT_TRUE(visibleIds(tracker).empty());
    tracker.update({});
    EXPECT_TRUE(tracker.tracks().empty());
}

TEST(Tracker, DifferentClassesNeverMatch) {
    Tracker tracker;
    for (int f = 0; f < 5; ++f) tracker.update({det(100, 100, 0), det(100, 100, 1)});
    EXPECT_EQ(visibleIds(tracker).size(), 2u);
}

TEST(Tracker, TwoSeparateObjectsKeepTheirIds) {
    Tracker tracker;
    std::set<int> seen;
    for (int f = 0; f < 40; ++f) {
        tracker.update({det(100 + 4.0f * f, 100), det(100 + 4.0f * f, 400)});
        for (int id : visibleIds(tracker)) seen.insert(id);
    }
    EXPECT_EQ(seen.size(), 2u);
}

// Known weaknesses of the current tracker, fixed in M3 (see PLAN.md).
// Run them with --gtest_also_run_disabled_tests; M3 enables them.

// A small box moving 30 px per frame barely overlaps its last position,
// so IoU matching fails and a new ID starts every frame.
TEST(Tracker, DISABLED_FastSmallObjectKeepsOneId) {
    Tracker tracker;
    std::set<int> seen;
    for (int f = 0; f < 30; ++f) {
        tracker.update({det(50 + 30.0f * f, 200, 0, 40)});
        for (int id : visibleIds(tracker)) seen.insert(id);
    }
    EXPECT_EQ(seen.size(), 1u);
}

// YOLO often flips between similar labels (tv/laptop); the track should survive it.
TEST(Tracker, DISABLED_ClassFlickerKeepsOneId) {
    constexpr int kTv = 62, kLaptop = 63;
    Tracker tracker;
    std::set<int> seen;
    for (int f = 0; f < 40; ++f) {
        tracker.update({det(200, 200, (f / 4) % 2 ? kLaptop : kTv, 120)});
        for (int id : visibleIds(tracker)) seen.insert(id);
    }
    EXPECT_EQ(seen.size(), 1u);
}
