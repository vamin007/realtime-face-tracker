// Tests for Tracker on synthetic detections fed at 30 FPS (no neural network involved):
// identity continuity (slow, fast and crossing objects, gaps), the track lifecycle (tentative,
// confirmed, coasting, dropped), the weak-detection rules, per-class numbering, objects leaving
// through the frame edge, and the shown box keeping up with fast motion.
#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <set>

#include "Overlay.h"  // isVisible
#include "Tracker.h"

namespace {

constexpr double kFrame = 1.0 / 30;  // seconds between frames
constexpr ClassId kPerson = 0;

// A confident, square detection with its top-left corner at (x, y).
Detection det(float x, float y, ClassId cls = kPerson, float size = 50, float conf = 0.9f) {
    Detection d;
    d.box = {x, y, size, size};
    d.classId = cls;
    d.conf = conf;
    return d;
}

// A weak (low-confidence) detection.
Detection weakDet(float x, float y, ClassId cls = kPerson, float size = 50) {
    Detection d = det(x, y, cls, size, 0.2f);
    d.weak = true;
    return d;
}

class TrackerTest : public ::testing::Test {
protected:
    // Feeds one frame (at 30 FPS) and records which tracks were visible.
    void step(Tracker& tracker, const std::vector<Detection>& dets) {
        tracker.update(dets, frame_ * kFrame);
        ++frame_;
        for (const Track& t : tracker.tracks()) {
            if (!isVisible(t)) continue;
            seen_.insert(t.id);
            if (!firstX_.count(t.id)) firstX_[t.id] = t.box.x;
            lastX_[t.id] = t.box.x;
        }
    }
    std::set<int> visibleIds(const Tracker& tracker) const {
        std::set<int> ids;
        for (const Track& t : tracker.tracks())
            if (isVisible(t)) ids.insert(t.id);
        return ids;
    }

    ClassRegistry classes;  // person, face
    long long frame_ = 0;
    std::set<int> seen_;                   // every id that was ever visible
    std::map<int, float> firstX_, lastX_;  // where each id was first and last seen
};

}  // namespace

// A new track stays hidden until it has been detected minHits (3) times.
TEST_F(TrackerTest, TrackShownOnlyAfterMinHits) {
    Tracker tracker(classes);  // minHits = 3
    step(tracker, {det(100, 100)});
    EXPECT_TRUE(visibleIds(tracker).empty());
    step(tracker, {det(100, 100)});
    EXPECT_TRUE(visibleIds(tracker).empty());
    step(tracker, {det(100, 100)});
    EXPECT_EQ(visibleIds(tracker).size(), 1u);
}

// Boxes overlap from frame to frame, so plain overlap matching is enough.
TEST_F(TrackerTest, SlowMovingObjectKeepsOneId) {
    Tracker tracker(classes);
    for (int f = 0; f < 60; ++f) step(tracker, {det(100 + 5.0f * f, 200)});
    EXPECT_EQ(seen_.size(), 1u);
}

// A small box moving 30 px per frame barely overlaps its last position, so matching leans on
// the center distance and on the Kalman filter's prediction of where the box goes next.
TEST_F(TrackerTest, FastSmallObjectKeepsOneId) {
    Tracker tracker(classes);
    for (int f = 0; f < 30; ++f) step(tracker, {det(50 + 30.0f * f, 200, kPerson, 40)});
    EXPECT_EQ(seen_.size(), 1u);
}

// A third of a second without detections is well inside the lost timeout (2 s).
TEST_F(TrackerTest, ShortGapKeepsId) {
    Tracker tracker(classes);
    for (int f = 0; f < 10; ++f) step(tracker, {det(300, 300)});
    for (int f = 0; f < 10; ++f) step(tracker, {});
    for (int f = 0; f < 10; ++f) step(tracker, {det(300, 300)});
    EXPECT_EQ(seen_.size(), 1u);
}

// The distance a lost track may be matched at grows with the time unseen, so an object that
// kept moving is found again further along.
TEST_F(TrackerTest, MovingObjectReappearsAfterGap) {
    Tracker tracker(classes);
    int f = 0;
    for (; f < 20; ++f) step(tracker, {det(100 + 6.0f * f, 200)});
    for (; f < 30; ++f) step(tracker, {});  // hidden, still moving
    for (; f < 45; ++f) step(tracker, {det(100 + 6.0f * f, 200)});
    EXPECT_EQ(seen_.size(), 1u);
}

// The old tracker kept moving a lost box at full speed, so an object that
// stopped while hidden was far from its track when it showed up again.
TEST_F(TrackerTest, ObjectThatStopsWhileHiddenKeepsId) {
    Tracker tracker(classes);
    int f = 0;
    for (; f < 20; ++f) step(tracker, {det(100 + 10.0f * f, 200)});
    const float stoppedAt = 100 + 10.0f * (f - 1);
    for (; f < 35; ++f) step(tracker, {});
    for (; f < 45; ++f) step(tracker, {det(stoppedAt, 200)});
    EXPECT_EQ(seen_.size(), 1u);
}

// A track detected only once never became real: it goes after tentativeTimeout (0.1 s).
TEST_F(TrackerTest, UnconfirmedTrackDroppedAfterGap) {
    Tracker tracker(classes);
    step(tracker, {det(100, 100)});
    for (int f = 0; f < 5; ++f) step(tracker, {});
    EXPECT_TRUE(tracker.tracks().empty());
}

TEST_F(TrackerTest, ConfirmedTrackDroppedAfterLostTimeout) {
    Tracker tracker(classes);
    for (int f = 0; f < 5; ++f) step(tracker, {det(100, 100)});
    for (int f = 0; f < 55; ++f) step(tracker, {});  // ~1.8 s
    EXPECT_EQ(tracker.tracks().size(), 1u);          // hidden, but still remembered
    EXPECT_TRUE(visibleIds(tracker).empty());
    for (int f = 0; f < 10; ++f) step(tracker, {});  // past the 2 s timeout
    EXPECT_TRUE(tracker.tracks().empty());
}

// Leaving through the frame edge: the cut-off box shrinks, and the longer exitTimeout (3 s)
// keeps the track for the object's return.
TEST_F(TrackerTest, ObjectThatLeavesAndComesBackKeepsId) {
    Tracker tracker(classes);
    tracker.setFrameSize({1280, 720});
    int f = 0;
    for (; f < 15; ++f) step(tracker, {det(600, 400 + 10.0f * f, classes.face(), 200)});
    // Sliding off the bottom edge: the box gets cut off and shrinks.
    for (float y = 560; y < 720; y += 40, ++f) {
        Detection d = det(600, y, classes.face(), 200);
        d.box &= cv::Rect2f(0, 0, 1280, 720);
        step(tracker, {d});
    }
    for (int gap = 0; gap < 75; ++gap, ++f) step(tracker, {});  // 2.5 s out of view
    for (int k = 0; k < 10; ++k, ++f) step(tracker, {det(650, 380, classes.face(), 200)});
    EXPECT_EQ(seen_.size(), 1u) << "came back where it left, within the exit timeout";
}

// A huge weak box must not match the track and stretch it (the size-ratio and overlap limits
// refuse the pair).
TEST_F(TrackerTest, JunkBoxesDontInflateATrack) {
    Tracker tracker(classes);
    tracker.setFrameSize({1280, 720});
    for (int f = 0; f < 10; ++f) step(tracker, {det(500, 300, classes.face(), 200)});
    // A weak detection covering most of the frame (the palm model gave these).
    Detection junk = weakDet(0, 0, classes.face(), 0);
    junk.box = {0, 0, 912, 720};
    for (int f = 0; f < 10; ++f) step(tracker, {junk});
    for (const Track& t : tracker.tracks()) EXPECT_LT(t.box.width, 400) << "track " << t.id;
}

// A track stays shown for coastTime (0.2 s) after its last detection, then is hidden.
TEST_F(TrackerTest, BoxStaysVisibleThroughBriefDropOut) {
    Tracker tracker(classes);
    for (int f = 0; f < 5; ++f) step(tracker, {det(100, 100)});
    step(tracker, {});
    step(tracker, {});
    EXPECT_EQ(visibleIds(tracker).size(), 1u) << "a 2-frame drop-out shouldn't flicker";
    for (int f = 0; f < 8; ++f) step(tracker, {});
    EXPECT_TRUE(visibleIds(tracker).empty()) << "hidden after the coast time";
}

// The same box as a person and as a face makes two separate tracks (class gating).
TEST_F(TrackerTest, PeopleAndFacesNeverShareATrack) {
    Tracker tracker(classes);
    for (int f = 0; f < 5; ++f)
        step(tracker, {det(100, 100, kPerson), det(100, 100, classes.face())});
    EXPECT_EQ(visibleIds(tracker).size(), 2u);
}

TEST_F(TrackerTest, WeakDetectionsNeverStartATrack) {
    Tracker tracker(classes);
    for (int f = 0; f < 10; ++f) step(tracker, {weakDet(100, 100)});
    EXPECT_TRUE(tracker.tracks().empty());
}

TEST_F(TrackerTest, WeakDetectionsKeepATrackGoing) {
    Tracker tracker(classes);
    int f = 0;
    for (; f < 10; ++f) step(tracker, {det(100 + 4.0f * f, 100)});
    for (; f < 35; ++f) {  // under a second of weak detections only
        step(tracker, {weakDet(100 + 4.0f * f, 100)});
        EXPECT_EQ(visibleIds(tracker).size(), 1u) << "frame " << f;
    }
    for (; f < 45; ++f) step(tracker, {det(100 + 4.0f * f, 100)});
    EXPECT_EQ(seen_.size(), 1u);
}

// A false person detection that once got confirmed mustn't live on weak detections.
TEST_F(TrackerTest, WeakDetectionsAloneDontKeepATrackForever) {
    Tracker tracker(classes);
    for (int f = 0; f < 5; ++f) step(tracker, {det(300, 300)});
    for (int f = 0; f < 40; ++f) step(tracker, {weakDet(300, 300)});  // 1.3 s
    EXPECT_TRUE(visibleIds(tracker).empty()) << "hidden after a second of weak detections";
    for (int f = 0; f < 120; ++f) step(tracker, {weakDet(300, 300)});  // 5.3 s in total
    EXPECT_TRUE(tracker.tracks().empty()) << "dropped after the weak timeout";
}

// The face model's weak detections are more reliable than YOLO's: a face seen
// only weakly stays shown for 2 s (a person for 1 s), and keeps its ID longer.
TEST_F(TrackerTest, WeaklySeenFaceStaysVisibleLonger) {
    Tracker tracker(classes);
    for (int f = 0; f < 5; ++f) step(tracker, {det(300, 300, classes.face())});
    for (int f = 0; f < 45; ++f) step(tracker, {weakDet(300, 300, classes.face())});  // 1.5 s
    EXPECT_EQ(visibleIds(tracker).size(), 1u);
    for (int f = 0; f < 30; ++f) step(tracker, {weakDet(300, 300, classes.face())});  // 2.5 s
    EXPECT_TRUE(visibleIds(tracker).empty()) << "hidden after 2 s of weak detections only";
    for (int f = 0; f < 3; ++f) step(tracker, {det(300, 300, classes.face())});
    EXPECT_EQ(seen_.size(), 1u) << "a confident detection brings back the same track";
}

// Two heavily overlapping boxes on one object (IoU above duplicateIou) start only one track.
TEST_F(TrackerTest, DuplicateBoxesMakeOneTrack) {
    Tracker tracker(classes);
    for (int f = 0; f < 10; ++f)
        step(tracker, {det(100, 100, kPerson, 100), det(104, 102, kPerson, 100)});
    EXPECT_EQ(seen_.size(), 1u);
}

TEST_F(TrackerTest, TwoSeparateObjectsKeepTheirIds) {
    Tracker tracker(classes);
    for (int f = 0; f < 40; ++f)
        step(tracker, {det(100 + 4.0f * f, 100), det(100 + 4.0f * f, 400)});
    EXPECT_EQ(seen_.size(), 2u);
}

// Two objects pass each other 30 px apart. Their ids must not swap: matching works with each
// track's predicted position, which carries on in its direction of travel.
TEST_F(TrackerTest, CrossingObjectsKeepTheirIds) {
    Tracker tracker(classes);
    for (int f = 0; f < 50; ++f)
        step(tracker,
             {det(100 + 8.0f * f, 200, kPerson, 60), det(500 - 8.0f * f, 230, kPerson, 60)});
    ASSERT_EQ(seen_.size(), 2u);
    for (int id : seen_) {
        const bool startedLeft = firstX_[id] < 300;
        EXPECT_EQ(startedLeft, lastX_[id] > 300) << "track " << id << " swapped objects";
    }
}

TEST_F(TrackerTest, NumbersCountPerClassFromOne) {
    Tracker tracker(classes);
    for (int f = 0; f < 5; ++f)
        step(tracker, {det(100, 100, classes.face()), det(400, 100, classes.face()),
                       det(600, 400, kPerson)});
    std::map<ClassId, std::set<int>> numbers;
    for (const Track& t : tracker.tracks())
        if (isVisible(t)) numbers[t.classId].insert(t.number);
    EXPECT_EQ(numbers[classes.face()], (std::set<int>{1, 2}));
    EXPECT_EQ(numbers[kPerson], (std::set<int>{1}));
}

TEST_F(TrackerTest, BriefFalseDetectionsDontUseNumbers) {
    Tracker tracker(classes);
    // One-frame blips in different places: never confirmed, never shown.
    for (int f = 0; f < 10; ++f) step(tracker, {det(50 + 90.0f * f, 50)});
    for (int f = 0; f < 5; ++f) step(tracker, {det(300, 400)});
    ASSERT_EQ(visibleIds(tracker).size(), 1u);
    for (const Track& t : tracker.tracks())
        if (isVisible(t)) EXPECT_EQ(t.number, 1);
}

// After the first person times out, the next one is #2, not #1 again.
TEST_F(TrackerTest, NumbersAreNotReused) {
    Tracker tracker(classes);
    for (int f = 0; f < 5; ++f) step(tracker, {det(100, 100)});
    for (int f = 0; f < 70; ++f) step(tracker, {});  // gone past the 2 s timeout
    for (int f = 0; f < 5; ++f) step(tracker, {det(100, 100)});
    ASSERT_EQ(tracker.tracks().size(), 1u);
    EXPECT_EQ(tracker.tracks().front().number, 2);
}

// The detector missing an object for a frame or two doesn't change how its box
// is drawn; a longer gap shows it as a prediction (thin), then hides it.
TEST_F(TrackerTest, BriefDropOutIsNotDrawnAsAPrediction) {
    Tracker tracker(classes);
    for (int f = 0; f < 5; ++f) step(tracker, {det(100, 100)});
    step(tracker, {});
    step(tracker, {});
    ASSERT_EQ(tracker.tracks().size(), 1u);
    EXPECT_FALSE(tracker.tracks().front().matched);
    EXPECT_FALSE(tracker.tracks().front().coasting) << "2 frames missing";
    step(tracker, {});
    step(tracker, {});
    EXPECT_TRUE(tracker.tracks().front().visible);
    EXPECT_TRUE(tracker.tracks().front().coasting) << "4 frames missing";
}

// Weak detections are often noise: they keep a track going but don't bring back
// one that has been gone for a while. A confident detection does.
TEST_F(TrackerTest, WeakDetectionsDontReviveALostTrack) {
    Tracker tracker(classes);
    for (int f = 0; f < 10; ++f) step(tracker, {det(300, 300, classes.face())});
    for (int f = 0; f < 30; ++f) step(tracker, {});  // 1 s
    for (int f = 0; f < 5; ++f) step(tracker, {weakDet(300, 300, classes.face())});
    EXPECT_TRUE(visibleIds(tracker).empty());
    step(tracker, {det(300, 300, classes.face())});
    EXPECT_EQ(visibleIds(tracker).size(), 1u);
    EXPECT_EQ(seen_.size(), 1u) << "same track as before";
}

// Fast back-and-forth motion: the shown box stays on the object instead of
// trailing behind it.
TEST_F(TrackerTest, FastBackAndForthBoxKeepsUp) {
    Tracker tracker(classes);
    double error = 0;
    int frames = 0;
    for (int f = 0; f < 90; ++f) {  // 3 s, twice a second, 150 px each way
        const float x = 500 + 150 * std::sin(2 * static_cast<float>(CV_PI) * 2 * f * kFrame);
        step(tracker, {det(x, 300, classes.face(), 120)});
        if (f < 10) continue;
        ASSERT_EQ(visibleIds(tracker).size(), 1u);
        error += std::abs(tracker.tracks().front().box.x - x);
        ++frames;
    }
    EXPECT_LT(error / frames, 6.0) << "mean px behind a 120 px face";
}
