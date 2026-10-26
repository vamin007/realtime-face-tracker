#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "Overlay.h"

namespace {

Track makeTrack(int id, int number, ClassId cls, const cv::Rect2f& box, bool visible = true) {
    Track t;
    t.id = id;
    t.number = number;
    t.classId = cls;
    t.conf = 0.8f;
    t.box = box;
    t.confirmed = t.matched = true;
    t.visible = visible;
    for (int i = 0; i < 10; ++i) t.trail.push_back({box.x + box.width / 2 + 5 * i, box.y + 50});
    return t;
}

// Pixels that are clearly red, the TARGET's color.
int redPixels(const cv::Mat& frame) {
    cv::Mat red;
    cv::inRange(frame, cv::Scalar(0, 0, 200), cv::Scalar(60, 60, 255), red);
    return cv::countNonZero(red);
}

class OverlayTest : public ::testing::Test {
protected:
    ClassRegistry classes;  // person, hand, face
    std::vector<Track> crowd() const {
        return {makeTrack(1, 1, classes.person(), {300, 150, 700, 570}),
                makeTrack(2, 1, classes.face(), {540, 200, 200, 260}),
                makeTrack(3, 1, classes.hand(), {310, 160, 180, 220}),
                makeTrack(4, 2, classes.hand(), {760, 170, 200, 240}),
                makeTrack(5, 3, classes.hand(), {0, 0, 120, 120}, false)};
    }
};

}  // namespace

TEST(Overlay, ViewKeysToggleTheirOption) {
    DrawOptions options;
    EXPECT_TRUE(applyViewKey('h', options));
    EXPECT_TRUE(options.help);
    EXPECT_TRUE(applyViewKey('h', options));
    EXPECT_FALSE(options.help);
    EXPECT_TRUE(applyViewKey('t', options));
    EXPECT_TRUE(options.trails);
    EXPECT_TRUE(applyViewKey('d', options));
    EXPECT_TRUE(options.debug);
    EXPECT_FALSE(applyViewKey('p', options));  // pause and quit are main's
    EXPECT_FALSE(applyViewKey('q', options));
    EXPECT_FALSE(applyViewKey(-1, options));  // no key
}

TEST(Overlay, TrackColorsAreNeverRed) {
    for (int id = 1; id <= 1000; ++id) {
        cv::Mat bgr(1, 1, CV_8UC3, trackColor(id)), hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        const int hue = hsv.at<cv::Vec3b>(0, 0)[0];  // 0..179; red at both ends
        EXPECT_GE(hue, 15) << id;
        EXPECT_LE(hue, 160) << id;
    }
}

TEST_F(OverlayTest, CountsVisibleTracksPerClass) {
    const auto counts = countByClass(crowd(), classes);
    const std::vector<std::pair<std::string, int>> expected = {
        {"face", 1}, {"hand", 2}, {"person", 1}};  // the hidden hand isn't counted
    EXPECT_EQ(counts, expected);
}

TEST_F(OverlayTest, OnlyTheTargetIsRed) {
    const cv::Mat empty(720, 1280, CV_8UC3, cv::Scalar(90, 90, 90));
    DrawOptions options;
    options.trails = options.debug = true;
    FrameInfo info;
    info.fps = 30;
    info.uptime = 10;
    info.recording = info.paused = true;

    // No target: there is something on screen, but nothing red.
    cv::Mat frame = empty.clone();
    Overlay overlay;
    overlay.draw(frame, crowd(), classes, options, info);
    EXPECT_GT(cv::norm(frame, empty, cv::NORM_L1), 0);
    EXPECT_EQ(redPixels(frame), 0);

    // A light comes on the face: it becomes the target and is drawn red.
    const cv::Rect2f face(540, 200, 200, 260);
    TargetSelector targeting(classes);
    std::vector<Track> tracks = {makeTrack(2, 1, classes.face(), face)};
    cv::Mat scene(720, 1280, CV_8UC3, cv::Scalar::all(150));
    scene(cv::Rect(face)).setTo(cv::Scalar::all(110));
    double now = 0;
    auto run = [&](int frames) {
        for (int i = 0; i < frames; ++i, now += 1.0 / 30) {
            tracks[0].lastSeen = tracks[0].lastStrong = now;
            targeting.update(scene, tracks, now);
        }
    };
    run(60);
    scene.setTo(cv::Scalar::all(110));
    scene(cv::Rect(face)).setTo(cv::Scalar::all(190));
    run(6);
    ASSERT_TRUE(targeting.target().active());
    options.targeting = &targeting;
    frame = empty.clone();
    overlay.draw(frame, crowd(), classes, options, info);
    const cv::Rect around(460, 100, 360, 440);  // the face's box, its marks and its tag
    EXPECT_GT(redPixels(frame(around)), 1000);
    cv::Mat outside = frame.clone();
    outside(around).setTo(cv::Scalar(90, 90, 90));
    outside(cv::Rect(440, 0, 400, 60)).setTo(cv::Scalar(90, 90, 90));  // "TARGET LOCKED"
    EXPECT_EQ(redPixels(outside), 0);
}

TEST_F(OverlayTest, DrawsOnAnyFrameSize) {
    DrawOptions options;
    options.trails = options.debug = options.help = true;
    FrameInfo info;
    info.uptime = 4.5;  // the hint is fading
    for (const cv::Size size : {cv::Size(1920, 1080), cv::Size(320, 240), cv::Size(64, 48)}) {
        cv::Mat frame(size, CV_8UC3, cv::Scalar::all(90));
        Overlay overlay;
        EXPECT_NO_THROW(overlay.draw(frame, crowd(), classes, options, info)) << size;
        options.help = false;
        EXPECT_NO_THROW(overlay.draw(frame, crowd(), classes, options, info)) << size;
    }
}
