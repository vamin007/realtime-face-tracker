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

TEST_F(OverlayTest, DrawsEachVisibleTrackInItsOwnColor) {
    const cv::Mat empty(720, 1280, CV_8UC3, cv::Scalar(90, 90, 90));
    cv::Mat frame = empty.clone();
    DrawOptions options;
    options.trails = options.debug = true;
    FrameInfo info;
    info.fps = 30;
    info.uptime = 10;
    info.recording = info.paused = true;
    Overlay overlay;
    overlay.draw(frame, crowd(), classes, options, info);
    EXPECT_GT(cv::norm(frame, empty, cv::NORM_L1), 0);

    // The middle of each visible box's left edge has the track's color. The
    // hidden hand's box would have its edge at the frame's left side.
    for (const Track& t : crowd()) {
        const cv::Point edge(static_cast<int>(t.box.x),
                             static_cast<int>(t.box.y + t.box.height / 2));
        const cv::Vec3b px = frame.at<cv::Vec3b>(edge);
        const cv::Scalar color = t.visible ? trackColor(t.id) : cv::Scalar(90, 90, 90);
        EXPECT_EQ(cv::Scalar(px[0], px[1], px[2]), color) << "track " << t.id;
    }
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
