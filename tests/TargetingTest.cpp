#include <gtest/gtest.h>

#include <algorithm>

#include "Targeting.h"

namespace {

constexpr double kFrame = 1.0 / 30;
const cv::Rect2f kFaceBox(540, 200, 200, 260);

// A synthetic scene: a gray room with a face-colored box (and optional extras).
struct Scene {
    float room = 150;  // background brightness
    float face = 110;  // face brightness (a bit darker than a bright room, as measured)
    std::vector<std::pair<cv::Rect2f, float>> patches;  // other areas and their brightness

    cv::Mat render() const {
        cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar::all(room));
        frame(cv::Rect(kFaceBox)).setTo(cv::Scalar::all(face));
        for (const auto& [box, value] : patches) frame(cv::Rect(box)).setTo(cv::Scalar::all(value));
        return frame;
    }
};

Track makeTrack(int id, ClassId cls, const cv::Rect2f& box) {
    Track t;
    t.id = id;
    t.classId = cls;
    t.conf = 0.9f;
    t.box = box;
    t.confirmed = t.visible = t.matched = true;
    return t;
}

class TargetingTest : public ::testing::Test {
protected:
    // Runs `frames` frames of a scene and returns how many had a target. Tracks
    // count as confidently detected in every frame unless `weakIds` lists them.
    int run(TargetSelector& selector, const Scene& scene, std::vector<Track> tracks, int frames,
            const std::vector<int>& weakIds = {}) {
        const cv::Mat frame = scene.render();
        int targeted = 0;
        for (int i = 0; i < frames; ++i, ++frame_) {
            const double now = frame_ * kFrame;
            for (Track& t : tracks) {
                t.lastSeen = now;
                if (std::find(weakIds.begin(), weakIds.end(), t.id) == weakIds.end())
                    t.lastStrong = now;
            }
            if (selector.update(frame, tracks, now).active()) ++targeted;
        }
        return targeted;
    }

    ClassRegistry classes;  // person, hand, face
    long long frame_ = 0;
};

}  // namespace

TEST(Brightness, MeanLumaOfTheMiddleOfABox) {
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar::all(0));
    frame(cv::Rect(20, 20, 60, 60)).setTo(cv::Scalar::all(200));
    EXPECT_NEAR(meanLuma(frame, cv::Rect2f(0, 0, 100, 100), 0.6f), 200, 1);  // middle 60x60
    EXPECT_NEAR(meanLuma(frame, cv::Rect2f(0, 0, 100, 100), 1.0f), 72, 1);   // whole box
}

TEST(Brightness, BackgroundIgnoresExcludedBoxesAndSmallBrightSpots) {
    cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar::all(100));
    frame(cv::Rect(0, 0, 640, 720)).setTo(cv::Scalar::all(250));  // half the frame is a face
    EXPECT_NEAR(backgroundLuma(frame, {cv::Rect2f(0, 0, 640, 720)}), 100, 2);
    frame(cv::Rect(900, 100, 40, 40)).setTo(cv::Scalar::all(255));  // a flashlight's glare
    EXPECT_NEAR(backgroundLuma(frame, {cv::Rect2f(0, 0, 640, 720)}), 100, 2);
}

TEST_F(TargetingTest, LightOnAFaceMakesItTheTarget) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox)};
    Scene scene;
    EXPECT_EQ(run(selector, scene, tracks, 60), 0);  // 2 s of normal light
    // Flashlight on: the face gets bright and the camera darkens the rest.
    scene.face = 190;
    scene.room = 110;
    run(selector, scene, tracks, 6);  // 0.2 s
    EXPECT_TRUE(selector.target().active()) << "TARGET within 0.2 s of the light";
    EXPECT_EQ(selector.target().trackId, 1);
    // Light off: released within half a second.
    scene = Scene{};
    run(selector, scene, tracks, 15);
    EXPECT_FALSE(selector.target().active());
}

TEST_F(TargetingTest, AFaceThatIsAlwaysBrightIsNotATarget) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox)};
    Scene scene;
    scene.face = 200;  // e.g. lit by a monitor in a dark room, from the start
    scene.room = 80;
    EXPECT_EQ(run(selector, scene, tracks, 300), 0);
}

TEST_F(TargetingTest, RoomLightChangesDontMakeATarget) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox)};
    Scene scene;
    run(selector, scene, tracks, 60);
    scene.face = 160;  // a ceiling light comes on: everything gets brighter together
    scene.room = 220;
    EXPECT_EQ(run(selector, scene, tracks, 90), 0);
    scene.face = 60;  // and off again
    scene.room = 80;
    EXPECT_EQ(run(selector, scene, tracks, 90), 0);
}

TEST_F(TargetingTest, OnlyFacesCanBeTargets) {
    TargetSelector selector(classes);
    const cv::Rect2f personBox(900, 150, 300, 550), handBox(60, 300, 150, 180);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox),
                                       makeTrack(2, classes.person(), personBox),
                                       makeTrack(3, classes.hand(), handBox)};
    Scene scene;
    run(selector, scene, tracks, 60);
    // Someone else in a bright light, and a hand pointing a flashlight at the camera.
    scene.patches = {{personBox, 255}, {handBox, 255}};
    EXPECT_EQ(run(selector, scene, tracks, 90), 0);
}

TEST_F(TargetingTest, AHandInFrontOfTheFaceIsNotALight) {
    TargetSelector selector(classes);
    const cv::Rect2f handBox(560, 220, 170, 230);
    std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox)};
    Scene scene;
    run(selector, scene, tracks, 60);
    tracks.push_back(makeTrack(2, classes.hand(), handBox));
    scene.patches = {{handBox, 200}};  // a palm held up to the face, lit by a window
    EXPECT_EQ(run(selector, scene, tracks, 90), 0);
}

// The palm model sometimes weakly mistakes a face for a palm; such a "hand"
// mustn't stop the face from being measured.
TEST_F(TargetingTest, AWeakPhantomHandDoesNotBlockTheFace) {
    TargetSelector selector(classes);
    const cv::Rect2f phantom(560, 220, 170, 230);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox),
                                       makeTrack(2, classes.hand(), phantom)};
    Scene scene;
    run(selector, scene, tracks, 60, {2});
    scene.face = 190;
    scene.room = 110;
    run(selector, scene, tracks, 6, {2});
    EXPECT_TRUE(selector.target().active());
}

TEST_F(TargetingTest, BrightestOfTwoLitFacesWins) {
    TargetSelector selector(classes);
    const cv::Rect2f otherFace(100, 200, 200, 260);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox),
                                       makeTrack(2, classes.face(), otherFace)};
    Scene scene;
    scene.patches = {{otherFace, 110}};
    run(selector, scene, tracks, 60);
    scene.face = 170;
    scene.patches = {{otherFace, 210}};
    scene.room = 110;
    run(selector, scene, tracks, 10);
    EXPECT_EQ(selector.target().trackId, 2);
}

TEST_F(TargetingTest, ANewFaceNeedsHistoryFirst) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox)};
    Scene scene;
    run(selector, scene, tracks, 10);  // seen for a third of a second only
    scene.face = 190;
    scene.room = 110;
    EXPECT_EQ(run(selector, scene, tracks, 10), 0);
}
