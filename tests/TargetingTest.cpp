#include <gtest/gtest.h>

#include <algorithm>
#include <opencv2/imgproc.hpp>

#include "Targeting.h"

namespace {

constexpr double kFrame = 1.0 / 30;
const cv::Rect2f kFaceBox(540, 200, 200, 260);
// YuNet's 5 landmarks for kFaceBox: right eye, left eye, nose, mouth corners.
const std::vector<cv::Point2f> kLandmarks = {{600, 290}, {680, 290}, {640, 330}, {610, 380},
                                             {670, 380}};
const cv::Scalar kRoom(120, 125, 115), kSkin(120, 150, 200), kHair(35, 40, 50), kLips(110, 110, 175);

// A synthetic scene: a room, faces (skin shaped like a head, hair on top, red
// lips), and laser dots: a red glow with an overexposed white middle.
struct Scene {
    struct Face {
        cv::Rect2f box;
        cv::Scalar skin = kSkin;
    };
    std::vector<Face> faces = {{kFaceBox}};
    std::vector<std::pair<cv::Rect2f, cv::Scalar>> patches;  // drawn over the faces
    std::vector<cv::Point> dots;
    int dotRadius = 4;
    cv::Scalar dotColor = cv::Scalar(40, 40, 235);

    cv::Mat render() const {
        cv::Mat frame(720, 1280, CV_8UC3, kRoom);
        for (const Face& f : faces) {
            const cv::Point c(static_cast<int>(f.box.x + f.box.width / 2),
                              static_cast<int>(f.box.y + f.box.height / 2));
            const cv::Size axes(static_cast<int>(f.box.width * 0.46f),
                                static_cast<int>(f.box.height * 0.52f));
            cv::ellipse(frame, c, axes, 0, 0, 360, f.skin, cv::FILLED);
            // Hair over the top of the head, lips below the nose.
            cv::ellipse(frame, c - cv::Point(0, axes.height / 2), {axes.width, axes.height / 2}, 0,
                        180, 360, kHair, cv::FILLED);
            cv::ellipse(frame, c + cv::Point(0, static_cast<int>(f.box.height * 0.2f)),
                        {static_cast<int>(f.box.width * 0.15f), 7}, 0, 0, 360, kLips, cv::FILLED);
        }
        for (const auto& [box, color] : patches) frame(cv::Rect(box)).setTo(color);
        for (const cv::Point& d : dots) {
            cv::circle(frame, d, dotRadius, dotColor, cv::FILLED, cv::LINE_AA);
            cv::circle(frame, d, dotRadius / 2, cv::Scalar(250, 250, 255), cv::FILLED);
        }
        return frame;
    }
};

Track makeTrack(int id, ClassId cls, const cv::Rect2f& box,
                const std::vector<cv::Point2f>& keypoints = {}) {
    Track t;
    t.id = id;
    t.classId = cls;
    t.conf = 0.9f;
    t.box = box;
    t.keypoints = keypoints;
    t.confirmed = t.visible = t.matched = true;
    return t;
}

// Landmarks of kFaceBox moved to another box of the same size.
std::vector<cv::Point2f> landmarksAt(const cv::Rect2f& box) {
    std::vector<cv::Point2f> moved;
    for (const cv::Point2f& p : kLandmarks) moved.push_back(p + (box.tl() - kFaceBox.tl()));
    return moved;
}

LaserDot find(const Scene& scene) {
    return findLaserDot(scene.render(), faceOval(kFaceBox, kLandmarks));
}

class TargetingTest : public ::testing::Test {
protected:
    // Runs `frames` frames of a scene and returns how many had a target. Tracks
    // count as confidently detected in every frame.
    int run(TargetSelector& selector, const Scene& scene, std::vector<Track> tracks, int frames) {
        const cv::Mat frame = scene.render();
        int targeted = 0;
        for (int i = 0; i < frames; ++i, ++frame_) {
            const double now = frame_ * kFrame;
            for (Track& t : tracks) t.lastSeen = t.lastStrong = now;
            if (selector.update(frame, tracks, now).active()) ++targeted;
        }
        return targeted;
    }

    ClassRegistry classes;  // person, face
    long long frame_ = 0;
};

}  // namespace

TEST(FaceOval, FollowsTheLandmarksFromForeheadToChin) {
    const cv::RotatedRect oval = faceOval(kFaceBox, kLandmarks);
    EXPECT_NEAR(oval.center.x, 640, 1);
    EXPECT_NEAR(oval.size.width, 160, 1);  // cheek to cheek: twice the eye distance
    EXPECT_NEAR(oval.size.height, 252, 1);  // eyes-to-mouth 90: 90 above, 72 below the mouth
    EXPECT_NEAR(oval.angle, 0, 0.5);
    // The box's corners (hair, background) are outside it (the oval isn't tilted).
    for (const cv::Point2f corner : {kFaceBox.tl(), kFaceBox.br(),
                                     cv::Point2f(kFaceBox.x, kFaceBox.y + kFaceBox.height)}) {
        const cv::Point2f d = corner - oval.center;
        const float a = oval.size.width / 2, b = oval.size.height / 2;
        EXPECT_GT(d.x * d.x / (a * a) + d.y * d.y / (b * b), 1) << corner;
    }
}

TEST(FaceOval, TiltsWithTheEyesWhetherOrNotTheImageIsMirrored) {
    std::vector<cv::Point2f> tilted = kLandmarks;
    tilted[1].y += 20;  // head tilted
    EXPECT_NEAR(faceOval(kFaceBox, tilted).angle, 14.0, 0.5);
    std::swap(tilted[0], tilted[1]);  // mirrored: the right eye is on the right
    std::swap(tilted[3], tilted[4]);
    EXPECT_NEAR(faceOval(kFaceBox, tilted).angle, 14.0, 0.5);
}

TEST(FaceOval, WithoutLandmarksItIsInsideTheBox) {
    const cv::RotatedRect oval = faceOval(kFaceBox, {});
    EXPECT_NEAR(oval.center.x, 640, 1);
    EXPECT_LT(oval.size.width, kFaceBox.width);
    EXPECT_LT(oval.size.height, kFaceBox.height);
}

TEST(LaserDot, ADotOnTheCheekIsFound) {
    Scene scene;
    scene.dots = {{685, 345}};
    const LaserDot dot = find(scene);
    ASSERT_TRUE(dot.found);
    EXPECT_NEAR(dot.center.x, 685, 1.5);
    EXPECT_NEAR(dot.center.y, 345, 1.5);
    EXPECT_GT(dot.redness, 60);
    EXPECT_GT(dot.skinAround, 0.9f);
}

TEST(LaserDot, ADotOnTheForeheadNoseOrChinIsFound) {
    for (const cv::Point p : {cv::Point(640, 275), cv::Point(640, 330), cv::Point(640, 425)}) {
        Scene scene;
        scene.dots = {p};
        EXPECT_TRUE(find(scene).found) << p;
    }
}

TEST(LaserDot, ADotInTheBoxButNotOnTheFaceIsIgnored) {
    // The wall in the box's corners, the hair, and just past the cheek.
    for (const cv::Point p : {cv::Point(552, 212), cv::Point(728, 445), cv::Point(640, 215),
                              cv::Point(600, 225), cv::Point(733, 330)}) {
        Scene scene;
        scene.dots = {p};
        EXPECT_FALSE(find(scene).found) << p;
    }
}

TEST(LaserDot, NoDotNoMatterHowTheFaceIsLit) {
    Scene scene;  // red lips only
    EXPECT_FALSE(find(scene).found);
    scene.faces[0].skin = cv::Scalar(225, 240, 255);  // a flashlight on the face
    EXPECT_FALSE(find(scene).found);
    scene.faces[0].skin = cv::Scalar::all(255);  // overexposed by it
    EXPECT_FALSE(find(scene).found);
    scene.faces[0].skin = cv::Scalar(70, 110, 190);  // warm lamp light: very red skin
    EXPECT_FALSE(find(scene).found);
}

TEST(LaserDot, RedThingsThatArentADotAreIgnored) {
    Scene scene;
    scene.patches = {{cv::Rect2f(600, 330, 60, 60), cv::Scalar(30, 30, 230)}};  // a red sticker
    EXPECT_FALSE(find(scene).found) << "too big";
    scene.patches = {{cv::Rect2f(590, 400, 70, 4), cv::Scalar(30, 30, 230)}};  // a red streak
    EXPECT_FALSE(find(scene).found) << "not round";
}

TEST(LaserDot, WorksOnDarkSkinAndForSmallOrLargeDots) {
    Scene scene;
    scene.faces[0].skin = cv::Scalar(55, 75, 105);
    scene.dots = {{685, 345}};
    scene.dotColor = cv::Scalar(40, 40, 210);
    EXPECT_TRUE(find(scene).found) << "dark skin";
    scene = Scene{};
    scene.dots = {{685, 345}};
    for (const int radius : {2, 4, 9}) {
        scene.dotRadius = radius;
        EXPECT_TRUE(find(scene).found) << "radius " << radius;
    }
}

TEST_F(TargetingTest, ALaserOnAFaceMakesItTheTarget) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene scene;
    EXPECT_EQ(run(selector, scene, tracks, 30), 0);
    scene.dots = {{685, 345}};
    run(selector, scene, tracks, 5);  // 0.17 s
    ASSERT_TRUE(selector.target().active()) << "TARGET within 0.2 s of the dot";
    EXPECT_EQ(selector.target().trackId, 1);
    EXPECT_NEAR(selector.target().dot.x, 685, 1.5);
    // The laser moves around on the face: still the target.
    scene.dots = {{620, 300}};
    EXPECT_EQ(run(selector, scene, tracks, 10), 10);
    // Laser off: released within 0.4 s.
    scene.dots.clear();
    run(selector, scene, tracks, 12);
    EXPECT_FALSE(selector.target().active());
}

TEST_F(TargetingTest, AFlashOfTheDotIsNotEnough) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene lit, dark;
    lit.dots = {{685, 345}};
    for (int i = 0; i < 10; ++i) {  // one frame with the dot, five without, ...
        EXPECT_EQ(run(selector, lit, tracks, 1), 0);
        EXPECT_EQ(run(selector, dark, tracks, 5), 0);
    }
}

TEST_F(TargetingTest, ABriefMissDoesNotDropTheTarget) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene lit, dark;
    lit.dots = {{685, 345}};
    run(selector, lit, tracks, 10);
    EXPECT_EQ(run(selector, dark, tracks, 3), 3) << "the camera missed the dot for 0.1 s";
    EXPECT_EQ(run(selector, lit, tracks, 5), 5);
}

TEST_F(TargetingTest, AFlashlightIsNotALaser) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene scene;
    run(selector, scene, tracks, 30);
    scene.faces[0].skin = cv::Scalar(225, 240, 255);
    EXPECT_EQ(run(selector, scene, tracks, 60), 0);
}

TEST_F(TargetingTest, ALaserBesideTheHeadIsNotATarget) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene scene;
    scene.dots = {{552, 212}};  // on the wall, inside the face's box
    EXPECT_EQ(run(selector, scene, tracks, 60), 0);
}

TEST_F(TargetingTest, OnlyFacesCanBeTargets) {
    TargetSelector selector(classes);
    const cv::Rect2f personBox(900, 150, 300, 550);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks),
                                       makeTrack(2, classes.person(), personBox)};
    Scene scene;
    scene.patches = {{personBox, kSkin}};
    scene.dots = {{1050, 400}};  // the laser on someone's shirt
    EXPECT_EQ(run(selector, scene, tracks, 60), 0);
}

TEST_F(TargetingTest, TheTargetFollowsTheLaserToAnotherFace) {
    TargetSelector selector(classes);
    const cv::Rect2f otherBox(100, 200, 200, 260);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks),
                                       makeTrack(2, classes.face(), otherBox, landmarksAt(otherBox))};
    Scene scene;
    scene.faces.push_back({otherBox});
    scene.dots = {{685, 345}};
    run(selector, scene, tracks, 10);
    EXPECT_EQ(selector.target().trackId, 1);
    scene.dots = {{245, 345}};  // the same spot on the other face
    run(selector, scene, tracks, 10);
    EXPECT_EQ(selector.target().trackId, 2);
}

TEST_F(TargetingTest, AFaceWithoutAFreshDetectionIsNotLookedAt) {
    TargetSelector selector(classes);
    std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    tracks[0].matched = false;  // a predicted box: its landmarks may be off the face
    Scene scene;
    scene.dots = {{685, 345}};
    EXPECT_EQ(run(selector, scene, tracks, 30), 0);
    EXPECT_FALSE(selector.faces().at(1).checked);
}
