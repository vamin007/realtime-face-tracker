// Tests for laser targeting: the face oval built from landmarks (FaceOval), the per-frame dot
// search (LaserDot) and the timing that turns it into a target (TargetingTest). The frames are
// synthetic. A Scene draws skin-colored heads (with hair and red lips) on a room background and
// adds laser dots as light, the way a camera records them, and red marks as pigment. That lets
// the tests say what must count as a dot (cheek, forehead, nose; dim, bright or dark skin) and
// what must not (wall, hair, lips, red stickers, flashlights).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

#include "Targeting.h"

namespace {

constexpr double kFrame = 1.0 / 30;
const cv::Rect2f kFaceBox(540, 200, 200, 260);
// YuNet's 5 landmarks for kFaceBox: right eye, left eye, nose, mouth corners.
const std::vector<cv::Point2f> kLandmarks = {{600, 290}, {680, 290}, {640, 330}, {610, 380},
                                             {670, 380}};
const cv::Scalar kRoom(120, 125, 115), kSkin(120, 150, 200), kHair(35, 40, 50), kLips(110, 110, 175);
const cv::Scalar kBrightSkin(170, 185, 220);  // a flashlight on the face: red at 220
const cv::Scalar kWall(150, 170, 205);        // a beige wall: as close to the skin as it gets
// How much of the skin's light a red mark reflects (B, G, R): a red sticker or
// a bindi, a fluorescent one, and a pink spot (a pimple) that is only redder.
const cv::Vec3f kRedMark(0.35f, 0.30f, 1.25f), kFluorescentMark(0.35f, 0.30f, 1.6f),
    kPinkSpot(1.0f, 1.0f, 1.3f);

// Conversions between 8-bit values and linear light (gamma 2.2), the space where light adds up.
float toLinear(int v) { return std::pow(v / 255.0f, 2.2f); }
uchar fromLinear(float v) {
    return cv::saturate_cast<uchar>(255 * std::pow(std::min(std::max(v, 0.0f), 1.0f), 1 / 2.2f));
}

// Changes the light around `c` in linear light: light(d2) is added to each
// channel times `add`, or the skin's light is multiplied by mix(d2) of `times`.
template <typename F>
void relight(cv::Mat& frame, cv::Point2f c, int r, F change) {
    for (int y = std::max(0, static_cast<int>(c.y) - r);
         y <= std::min(frame.rows - 1, static_cast<int>(c.y) + r); ++y)
        for (int x = std::max(0, static_cast<int>(c.x) - r);
             x <= std::min(frame.cols - 1, static_cast<int>(c.x) + r); ++x) {
            cv::Vec3b& p = frame.at<cv::Vec3b>(y, x);
            const float d2 = (x - c.x) * (x - c.x) + (y - c.y) * (y - c.y);
            for (int ch = 0; ch < 3; ++ch) p[ch] = fromLinear(change(toLinear(p[ch]), ch, d2));
        }
}

// A laser dot as a camera sees it: light added to the skin's, a narrow spot that
// saturates red and, through the sensor's crosstalk, a little green and blue
// (white in the middle when it's bright), plus a faint wide glow.
void addLaser(cv::Mat& frame, cv::Point2f c, float strength, float sigma) {
    const float crosstalk[3] = {0.05f, 0.08f, 1.0f};
    relight(frame, c, 20, [&](float v, int ch, float d2) {
        const float light =
            strength * 25 * std::exp(-d2 / (2 * sigma * sigma)) + 0.12f * std::exp(-d2 / 50);
        return v + light * crosstalk[ch];
    });
}

// A red mark (pigment): lit like the skin around it, reflecting more red and
// much less green and blue. Radius 4.5 px with a soft edge.
void addMark(cv::Mat& frame, cv::Point2f c, cv::Vec3f reflectance) {
    relight(frame, c, 7, [&](float v, int ch, float d2) {
        const float cover = std::min(std::max(5.0f - std::sqrt(d2), 0.0f), 1.0f);
        return v * (1 + cover * (reflectance[ch] - 1));
    });
}

// A synthetic scene: a room, faces (skin shaped like a head, hair on top, red
// lips), laser dots, and red marks.
struct Scene {
    struct Face {
        cv::Rect2f box;
        cv::Scalar skin = kSkin;
    };
    std::vector<Face> faces = {{kFaceBox}};
    std::vector<std::pair<cv::Rect2f, cv::Scalar>> patches;  // drawn over the faces
    std::vector<cv::Point2f> dots;
    float dotStrength = 1, dotSigma = 1.3f;
    std::vector<cv::Point2f> marks;
    cv::Vec3f markReflectance = kRedMark;

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
        for (const auto& [box, color] : patches)
            frame(cv::Rect(box) & cv::Rect(0, 0, frame.cols, frame.rows)).setTo(color);
        for (const cv::Point2f& d : dots) addLaser(frame, d, dotStrength, dotSigma);
        for (const cv::Point2f& m : marks) addMark(frame, m, markReflectance);
        return frame;
    }
};

Track makeTrack(int id, ClassId cls, const cv::Rect2f& box,
                const std::vector<cv::Point2f>& keypoints = {}) {
    Track t;
    t.id = id;
    t.classId = cls;
    t.conf = 0.9f;
    t.box = t.detected = box;
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

// Landmarks YuNet gives when hands cover the face: eyes almost above each
// other, the mouth far below them (eyes-to-mouth 2.9 eye distances).
const std::vector<cv::Point2f> kCoveredLandmarks = {
    {640, 250}, {648, 300}, {650, 340}, {640, 420}, {645, 425}};

// Renders the scene and searches kFaceBox for a dot, with the oval built from the landmarks.
LaserDot find(const Scene& scene, const std::vector<cv::Point2f>& landmarks = kLandmarks) {
    return findLaserDot(scene.render(), faceOval(kFaceBox, landmarks), kFaceBox);
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
    EXPECT_NEAR(oval.size.height, 234, 1);  // eyes-to-mouth 90: 90 above, 54 below the mouth
    EXPECT_NEAR(oval.angle, 0, 0.5);
    // The box's corners (hair, background) are outside it (the oval isn't tilted).
    for (const cv::Point2f corner : {kFaceBox.tl(), kFaceBox.br(),
                                     cv::Point2f(kFaceBox.x, kFaceBox.y + kFaceBox.height)}) {
        const cv::Point2f d = corner - oval.center;
        const float a = oval.size.width / 2, b = oval.size.height / 2;
        EXPECT_GT(d.x * d.x / (a * a) + d.y * d.y / (b * b), 1) << corner;
    }
}

// 20 px of drop over 80 px between the eyes tilts the oval by atan(20 / 80), about 14 degrees,
// and swapping the landmark order (a mirrored image) gives the same angle.
TEST(FaceOval, TiltsWithTheEyesWhetherOrNotTheImageIsMirrored) {
    std::vector<cv::Point2f> tilted = kLandmarks;
    tilted[1].y += 20;  // head tilted
    EXPECT_NEAR(faceOval(kFaceBox, tilted).angle, 14.0, 0.5);
    std::swap(tilted[0], tilted[1]);  // mirrored: the right eye is on the right
    std::swap(tilted[3], tilted[4]);
    EXPECT_NEAR(faceOval(kFaceBox, tilted).angle, 14.0, 0.5);
}

// The fallback: an oval inside the middle of the box.
TEST(FaceOval, WithoutLandmarksItIsInsideTheBox) {
    const cv::RotatedRect oval = faceOval(kFaceBox, {});
    EXPECT_NEAR(oval.center.x, 640, 1);
    EXPECT_LT(oval.size.width, kFaceBox.width);
    EXPECT_LT(oval.size.height, kFaceBox.height);
}

// Hands over the face make YuNet's landmarks go astray; an oval from them would
// be a sliver reaching off the face.
TEST(FaceOval, LandmarksNotShapedLikeAFaceAreIgnored) {
    const cv::RotatedRect fallback = faceOval(kFaceBox, {});
    std::vector<cv::Point2f> squashed = kLandmarks;  // the mouth up at the nose
    squashed[3].y = squashed[4].y = 320;
    for (const auto& landmarks : {kCoveredLandmarks, squashed}) {
        const cv::RotatedRect oval = faceOval(kFaceBox, landmarks);
        EXPECT_EQ(oval.center, fallback.center);
        EXPECT_EQ(oval.size, fallback.size);
        EXPECT_EQ(oval.angle, fallback.angle);
    }
}

// The basic case: a dot on the cheek is found where it is, with a clear red rise, a glow and a
// ring of skin around it.
TEST(LaserDot, ADotOnTheCheekIsFound) {
    Scene scene;
    scene.dots = {{685, 345}};
    const LaserDot dot = find(scene);
    ASSERT_TRUE(dot.found);
    EXPECT_NEAR(dot.center.x, 685, 1.5);
    EXPECT_NEAR(dot.center.y, 345, 1.5);
    EXPECT_GT(dot.redness, 25);  // red rose this much more than green and blue
    EXPECT_GE(dot.glow, 30);
    EXPECT_GT(dot.skinAround, 0.9f);
}

// Other places on the face (the third spot is near the chin).
TEST(LaserDot, ADotOnTheForeheadNoseOrChinIsFound) {
    for (const cv::Point p : {cv::Point(640, 275), cv::Point(640, 330), cv::Point(640, 412)}) {
        Scene scene;
        scene.dots = {p};
        EXPECT_TRUE(find(scene).found) << p;
    }
}

TEST(LaserDot, ADotInTheBoxButNotOnTheFaceIsIgnored) {
    // The wall in the box's corners, the hair, just past the cheek, and the
    // neck below the chin.
    for (const cv::Point p : {cv::Point(552, 212), cv::Point(728, 445), cv::Point(640, 215),
                              cv::Point(600, 225), cv::Point(733, 330), cv::Point(640, 452)}) {
        Scene scene;
        scene.dots = {p};
        EXPECT_FALSE(find(scene).found) << p;
    }
}

// Light only adds: on skin whose red is already at 220 a dot's red can rise by
// 35 at most, far less than on the face in normal light.
TEST(LaserDot, ADotOnBrightlyLitSkinIsFound) {
    Scene scene;
    scene.faces[0].skin = kBrightSkin;
    for (const cv::Point p : {cv::Point(685, 345), cv::Point(640, 275), cv::Point(640, 330)}) {
        scene.dots = {p};
        EXPECT_TRUE(find(scene).found) << p;
    }
    scene.dots = {{685, 345}};
    scene.dotStrength = 0.02f;  // too dim to turn white in the middle
    EXPECT_TRUE(find(scene).found) << "a dim dot";
    scene.faces[0].skin = cv::Scalar(190, 200, 232);  // red at 232: white in the middle
    scene.dotStrength = 1;
    EXPECT_TRUE(find(scene).found) << "very bright skin";
}

// With no dot, nothing is found whatever the lighting: red lips alone, a flashlight, an
// overexposed face, or warm lamp light on very red skin.
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

// A red mark the size of a dot (a bindi, a red sticker, a cherry angioma) is lit
// like the skin around it: it doesn't glow, and it is darker in green and blue.
TEST(LaserDot, AStaticRedMarkIsIgnored) {
    for (const cv::Scalar skin : {kSkin, kBrightSkin}) {
        for (const cv::Vec3f reflectance : {kRedMark, kFluorescentMark, kPinkSpot}) {
            Scene scene;
            scene.faces[0].skin = skin;
            scene.marks = {{685, 345}};
            scene.markReflectance = reflectance;
            EXPECT_FALSE(find(scene).found) << "skin " << skin << " reflectance " << reflectance;
        }
    }
}

// On skin this bright the red has almost no room left: lit lips (pink, a little
// darker in green and blue) look like a dot's red rim. A laser bright enough to
// see there is white in the middle; they aren't.
TEST(LaserDot, OnVeryBrightSkinLitLipsAreNotADot) {
    Scene scene;
    scene.faces[0].skin = cv::Scalar(190, 200, 232);
    scene.marks = {{685, 345}};
    scene.markReflectance = cv::Vec3f(0.9f, 0.85f, 1.3f);
    EXPECT_FALSE(find(scene).found);
}

// Dark skin leaves plenty of room for the red to rise. sigma is the spot's width in pixels.
TEST(LaserDot, WorksOnDarkSkinAndForSmallOrLargeDots) {
    Scene scene;
    scene.faces[0].skin = cv::Scalar(55, 75, 105);
    scene.dots = {{685, 345}};
    EXPECT_TRUE(find(scene).found) << "dark skin";
    scene = Scene{};
    scene.dots = {{685, 345}};
    for (const float sigma : {0.7f, 1.3f, 2.4f}) {
        scene.dotSigma = sigma;
        EXPECT_TRUE(find(scene).found) << "sigma " << sigma;
    }
}

// A bright, close laser: its overexposed middle is far wider than its red rim,
// and the whole of it belongs to the dot.
TEST(LaserDot, ADotWithAWideWhiteMiddleIsFound) {
    Scene scene;
    scene.dots = {{685, 345}};
    scene.dotStrength = 8;
    scene.dotSigma = 2.4f;
    const LaserDot dot = find(scene);
    ASSERT_TRUE(dot.found);
    EXPECT_NEAR(dot.center.x, 685, 1);
    EXPECT_GT(dot.area, 150);
}

// Where the oval reaches outside the face's box (the beige wall beside it), a
// dot doesn't count.
TEST(LaserDot, ADotOutsideTheBoxNeverCounts) {
    Scene scene;
    scene.patches = {{cv::Rect2f(745, 250, 150, 170), kWall}};
    scene.dots = {{790, 330}};
    const cv::RotatedRect sliver({720, 330}, {300, 50}, 0);  // from landmarks gone astray
    const cv::Mat frame = scene.render();
    EXPECT_TRUE(findLaserDot(frame, sliver, {540, 200, 400, 260}).found) << "if the box were wider";
    EXPECT_FALSE(findLaserDot(frame, sliver, kFaceBox).found);
}

// With the landmarks of a covered face the oval falls back to the box: a dot on the cheek still
// counts, a dot on the wall beside the face doesn't.
TEST(LaserDot, LandmarksGoneAstrayFallBackToTheBox) {
    Scene scene;
    scene.patches = {{cv::Rect2f(745, 250, 150, 170), kWall}};
    scene.dots = {{685, 345}};
    EXPECT_TRUE(find(scene, kCoveredLandmarks).found) << "the cheek is in the box's oval";
    scene.dots = {{790, 330}};
    EXPECT_FALSE(find(scene, kCoveredLandmarks).found) << "the wall beside the face";
}

// A box that starts left of the frame: the search area is cut to the frame, and a dot at the
// very edge doesn't crash anything.
TEST(LaserDot, AFacePartlyOutsideTheFrame) {
    const cv::Rect2f box(-60, 200, 200, 260);
    Scene scene;
    scene.faces = {{box}};
    scene.dots = {{85, 345}};  // the cheek, still in the frame
    const cv::Mat frame = scene.render();
    const LaserDot dot = findLaserDot(frame, faceOval(box, landmarksAt(box)), box);
    ASSERT_TRUE(dot.found);
    EXPECT_NEAR(dot.center.x, 85, 1.5);
    scene.dots = {{1, 330}};  // right at the edge
    EXPECT_NO_THROW(findLaserDot(scene.render(), faceOval(box, landmarksAt(box)), box));
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
    EXPECT_NEAR(selector.faces().at(1).dot.center.x, 685, 1.5);
    // The laser moves around on the face: still the target.
    scene.dots = {{620, 300}};
    EXPECT_EQ(run(selector, scene, tracks, 10), 10);
    // Laser off: released within 0.4 s.
    scene.dots.clear();
    run(selector, scene, tracks, 12);
    EXPECT_FALSE(selector.target().active());
}

// The same on skin with little room left for the red to rise.
TEST_F(TargetingTest, ALaserOnABrightlyLitFaceMakesItTheTarget) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene scene;
    scene.faces[0].skin = kBrightSkin;
    EXPECT_EQ(run(selector, scene, tracks, 30), 0);
    scene.dots = {{685, 345}};
    run(selector, scene, tracks, 5);
    EXPECT_TRUE(selector.target().active());
}

// A dot that appears for one frame at a time never lasts enterTime, so it never becomes the
// target.
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

// While the face isn't detected (a hand passes in front of it), the target
// stays; once the face is seen again without the dot, it is released after
// releaseTime of looking, not right away.
TEST_F(TargetingTest, TheTargetRidesOutAShortDetectionGap) {
    TargetSelector selector(classes);
    std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene scene;
    scene.dots = {{685, 345}};
    run(selector, scene, tracks, 10);
    ASSERT_TRUE(selector.target().active());
    tracks[0].matched = false;  // predicted for a while, then hidden: 0.5 s in all
    EXPECT_EQ(run(selector, scene, tracks, 6), 6);
    tracks[0].visible = false;
    EXPECT_EQ(run(selector, scene, tracks, 9), 9);
    tracks[0].matched = tracks[0].visible = true;
    scene.dots.clear();
    EXPECT_EQ(run(selector, scene, tracks, 9), 9) << "0.3 s of looking without the dot";
    run(selector, scene, tracks, 2);
    EXPECT_FALSE(selector.target().active());
}

// If the face is not detected at all, the target is kept for coastTime (0.75 s, 22 of the 30
// frames) and then released.
TEST_F(TargetingTest, TheTargetIsReleasedWhenItsFaceStaysUndetected) {
    TargetSelector selector(classes);
    std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene scene;
    scene.dots = {{685, 345}};
    run(selector, scene, tracks, 10);
    tracks[0].matched = false;
    EXPECT_EQ(run(selector, scene, tracks, 30), 22) << "kept for 0.75 s";
    EXPECT_FALSE(selector.target().active());
}

// The face suddenly gets much brighter (a flashlight): no dot, so no target.
TEST_F(TargetingTest, AFlashlightIsNotALaser) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene scene;
    run(selector, scene, tracks, 30);
    scene.faces[0].skin = cv::Scalar(225, 240, 255);
    EXPECT_EQ(run(selector, scene, tracks, 60), 0);
}

TEST_F(TargetingTest, AStaticRedMarkNeverBecomesTheTarget) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene scene;
    scene.marks = {{685, 345}};
    EXPECT_EQ(run(selector, scene, tracks, 60), 0);
    scene.markReflectance = kFluorescentMark;
    scene.faces[0].skin = kBrightSkin;
    EXPECT_EQ(run(selector, scene, tracks, 60), 0);
}

TEST_F(TargetingTest, ALaserBesideTheHeadIsNotATarget) {
    TargetSelector selector(classes);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks)};
    Scene scene;
    scene.dots = {{552, 212}};  // on the wall, inside the face's box
    EXPECT_EQ(run(selector, scene, tracks, 60), 0);
}

// A dot on a person track (not a face) never counts.
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

// When the dot moves to another face, that face becomes the target.
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

// No flip-flopping: while the dot is still on the current target, a second dot on another face
// doesn't take over.
TEST_F(TargetingTest, WithADotOnTwoFacesTheTargetStays) {
    TargetSelector selector(classes);
    const cv::Rect2f otherBox(100, 200, 200, 260);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), kFaceBox, kLandmarks),
                                       makeTrack(2, classes.face(), otherBox, landmarksAt(otherBox))};
    Scene scene;
    scene.faces.push_back({otherBox});
    scene.dots = {{685, 345}};
    run(selector, scene, tracks, 10);
    ASSERT_EQ(selector.target().trackId, 1);
    scene.dots = {{685, 345}, {245, 345}};  // a second laser on the other face
    for (int i = 0; i < 30; ++i) {
        run(selector, scene, tracks, 1);
        EXPECT_EQ(selector.target().trackId, 1) << "frame " << i;
    }
    EXPECT_TRUE(selector.faces().at(2).dot.found);
}

TEST_F(TargetingTest, AFacePartlyOutsideTheFrameCanBeTheTarget) {
    TargetSelector selector(classes);
    const cv::Rect2f box(-60, 200, 200, 260);
    const std::vector<Track> tracks = {makeTrack(1, classes.face(), box, landmarksAt(box))};
    Scene scene;
    scene.faces = {{box}};
    scene.dots = {{85, 345}};
    run(selector, scene, tracks, 6);
    EXPECT_TRUE(selector.target().active());
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
