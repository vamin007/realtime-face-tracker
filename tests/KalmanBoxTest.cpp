#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "KalmanBox.h"

namespace {

constexpr double kFrame = 1.0 / 30;

}  // namespace

TEST(KalmanBox, StartsAtTheFirstBox) {
    const KalmanBox kf({100, 50, 40, 20}, {});
    EXPECT_FLOAT_EQ(kf.box().x, 100);
    EXPECT_FLOAT_EQ(kf.box().y, 50);
    EXPECT_FLOAT_EQ(kf.box().width, 40);
    EXPECT_FLOAT_EQ(kf.box().height, 20);
}

TEST(KalmanBox, LearnsAConstantVelocity) {
    KalmanBox kf({0, 100, 50, 50}, {});
    for (int f = 1; f <= 30; ++f) {  // 300 px/s to the right
        kf.predict(kFrame);
        kf.correct({10.0f * f, 100, 50, 50});
    }
    EXPECT_NEAR(kf.velocity().x, 300, 15);
    EXPECT_NEAR(kf.velocity().y, 0, 15);
    kf.predict(0.1);  // a tenth of a second ahead: about 30 px further
    EXPECT_NEAR(kf.box().x, 330, 5);
}

TEST(KalmanBox, UsesRealTimeBetweenFrames) {
    KalmanBox a({0, 0, 50, 50}, {}), b({0, 0, 50, 50}, {});
    for (int f = 1; f <= 30; ++f) {
        a.predict(kFrame);
        a.correct({10.0f * f, 0, 50, 50});  // 10 px every 1/30 s
        if (f % 2 == 0) {
            b.predict(2 * kFrame);
            b.correct({10.0f * f, 0, 50, 50});  // same motion seen at 15 FPS
        }
    }
    EXPECT_NEAR(a.velocity().x, b.velocity().x, 20) << "speed must not depend on the frame rate";
}

TEST(KalmanBox, DampingSlowsItDown) {
    KalmanBox kf({0, 100, 50, 50}, {});
    for (int f = 1; f <= 30; ++f) {
        kf.predict(kFrame);
        kf.correct({10.0f * f, 100, 50, 50});
    }
    const float before = kf.velocity().x;
    kf.dampVelocity(0.5f);
    EXPECT_NEAR(kf.velocity().x, before / 2, 1e-3);
}

// A detection far from the filtered box (the object sped up or turned): the box
// follows the detection instead of trailing behind it.
TEST(KalmanBox, FollowsADetectionFarFromItsEstimate) {
    KalmanBox kf({0, 100, 50, 50}, {});
    for (int f = 1; f <= 10; ++f) {
        kf.predict(kFrame);
        kf.correct({0, 100, 50, 50});
    }
    kf.predict(kFrame);
    kf.correct({30, 100, 50, 50});  // 0.6 box sizes away
    EXPECT_NEAR(kf.box().x, 30, 1);
}

// Small differences are detection jitter: the box moves less than the detections.
TEST(KalmanBox, SmoothsSmallJitter) {
    KalmanBox kf({0, 100, 50, 50}, {});
    float lo = 1e9f, hi = -1e9f;
    for (int f = 1; f <= 40; ++f) {
        kf.predict(kFrame);
        kf.correct({f % 2 ? 2.0f : 0.0f, 100, 50, 50});  // +-1 px around x = 1
        if (f > 20) lo = std::min(lo, kf.box().x), hi = std::max(hi, kf.box().x);
    }
    EXPECT_LT(hi - lo, 1.0f) << "detections jump 2 px every frame";
}

// Fast back-and-forth motion: the box stays on the detections, where a plain
// constant-velocity filter trails behind.
TEST(KalmanBox, KeepsUpWithFastBackAndForthMotion) {
    KalmanBox::Params plain;
    plain.snapStart = plain.snapFull = 1e9f;  // never follows the detection
    KalmanBox snapping({0, 300, 50, 50}, {}), smoothOnly({0, 300, 50, 50}, plain);
    double snapError = 0, plainError = 0;
    for (int f = 1; f <= 60; ++f) {  // 2 s, waving twice a second, 100 px each way
        const float x = 100 * std::sin(2 * static_cast<float>(CV_PI) * 2 * f * kFrame);
        for (KalmanBox* kf : {&snapping, &smoothOnly}) {
            kf->predict(kFrame);
            kf->correct({x, 300, 50, 50});
        }
        snapError += std::abs(snapping.box().x - x) / 60;
        plainError += std::abs(smoothOnly.box().x - x) / 60;
    }
    EXPECT_LT(snapError, 3.0) << "mean px behind the object";
    EXPECT_LT(snapError, plainError / 2);
}

// A weak detection counts as noisier: a small difference (jitter) moves the box
// less than from a confident one, and it takes a bigger one to snap to it.
TEST(KalmanBox, NoisierDetectionsMoveItLess) {
    KalmanBox confident({0, 100, 50, 50}, {});
    for (int f = 1; f <= 10; ++f) {
        confident.predict(kFrame);
        confident.correct({0, 100, 50, 50});
    }
    KalmanBox weak = confident;
    confident.predict(kFrame);
    confident.correct({3, 100, 50, 50});
    weak.predict(kFrame);
    weak.correct({3, 100, 50, 50}, 1.5f);
    EXPECT_GT(weak.box().x, 0.0f);
    EXPECT_LT(weak.box().x, confident.box().x - 0.2f);
}
