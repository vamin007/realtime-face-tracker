#include <gtest/gtest.h>

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
