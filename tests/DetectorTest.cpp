#include <gtest/gtest.h>

#include "HandDetector.h"
#include "YoloDetector.h"

namespace {

// Builds a fake YOLOv8 output [1, 4 + classes, boxes] from (cx, cy, w, h, class, score).
struct FakeBox {
    float cx, cy, w, h;
    int cls;
    float score;
};

cv::Mat fakeYoloOutput(int numClasses, const std::vector<FakeBox>& boxes) {
    const int attrs = 4 + numClasses, n = static_cast<int>(boxes.size());
    const int shape[] = {1, attrs, n};
    cv::Mat out(3, shape, CV_32F, cv::Scalar(0));
    float* p = out.ptr<float>();
    for (int i = 0; i < n; ++i) {
        p[0 * n + i] = boxes[i].cx;
        p[1 * n + i] = boxes[i].cy;
        p[2 * n + i] = boxes[i].w;
        p[3 * n + i] = boxes[i].h;
        p[(4 + boxes[i].cls) * n + i] = boxes[i].score;
    }
    return out;
}

const Detection* findClass(const std::vector<Detection>& dets, int cls) {
    for (const Detection& d : dets)
        if (d.classId == cls) return &d;
    return nullptr;
}

}  // namespace

TEST(DecodeYolo, ScalesBoxesBackToFramePixels) {
    const cv::Mat out = fakeYoloOutput(3, {{100, 50, 40, 20, 1, 0.9f}});
    const auto dets = decodeYolo(out, 0.5f, {1280, 720}, {0.4f});
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_EQ(dets[0].classId, 1);
    EXPECT_FLOAT_EQ(dets[0].box.x, 160);  // (100 - 20) / 0.5
    EXPECT_FLOAT_EQ(dets[0].box.y, 80);   // (50 - 10) / 0.5
    EXPECT_FLOAT_EQ(dets[0].box.width, 80);
    EXPECT_FLOAT_EQ(dets[0].box.height, 40);
    EXPECT_FALSE(dets[0].weak);
}

TEST(DecodeYolo, MergesDuplicatesAndMarksWeakDetections) {
    const cv::Mat out =
        fakeYoloOutput(3, {
                              {100, 100, 50, 50, 1, 0.90f},  // kept
                              {102, 101, 50, 50, 1, 0.80f},  // same class duplicate
                              {101, 100, 50, 52, 2, 0.85f},  // other class, same box
                              {400, 300, 60, 60, 0, 0.20f},  // weak, kept
                              {600, 300, 60, 60, 0, 0.05f},  // below low threshold
                          });
    const auto dets = decodeYolo(out, 1.0f, {1280, 720}, {0.4f, 0.1f, 0.45f});
    ASSERT_EQ(dets.size(), 2u);
    const Detection* strong = findClass(dets, 1);
    const Detection* weak = findClass(dets, 0);
    ASSERT_NE(strong, nullptr);
    ASSERT_NE(weak, nullptr);
    EXPECT_FLOAT_EQ(strong->conf, 0.90f);
    EXPECT_FALSE(strong->weak);
    EXPECT_TRUE(weak->weak);
    EXPECT_EQ(findClass(dets, 2), nullptr);
}

TEST(DecodeYolo, ClipsBoxesToTheFrame) {
    const cv::Mat out = fakeYoloOutput(1, {{5, 5, 40, 40, 0, 0.9f}});
    const auto dets = decodeYolo(out, 1.0f, {100, 100}, {0.4f});
    ASSERT_EQ(dets.size(), 1u);
    EXPECT_FLOAT_EQ(dets[0].box.x, 0);
    EXPECT_FLOAT_EQ(dets[0].box.y, 0);
    EXPECT_FLOAT_EQ(dets[0].box.width, 25);
}

TEST(HandBoxFromPalm, ExtendsTowardTheFingers) {
    const cv::Rect2f palm(100, 100, 100, 100);  // center (150, 150)
    // Wrist below the palm center, middle finger base above it: fingers point up.
    const std::vector<cv::Point2f> up = {{150, 190}, {130, 115}, {150, 110}};
    const cv::Rect2f hand = handBoxFromPalm(palm, up);
    EXPECT_TRUE((hand & palm) == palm) << "hand box must contain the palm";
    const float aboveTop = palm.y - hand.y;                                     // finger side
    const float belowBottom = (hand.y + hand.height) - (palm.y + palm.height);  // wrist side
    EXPECT_GT(aboveTop, belowBottom);
    EXPECT_GT(hand.height, hand.width);
}

TEST(HandBoxFromPalm, FollowsTheHandDirection) {
    const cv::Rect2f palm(100, 100, 100, 100);
    // Wrist on the left, fingers pointing right.
    const std::vector<cv::Point2f> right = {{110, 150}, {185, 130}, {190, 150}};
    const cv::Rect2f hand = handBoxFromPalm(palm, right);
    const float rightSide = (hand.x + hand.width) - (palm.x + palm.width);
    const float leftSide = palm.x - hand.x;
    EXPECT_GT(rightSide, leftSide);
    EXPECT_GT(hand.width, hand.height);
}
