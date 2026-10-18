#include <gtest/gtest.h>

#include <filesystem>
#include <opencv2/imgproc.hpp>

#include "Embedder.h"

namespace {

const std::string kModels = PROJECT_DIR "/models/";

// A frame with a few distinct shapes, for embedding sanity checks.
cv::Mat testFrame() {
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(90, 120, 150));
    cv::rectangle(frame, {100, 100, 160, 220}, {20, 200, 240}, cv::FILLED);
    cv::circle(frame, {180, 210}, 50, {200, 40, 40}, cv::FILLED);
    cv::rectangle(frame, {400, 150, 150, 150}, {30, 30, 30}, cv::FILLED);
    cv::line(frame, {400, 150}, {550, 300}, {255, 255, 255}, 8);
    return frame;
}

}  // namespace

TEST(Embedder, SimilarityOfUnitVectors) {
    EXPECT_FLOAT_EQ(similarity({1, 0}, {1, 0}), 1);
    EXPECT_FLOAT_EQ(similarity({1, 0}, {0, 1}), 0);
    EXPECT_FLOAT_EQ(similarity({1, 0}, {1, 0, 0}), 0) << "different sizes don't compare";
}

TEST(Embedder, MirroringSwapsFaceSides) {
    cv::Rect2f box(100, 50, 40, 60);
    // right eye, left eye, nose, right mouth corner, left mouth corner
    std::vector<cv::Point2f> kp = {{110, 60}, {130, 60}, {120, 75}, {112, 95}, {128, 95}};
    mirrorDetection(200, box, kp, true);
    EXPECT_FLOAT_EQ(box.x, 60);    // 200 - 100 - 40
    EXPECT_FLOAT_EQ(kp[0].x, 70);  // the old left eye (130), mirrored
    EXPECT_FLOAT_EQ(kp[1].x, 90);  // the old right eye (110), mirrored
    EXPECT_FLOAT_EQ(kp[2].x, 80);
    EXPECT_FLOAT_EQ(kp[3].x, 72);  // the old left mouth corner (128), mirrored
}

TEST(Embedder, ObjectEmbeddingsAreUnitLengthAndTellThingsApart) {
    const std::string path = kModels + "dinov2_small.onnx";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "missing " << path;
    ObjectEmbedder embedder(path, Backend::Cpu, PROJECT_DIR "/.cache/coreml");
    const cv::Mat frame = testFrame();
    const Embedding a = embedder.embed(frame, {100, 100, 160, 220}, {});
    const Embedding a2 = embedder.embed(frame, {104, 96, 158, 224}, {});  // nearly the same box
    const Embedding b = embedder.embed(frame, {400, 150, 150, 150}, {});
    ASSERT_EQ(a.size(), 384u);
    EXPECT_NEAR(similarity(a, a), 1, 1e-4);
    EXPECT_GT(similarity(a, a2), 0.9f);
    EXPECT_LT(similarity(a, b), similarity(a, a2));
    EXPECT_TRUE(embedder.embed(frame, {10, 10, 3, 3}, {}).empty()) << "too small to embed";
}
