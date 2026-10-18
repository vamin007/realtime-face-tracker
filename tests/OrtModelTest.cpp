#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>

#include "OrtModel.h"

namespace {

const std::string kModels = PROJECT_DIR "/models/";
const std::string kCache = PROJECT_DIR "/.cache/coreml";

}  // namespace

// These load the real models; run scripts/fetch_models.sh first.
TEST(OrtModel, PalmModelShapesOnCpu) {
    const std::string path = kModels + "palm_detection_mediapipe_2023feb.onnx";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "missing " << path;
    OrtModel model(path, Backend::Cpu, kCache);
    EXPECT_EQ(model.inputShape(), (std::vector<int64_t>{1, 192, 192, 3}));
    EXPECT_EQ(model.backendInfo(), "CPU");

    const int shape[] = {1, 192, 192, 3};
    const cv::Mat input(4, shape, CV_32F, cv::Scalar(0));
    const std::vector<cv::Mat> out = model.run(input);
    ASSERT_EQ(out.size(), 2u);
    size_t a = out[0].total(), b = out[1].total();
    if (a < b) std::swap(a, b);
    EXPECT_EQ(a, 2016u * 18);  // boxes + keypoints per anchor
    EXPECT_EQ(b, 2016u);       // one score per anchor
}

TEST(OrtModel, Yolo16by9ModelRunsOnCoreML) {
    const std::string path = kModels + "yolov8n_384x640.onnx";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "missing " << path;
    OrtModel model(path, Backend::CoreML, kCache);
    EXPECT_EQ(model.inputShape(), (std::vector<int64_t>{1, 3, 384, 640}));
    EXPECT_EQ(model.backendInfo().rfind("CoreML", 0), 0u) << model.backendInfo();

    const int shape[] = {1, 3, 384, 640};
    const cv::Mat input(4, shape, CV_32F, cv::Scalar(0.5));
    const std::vector<cv::Mat> out = model.run(input);
    ASSERT_EQ(out.size(), 1u);
    ASSERT_EQ(out[0].dims, 3);
    EXPECT_EQ(out[0].size[1], 84);    // 4 box values + 80 class scores
    EXPECT_EQ(out[0].size[2], 5040);  // candidate boxes for a 384x640 input
}

// On the CPU, as the app runs it (CoreML takes 8-38 s to load it).
TEST(OrtModel, Dinov2GivesOneSummaryTokenPlusPatches) {
    const std::string path = kModels + "dinov2_small.onnx";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "missing " << path;
    const auto t0 = std::chrono::steady_clock::now();
    OrtModel model(path, Backend::Cpu, kCache,
                   {{"batch_size", 1}, {"num_channels", 3}, {"height", 224}, {"width", 224}});
    const auto t1 = std::chrono::steady_clock::now();
    const int shape[] = {1, 3, 224, 224};
    const cv::Mat input(4, shape, CV_32F, cv::Scalar(0.1));
    std::vector<cv::Mat> out = model.run(input);  // first run warms up
    const auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i) out = model.run(input);
    const auto t3 = std::chrono::steady_clock::now();
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].size[1], 257);  // 1 summary token + 16x16 patches
    EXPECT_EQ(out[0].size[2], 384);
    using ms = std::chrono::duration<double, std::milli>;
    std::printf("[ info ] DINOv2 on %s: load %.0f ms, first run %.0f ms, then %.1f ms per run\n",
                model.backendInfo().c_str(), ms(t1 - t0).count(), ms(t2 - t1).count(),
                ms(t3 - t2).count() / 5);
}

TEST(OrtModel, MissingFileHasAHelpfulError) {
    try {
        OrtModel model(kModels + "nope.onnx", Backend::Cpu, kCache);
        FAIL() << "expected an exception";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("fetch_models.sh"), std::string::npos);
    }
}
