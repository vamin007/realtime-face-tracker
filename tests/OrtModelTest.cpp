#include <gtest/gtest.h>

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

TEST(OrtModel, MissingFileHasAHelpfulError) {
    try {
        OrtModel model(kModels + "nope.onnx", Backend::Cpu, kCache);
        FAIL() << "expected an exception";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("fetch_models.sh"), std::string::npos);
    }
}
