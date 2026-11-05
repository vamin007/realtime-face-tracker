#include <gtest/gtest.h>

#include <filesystem>

#include "OrtModel.h"

namespace {

const std::string kModels = PROJECT_DIR "/models/";
const std::string kCache = PROJECT_DIR "/.cache/coreml";

}  // namespace

// These load the real models; run scripts/fetch_models.sh first.
TEST(OrtModel, Yolo16by9ModelShapesOnCpu) {
    const std::string path = kModels + "yolov8n_384x640.onnx";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "missing " << path;
    OrtModel model(path, Backend::Cpu, kCache);
    EXPECT_EQ(model.inputShape(), (std::vector<int64_t>{1, 3, 384, 640}));
    EXPECT_EQ(model.backendInfo(), "CPU");

    const int shape[] = {1, 3, 384, 640};
    const cv::Mat input(4, shape, CV_32F, cv::Scalar(0.5));
    const std::vector<cv::Mat> out = model.run(input);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].total(), 84u * 5040);  // (4 box values + 80 class scores) per candidate
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
