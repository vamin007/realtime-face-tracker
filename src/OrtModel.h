#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

enum class Backend { CoreML, Cpu };

// One ONNX model run with ONNX Runtime. With Backend::CoreML the model runs on
// the GPU / Neural Engine through Apple's CoreML; anything CoreML can't run
// (or a failure to set it up) falls back to the CPU.
class OrtModel {
public:
    // cacheDir: where CoreML keeps its compiled copy of the model, so later
    // launches skip the compile.
    OrtModel(const std::string& path, Backend backend, const std::string& cacheDir);
    ~OrtModel();
    OrtModel(const OrtModel&) = delete;
    OrtModel& operator=(const OrtModel&) = delete;

    // Shape of the first input; -1 for a dynamic dimension.
    std::vector<int64_t> inputShape() const;

    // Runs the model on one float input (an N-d CV_32F Mat shaped like the input)
    // and returns every output as an N-d CV_32F Mat.
    std::vector<cv::Mat> run(const cv::Mat& input);

    // "CoreML (312/315 nodes)", "CoreML", or "CPU".
    const std::string& backendInfo() const { return backendInfo_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string backendInfo_;
};
