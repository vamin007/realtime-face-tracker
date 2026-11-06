// A thin wrapper around ONNX Runtime (ORT) that loads one .onnx model and runs it on a float
// tensor. Pipeline role: it is the inference engine under YoloDetector (people), so the heavy
// neural network can run on Apple's GPU / Neural Engine while the face detector uses the CPU.
// Key ideas:
//  - an "execution provider" is the hardware backend ORT hands graph nodes to (here CoreML or
//    the plain CPU one);
//  - CoreML is used only if it can run the *whole* graph: a model split between CoreML and the
//    CPU was measured to be slower than the CPU alone, so partial coverage falls back to CPU;
//  - this header includes no ONNX Runtime headers (pimpl idiom: they live in a hidden Impl
//    struct), so only OrtModel.cpp depends on them and the rest of the code compiles faster.
#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Which execution provider to try first. Cpu never touches CoreML.
enum class Backend { CoreML, Cpu };

// One ONNX model run with ONNX Runtime. With Backend::CoreML the model runs on
// the GPU / Neural Engine through Apple's CoreML; anything CoreML can't run
// (or a failure to set it up) falls back to the CPU.
// Each instance owns its session; it is used by one thread at a time (each detector has its own).
class OrtModel {
public:
    // cacheDir: where CoreML keeps its compiled copy of the model, so later
    // launches skip the compile.
    // Throws std::runtime_error if the file is missing (the message points at
    // scripts/fetch_models.sh); a CoreML failure is not an error, it falls back to the CPU.
    OrtModel(const std::string& path, Backend backend, const std::string& cacheDir);
    ~OrtModel();
    OrtModel(const OrtModel&) = delete;
    OrtModel& operator=(const OrtModel&) = delete;

    // Shape of the first input; -1 for a dynamic dimension.
    std::vector<int64_t> inputShape() const;

    // Runs the model on one float input (an N-d CV_32F Mat shaped like the input)
    // and returns every output as an N-d CV_32F Mat.
    // The input must be continuous in memory. The outputs are copies, owned by the caller.
    std::vector<cv::Mat> run(const cv::Mat& input);

    // e.g. "CoreML (315/315 nodes)", "CoreML", or "CPU" (CoreML only when it takes every node).
    // For the startup summary. The CoreML text also names the model format that worked, e.g.
    // "CoreML MLProgram (315/315 nodes)". When CoreML was tried but didn't work, "CPU" is
    // followed by the reason, e.g. "CPU (CoreML could only take 100/120 nodes)".
    const std::string& backendInfo() const { return backendInfo_; }

private:
    // Holds the ONNX Runtime session, so this header needn't include ORT.
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string backendInfo_;
};
