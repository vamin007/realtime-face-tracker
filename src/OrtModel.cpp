#include "OrtModel.h"

#include <coreml_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <unordered_map>

namespace {

// ONNX Runtime reports how many graph nodes CoreML took in a log message; this
// keeps the latest one so it can be shown next to the model.
std::mutex gLogMutex;
std::string gCoremlSummary;

void ORT_API_CALL onOrtLog(void*, OrtLoggingLevel severity, const char*, const char*, const char*,
                           const char* message) {
    const std::string text = message ? message : "";
    if (text.find("number of nodes supported by CoreML") != std::string::npos) {
        std::lock_guard<std::mutex> lock(gLogMutex);
        gCoremlSummary = text;
        return;
    }
    if (severity >= ORT_LOGGING_LEVEL_ERROR) std::cerr << "onnxruntime: " << text << '\n';
}

Ort::Env& ortEnv() {
    // INFO so the CoreML summary reaches onOrtLog; other messages are filtered there.
    static Ort::Env env(ORT_LOGGING_LEVEL_INFO, "tracker", onOrtLog, nullptr);
    return env;
}

struct Coverage {
    int supported = -1;  // -1: unknown (no summary was logged)
    int total = -1;
    bool full() const { return supported < 0 || supported == total; }
    std::string text() const {
        return supported < 0 ? ""
                             : std::to_string(supported) + "/" + std::to_string(total) + " nodes";
    }
};

// Parses "... nodes in the graph: 315 number of nodes supported by CoreML: 312".
Coverage coverage(const std::string& summary) {
    static const std::regex re(
        R"(number of nodes in the graph:\s*(\d+).*number of nodes supported by CoreML:\s*(\d+))");
    std::smatch m;
    if (!std::regex_search(summary, m, re)) return {};
    return {std::stoi(m[2].str()), std::stoi(m[1].str())};
}

// A cache folder per model version: CoreML doesn't notice when a model file
// changes, so the file's size and modification time are part of the name.
std::string cacheFolder(const std::string& base, const std::string& modelPath) {
    namespace fs = std::filesystem;
    const auto size = static_cast<unsigned long long>(fs::file_size(modelPath));
    // libc++ counts file times in a 128-bit integer; seconds fit in 64 bits.
    const auto mtime = static_cast<long long>(std::chrono::duration_cast<std::chrono::seconds>(
                                                  fs::last_write_time(modelPath).time_since_epoch())
                                                  .count());
    const fs::path dir = fs::path(base) / (fs::path(modelPath).stem().string() + "_" +
                                           std::to_string(size) + "_" + std::to_string(mtime));
    fs::create_directories(dir);
    return dir.string();
}

}  // namespace

struct OrtModel::Impl {
    Ort::Session session{nullptr};
    std::vector<std::string> inputNames, outputNames;
    std::vector<int64_t> inputShape;
};

OrtModel::OrtModel(const std::string& path, Backend backend, const std::string& cacheDir)
    : impl_(std::make_unique<Impl>()) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error("Model file not found: " + path +
                                 " (run scripts/fetch_models.sh)");

    auto baseOptions = [&] {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(2);  // detectors run side by side; leave cores for each other
        return options;
    };

    // CoreML has two model formats and some models only convert to one of them,
    // so try the newer one, then the older one, then plain CPU. CoreML is used
    // only if it takes the whole model: when part of it stays on the CPU, moving
    // data back and forth costs more than it saves (measured on the palm model:
    // 14.6 ms partly on CoreML vs 12.5 ms on the CPU).
    bool created = false;
    std::string whyCpu;
    if (backend == Backend::CoreML) {
        for (const std::string format : {"MLProgram", "NeuralNetwork"}) {
            try {
                Ort::SessionOptions options = baseOptions();
                options.AppendExecutionProvider("CoreML",
                                                {{kCoremlProviderOption_ModelFormat, format},
                                                 {kCoremlProviderOption_MLComputeUnits,
                                                  kCoremlProviderOption_MLComputeUnits_ALL},
                                                 {kCoremlProviderOption_ModelCacheDirectory,
                                                  cacheFolder(cacheDir, path) + "/" + format}});
                {
                    std::lock_guard<std::mutex> lock(gLogMutex);
                    gCoremlSummary.clear();
                }
                Ort::Session session(ortEnv(), path.c_str(), options);
                Coverage cov;
                {
                    std::lock_guard<std::mutex> lock(gLogMutex);
                    cov = coverage(gCoremlSummary);
                }
                if (!cov.full()) {
                    whyCpu = "CoreML could only take " + cov.text();
                    continue;
                }
                impl_->session = std::move(session);
                created = true;
                backendInfo_ =
                    "CoreML " + format + (cov.text().empty() ? "" : " (" + cov.text() + ")");
                break;
            } catch (const std::exception& e) {
                if (whyCpu.empty()) whyCpu = std::string("CoreML failed: ") + e.what();
            }
        }
    }
    if (!created) {
        impl_->session = Ort::Session(ortEnv(), path.c_str(), baseOptions());
        backendInfo_ = "CPU";
        if (!whyCpu.empty()) backendInfo_ += " (" + whyCpu.substr(0, 100) + ")";
    }

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < impl_->session.GetInputCount(); ++i)
        impl_->inputNames.push_back(impl_->session.GetInputNameAllocated(i, allocator).get());
    for (size_t i = 0; i < impl_->session.GetOutputCount(); ++i)
        impl_->outputNames.push_back(impl_->session.GetOutputNameAllocated(i, allocator).get());
    impl_->inputShape = impl_->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

OrtModel::~OrtModel() = default;

std::vector<int64_t> OrtModel::inputShape() const { return impl_->inputShape; }

std::vector<cv::Mat> OrtModel::run(const cv::Mat& input) {
    CV_Assert(input.type() == CV_32F && input.isContinuous());
    std::vector<int64_t> shape(input.size.p, input.size.p + input.dims);
    const Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(input.ptr<float>()), input.total(), shape.data(), shape.size());

    std::vector<const char*> inNames, outNames;
    for (const auto& n : impl_->inputNames) inNames.push_back(n.c_str());
    for (const auto& n : impl_->outputNames) outNames.push_back(n.c_str());
    std::vector<Ort::Value> outputs = impl_->session.Run(
        Ort::RunOptions{nullptr}, inNames.data(), &tensor, 1, outNames.data(), outNames.size());

    std::vector<cv::Mat> result;
    for (Ort::Value& out : outputs) {
        const auto info = out.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Unsupported (non-float) model output");
        const std::vector<int64_t> outShape = info.GetShape();
        std::vector<int> dims(outShape.begin(), outShape.end());
        if (dims.size() == 1) dims.insert(dims.begin(), 1);  // cv::Mat needs at least 2 dims
        // Copy out: the ONNX Runtime tensor is freed when `outputs` goes away.
        result.push_back(cv::Mat(static_cast<int>(dims.size()), dims.data(), CV_32F,
                                 out.GetTensorMutableData<float>())
                             .clone());
    }
    return result;
}
