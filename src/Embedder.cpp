#include "Embedder.h"

#include <cmath>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "IDetector.h"  // requireModelFile

namespace {

constexpr int kObjectInput = 224;  // DINOv2 input size (16 x 16 patches of 14 px)

Embedding normalized(const float* data, int n) {
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += static_cast<double>(data[i]) * data[i];
    const float inv = sum > 0 ? static_cast<float>(1 / std::sqrt(sum)) : 0;
    Embedding e(n);
    for (int i = 0; i < n; ++i) e[i] = data[i] * inv;
    return e;
}

cv::Rect clipped(const cv::Rect2f& box, const cv::Mat& frame) {
    return cv::Rect(box) & cv::Rect(0, 0, frame.cols, frame.rows);
}

}  // namespace

float similarity(const Embedding& a, const Embedding& b) {
    if (a.size() != b.size()) return 0;
    float dot = 0;
    for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
    return dot;
}

void mirrorDetection(int width, cv::Rect2f& box, std::vector<cv::Point2f>& keypoints, bool isFace) {
    box.x = width - box.x - box.width;
    for (cv::Point2f& p : keypoints) p.x = width - p.x;
    // YuNet order: right eye, left eye, nose, right mouth corner, left mouth corner.
    if (isFace && keypoints.size() >= 5) {
        std::swap(keypoints[0], keypoints[1]);
        std::swap(keypoints[3], keypoints[4]);
    }
}

FaceEmbedder::FaceEmbedder(const std::string& modelPath) {
    requireModelFile(modelPath);
    net_ = cv::FaceRecognizerSF::create(modelPath, "");
    if (!net_) throw std::runtime_error("Failed to load face recognition model: " + modelPath);
}

Embedding FaceEmbedder::embed(const cv::Mat& frame, const cv::Rect2f& box,
                              const std::vector<cv::Point2f>& keypoints) {
    if (clipped(box, frame).area() <= 0) return {};
    cv::Mat aligned;
    if (keypoints.size() >= 5) {
        // FaceRecognizerSF straightens the face from a YuNet-style row:
        // x, y, w, h, 5 landmark (x, y) pairs, score.
        cv::Mat row(1, 15, CV_32F);
        float* r = row.ptr<float>();
        r[0] = box.x;
        r[1] = box.y;
        r[2] = box.width;
        r[3] = box.height;
        for (int k = 0; k < 5; ++k) {
            r[4 + 2 * k] = keypoints[k].x;
            r[5 + 2 * k] = keypoints[k].y;
        }
        r[14] = 1;
        net_->alignCrop(frame, row, aligned);
    } else {
        cv::resize(frame(clipped(box, frame)), aligned, cv::Size(112, 112));
    }
    cv::Mat feature;
    net_->feature(aligned, feature);
    return normalized(feature.ptr<float>(), static_cast<int>(feature.total()));
}

ObjectEmbedder::ObjectEmbedder(const std::string& modelPath, Backend backend,
                               const std::string& cacheDir)
    : model_(std::make_unique<OrtModel>(modelPath, backend, cacheDir,
                                        std::map<std::string, int64_t>{{"batch_size", 1},
                                                                       {"num_channels", 3},
                                                                       {"height", kObjectInput},
                                                                       {"width", kObjectInput}})) {}

Embedding ObjectEmbedder::embed(const cv::Mat& frame, const cv::Rect2f& box,
                                const std::vector<cv::Point2f>&) {
    // Just the box (context made matching worse), padded to a square so the
    // object keeps its shape, then the ImageNet normalization DINOv2 was trained with.
    const cv::Rect roi = clipped(box, frame);
    if (roi.width < 8 || roi.height < 8) return {};
    const int side = std::max(roi.width, roi.height);
    cv::Mat square;
    cv::copyMakeBorder(frame(roi), square, (side - roi.height) / 2,
                       side - roi.height - (side - roi.height) / 2, (side - roi.width) / 2,
                       side - roi.width - (side - roi.width) / 2, cv::BORDER_CONSTANT,
                       cv::Scalar(124, 116, 104));  // the ImageNet mean color, in BGR
    cv::Mat resized, rgb, input;
    cv::resize(square, resized, cv::Size(kObjectInput, kObjectInput), 0, 0, cv::INTER_AREA);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(input, CV_32F, 1.0 / 255);
    cv::subtract(input, cv::Scalar(0.485, 0.456, 0.406), input);
    cv::divide(input, cv::Scalar(0.229, 0.224, 0.225), input);
    const cv::Mat blob = cv::dnn::blobFromImage(input);  // HWC -> NCHW

    // last_hidden_state [1, 1 + 16x16, 384]: a summary token, then one per image
    // patch. The average of the patch tokens told the same phone from other
    // objects better than the summary token (tested on the flashlight clip:
    // 0.965 vs 0.937 chance of ranking a same-phone pair above a different pair).
    const cv::Mat out = model_->run(blob).at(0);
    const int tokens = out.size[1], dims = out.size[2];
    const float* data = out.ptr<float>();
    std::vector<float> mean(dims, 0.0f);
    for (int t = 1; t < tokens; ++t)
        for (int k = 0; k < dims; ++k) mean[k] += data[t * dims + k];
    return normalized(mean.data(), dims);
}
