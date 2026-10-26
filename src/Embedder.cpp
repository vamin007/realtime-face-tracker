#include "Embedder.h"

#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#include "IDetector.h"  // requireModelFile

namespace {

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

void mirrorDetection(int width, cv::Rect2f& box, std::vector<cv::Point2f>& keypoints) {
    box.x = width - box.x - box.width;
    for (cv::Point2f& p : keypoints) p.x = width - p.x;
    // YuNet order: right eye, left eye, nose, right mouth corner, left mouth corner.
    if (keypoints.size() >= 5) {
        std::swap(keypoints[0], keypoints[1]);
        std::swap(keypoints[3], keypoints[4]);
    }
}

void FaceEmbedder::load() {
    if (net_) return;
    requireModelFile(modelPath_);
    net_ = cv::FaceRecognizerSF::create(modelPath_, "");
    if (!net_) throw std::runtime_error("Failed to load face recognition model: " + modelPath_);
}

Embedding FaceEmbedder::embed(const cv::Mat& frame, const cv::Rect2f& box,
                              const std::vector<cv::Point2f>& keypoints) {
    if (clipped(box, frame).area() <= 0) return {};
    load();
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
