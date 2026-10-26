#include "YoloDetector.h"

#include <algorithm>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

YoloDetector::YoloDetector(const std::string& modelPath, int numClasses,
                           DetectorThresholds thresholds, Backend backend,
                           const std::string& cacheDir)
    : model_(std::make_unique<OrtModel>(modelPath, backend, cacheDir)),
      numClasses_(numClasses),
      th_(thresholds) {
    const std::vector<int64_t> shape = model_->inputShape();  // [1, 3, height, width]
    if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0)
        throw std::runtime_error("YOLO model needs a fixed [1, 3, H, W] input: " + modelPath);
    inputSize_ = cv::Size(static_cast<int>(shape[3]), static_cast<int>(shape[2]));
}

std::string YoloDetector::info() const {
    return std::to_string(inputSize_.width) + "x" + std::to_string(inputSize_.height) + ", " +
           model_->backendInfo();
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& frame) {
    // Letterbox: scale to fit, keep the aspect ratio, pad the rest with gray.
    const float scale = std::min(static_cast<float>(inputSize_.width) / frame.cols,
                                 static_cast<float>(inputSize_.height) / frame.rows);
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::Mat input(inputSize_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(input(cv::Rect(0, 0, resized.cols, resized.rows)));

    const cv::Mat blob =
        cv::dnn::blobFromImage(input, 1.0 / 255.0, inputSize_, cv::Scalar(), true, false);
    const cv::Mat out = model_->run(blob).at(0);
    if (out.dims != 3 || out.size[1] - 4 != numClasses_)
        throw std::runtime_error("YOLO model has " + std::to_string(out.size[1] - 4) +
                                 " classes, expected " + std::to_string(numClasses_) +
                                 " (COCO's, person first)");
    return decodeYolo(out, scale, frame.size(), th_);
}

std::vector<Detection> decodeYolo(const cv::Mat& output, float scale, cv::Size frameSize,
                                  const DetectorThresholds& th) {
    // One row per attribute (cx, cy, w, h, then one score per class), one column
    // per candidate box. Only the box rows and person's scores are read: boxes of
    // other classes never become candidates, so they never reach NMS.
    const int numBoxes = output.size[2];
    const float* rows = output.ptr<float>();
    auto row = [&](int attr) { return rows + static_cast<size_t>(attr) * numBoxes; };
    const float *cxs = row(0), *cys = row(1), *ws = row(2), *hs = row(3);
    const float* personScores = row(4 + kCocoPerson);

    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    for (int i = 0; i < numBoxes; ++i) {
        if (personScores[i] < th.low) continue;
        const float cx = cxs[i] / scale, cy = cys[i] / scale;
        const float w = ws[i] / scale, h = hs[i] / scale;
        boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);
        scores.push_back(personScores[i]);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, th.low, th.nms, keep);

    const cv::Rect2f bounds(0, 0, static_cast<float>(frameSize.width),
                            static_cast<float>(frameSize.height));
    std::vector<Detection> result;
    for (int k : keep) {
        const cv::Rect2f box = cv::Rect2f(boxes[k]) & bounds;
        if (box.area() <= 0) continue;
        Detection d;
        d.box = box;
        d.classId = kCocoPerson;
        d.conf = scores[k];
        d.weak = scores[k] < th.conf;
        result.push_back(std::move(d));
    }
    return result;
}
